//! sky-baker: equirectangular HDRI (.exr) -> KTX2 cubemap (Rgba16Float, mips付き) を
//! 生成するスタンドアロンCLIツール。
//!
//! 実行例:
//!   cargo run --release -p sky-baker -- \
//!       --input sky.exr \
//!       --output rgba16f.ktx2 \
//!       --face-size 512

mod equirect_to_cubemap;
mod exr_loader;
mod ktx2_hdri_bake;

use clap::Parser;
use equirect_to_cubemap::EquirectToCubemap;

#[derive(Parser)]
#[command(about = "equirectangular HDRI (.exr) を KTX2 cubemap に焼くツール")]
struct Args {
    /// 入力HDRI (.exr)
    #[arg(short, long)]
    input: String,

    /// 出力KTX2ファイルパス
    #[arg(short, long)]
    output: String,

    /// キューブ1面あたりの解像度
    #[arg(long, default_value_t = 512)]
    face_size: u32,
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args = Args::parse();
    pollster::block_on(run(args))
}

async fn run(args: Args) -> anyhow::Result<()> {
    // --- wgpuデバイスをヘッドレスで初期化 (ウィンドウ・サーフェス不要) ---
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..Default::default()
    });

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .map_err(|e| anyhow::anyhow!("利用可能なGPUアダプタが見つかりません: {e:?}"))?;

    log::info!("adapter: {:?}", adapter.get_info());

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("sky_baker_device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
        })
        .await?;

    // --- HDRI (.exr) 読み込み ---
    log::info!("loading exr: {}", args.input);
    let exr = exr_loader::load_exr_equirect(&args.input)?;
    log::info!("exr resolution: {}x{}", exr.width, exr.height);

    let (_equirect_texture, equirect_view) = exr.to_wgpu_texture_f16(&device, &queue);

    // --- equirect -> cubemap -> ミップ生成 -> KTX2書き出し ---
    let converter = EquirectToCubemap::new(&device);

    log::info!(
        "baking cubemap: face_size={}, output={}",
        args.face_size,
        args.output
    );
    ktx2_hdri_bake::bake_equirect_to_ktx2(
        &device,
        &queue,
        &converter,
        &equirect_view,
        args.face_size,
        &args.output,
    )?;

    log::info!("done: {}", args.output);
    Ok(())
}
