//! .exr HDRIファイルを読み込むローダー。

use exr::prelude::*;

/// EXRから読み込んだ生データ。RGBA f32, 行優先, 左上原点(y=0が画像の上端)。
/// EXRはリニア値で格納されているため、ここではガンマ補正等は一切行わない。
pub struct LoadedExr {
    pub width: u32,
    pub height: u32,
    /// len = width * height * 4
    pub data: Vec<f32>,
}

/// Pixels用の一時ストレージ: (width, height, flat RGBA buffer)
type RawBuffer = (usize, usize, Vec<f32>);

/// .exr (equirectangular, RGB/RGBA) を読み込み、フラットなRGBA f32バッファへ展開する。
/// アルファチャンネルが無い場合は 1.0 で埋められる (exrクレートの仕様)。
pub fn load_exr_equirect(path: &str) -> anyhow::Result<LoadedExr> {
    let image: PixelImage<RawBuffer, RgbaChannels> = read_first_rgba_layer_from_file(
        path,
        |resolution, _channels| -> RawBuffer {
            let width = resolution.width();
            let height = resolution.height();
            (width, height, vec![0.0f32; width * height * 4])
        },
        |buffer: &mut RawBuffer, position: Vec2<usize>, (r, g, b, a): (f32, f32, f32, f32)| {
            let (width, _height, pixels) = buffer;
            let idx = (position.y() * *width + position.x()) * 4;
            pixels[idx] = r;
            pixels[idx + 1] = g;
            pixels[idx + 2] = b;
            pixels[idx + 3] = a;
        },
    )
    .map_err(|e| anyhow::anyhow!("failed to read exr: {e}"))?;

    let (width, height, data) = image.layer_data.channel_data.pixels;

    Ok(LoadedExr {
        width: width as u32,
        height: height as u32,
        data,
    })
}

impl LoadedExr {
    /// f32のまま Rgba32Float テクスチャとしてアップロードする。
    /// 線形補間サンプリングには wgpu::Features::FLOAT32_FILTERABLE が必要。
    pub fn to_wgpu_texture(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("equirect_hdri_texture_f32"),
            size: wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let bytes: &[u8] = bytemuck::cast_slice(self.data.as_slice());
        let bytes_per_row = 4 * std::mem::size_of::<f32>() as u32 * self.width;

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(self.height),
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    /// f32 -> f16 に変換してから Rgba16Float テクスチャとしてアップロードする。
    /// 追加Featureなしで線形フィルタリングが使えるので、こちらを推奨。
    pub fn to_wgpu_texture_f16(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        // f16のまま cast_slice すると half::f16 が Pod/NoUninit を実装していないため
        // コンパイルエラーになる。u16(ビットパターン)に変換してから cast_slice する。
        let half_bits: Vec<u16> = self
            .data
            .iter()
            .map(|&v| half::f16::from_f32(v).to_bits())
            .collect();

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("equirect_hdri_texture_f16"),
            size: wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let bytes: &[u8] = bytemuck::cast_slice(half_bits.as_slice());
        let bytes_per_row = 4 * std::mem::size_of::<u16>() as u32 * self.width;

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(self.height),
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }
}
