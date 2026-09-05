//! equirectangular HDRI -> cubemap 変換ルーチン (wgpu, compute shader版)

pub struct EquirectToCubemap {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
}

impl EquirectToCubemap {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("equirect_to_cubemap_shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
        });

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("equirect_to_cubemap_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba16Float,
                            view_dimension: wgpu::TextureViewDimension::D2Array,
                        },
                        count: None,
                    },
                ],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("equirect_to_cubemap_pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("equirect_to_cubemap_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("equirect_sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        Self {
            pipeline,
            bind_group_layout,
            sampler,
        }
    }

    /// equirect_view: 元のHDRI画像のTextureView (2D, filterable float format)
    /// face_size: 出力キューブマップの1面あたりの解像度
    pub fn convert(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        equirect_view: &wgpu::TextureView,
        face_size: u32,
    ) -> wgpu::Texture {
        let cubemap = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("cubemap_texture"),
            size: wgpu::Extent3d {
                width: face_size,
                height: face_size,
                depth_or_array_layers: 6,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let storage_view = cubemap.create_view(&wgpu::TextureViewDescriptor {
            label: Some("cubemap_storage_view"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("equirect_to_cubemap_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(equirect_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&storage_view),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("equirect_to_cubemap_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("equirect_to_cubemap_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (face_size + 7) / 8;
            pass.dispatch_workgroups(workgroups, workgroups, 6);
        }
        queue.submit(Some(encoder.finish()));

        cubemap
    }

    /// 通常のcubemapとして使うためのビューを作るヘルパー
    pub fn create_cube_view(cubemap: &wgpu::Texture) -> wgpu::TextureView {
        cubemap.create_view(&wgpu::TextureViewDescriptor {
            label: Some("cubemap_cube_view"),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        })
    }
}

const SHADER_SRC: &str = r#"
@group(0) @binding(0) var equirect_tex: texture_2d<f32>;
@group(0) @binding(1) var equirect_sampler: sampler;
@group(0) @binding(2) var cubemap_out: texture_storage_2d_array<rgba16float, write>;

const PI: f32 = 3.14159265359;

// face index (0..5) と面内UV(-1..1)から、キューブの外向き方向ベクトルを求める
// 0:+X 1:-X 2:+Y 3:-Y 4:+Z 5:-Z (wgpuのcubemap面順)
fn face_dir(face: u32, uv: vec2<f32>) -> vec3<f32> {
    switch face {
        case 0u: { return normalize(vec3<f32>( 1.0, -uv.y, -uv.x)); }
        case 1u: { return normalize(vec3<f32>(-1.0, -uv.y,  uv.x)); }
        case 2u: { return normalize(vec3<f32>( uv.x,  1.0,  uv.y)); }
        case 3u: { return normalize(vec3<f32>( uv.x, -1.0, -uv.y)); }
        case 4u: { return normalize(vec3<f32>( uv.x, -uv.y,  1.0)); }
        default: { return normalize(vec3<f32>(-uv.x, -uv.y, -1.0)); }
    }
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(cubemap_out).xy;
    if (gid.x >= size.x || gid.y >= size.y) {
        return;
    }

    let face = gid.z;

    let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5, 0.5))
        / vec2<f32>(size) * 2.0 - vec2<f32>(1.0, 1.0);

    let dir = face_dir(face, uv);

    let phi = atan2(dir.z, dir.x);
    let theta = acos(clamp(dir.y, -1.0, 1.0));

    let sample_uv = vec2<f32>(
        phi / (2.0 * PI) + 0.5,
        theta / PI
    );

    let color = textureSampleLevel(equirect_tex, equirect_sampler, sample_uv, 0.0);
    textureStore(cubemap_out, vec2<i32>(gid.xy), i32(face), color);
}
"#;
