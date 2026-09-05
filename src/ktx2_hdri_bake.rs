//! GPU上で生成したcubemap (Rgba16Float) をミップ付きKTX2ファイルとして書き出す。
//!
//! 出力ファイルは、下記のような既存の読み込みコードとバイト互換になるよう設計:
//!
//!   let reader = ktx2::Reader::new(bytes.as_slice()).unwrap();
//!   let header = reader.header();
//!   let mut image = Vec::new();
//!   for level in reader.levels() { image.extend_from_slice(level.data); }
//!   device.create_texture_with_data(queue, &TextureDescriptor {
//!       size: Extent3d { width, height, depth_or_array_layers: 6 },
//!       mip_level_count: header.level_count,
//!       dimension: TextureDimension::D2,
//!       format: wgpu::TextureFormat::Rgba16Float,
//!       ..
//!   }, TextureDataOrder::MipMajor, &image);
//!
//! faceCount=6, layerCount=0 (非配列), レベルはミップメジャー順
//! (各レベル = 6面ぶん連続、各面 = face_size*face_size*8byte(RGBA16F))。

use half::f16;
use std::io::Write;

const KTX2_MAGIC: [u8; 12] = [
    0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A,
];

/// Vulkan VK_FORMAT_R16G16B16A16_SFLOAT (wgpu::TextureFormat::Rgba16Float に対応)
const VK_FORMAT_R16G16B16A16_SFLOAT: u32 = 97;

// ---------------------------------------------------------------------
// DFD (Data Format Descriptor) 構築: 非圧縮 RGBA16F 用の Basic DFD
// ---------------------------------------------------------------------

/// R16G16B16A16_SFLOAT 用の Data Format Descriptor (KHR_df 準拠) を組み立てる。
/// dfdTotalSize(4byte) + descriptor block(88byte) = 92byte を返す。
fn build_dfd_rgba16f() -> Vec<u8> {
    let mut buf = Vec::with_capacity(92);
    let mut block = Vec::with_capacity(88);

    // Word0: vendorId(17bit)=0, descriptorType(15bit)=0 (Khronos basic format descriptor)
    block.extend_from_slice(&0u32.to_le_bytes());
    // Word1: versionNumber(16bit)=2, descriptorBlockSize(16bit)=88
    let word1: u32 = 2u32 | (88u32 << 16);
    block.extend_from_slice(&word1.to_le_bytes());
    // Word2: colorModel=1(RGBSDA), colorPrimaries=1(BT709), transferFunction=1(LINEAR), flags=0
    let word2: u32 = 1 | (1 << 8) | (1 << 16) | (0 << 24);
    block.extend_from_slice(&word2.to_le_bytes());
    // Word3: texelBlockDimension0..3 = 0 (1x1 ブロック = 非圧縮)
    block.extend_from_slice(&0u32.to_le_bytes());
    // Word4: bytesPlane0=8 (RGBA16F = 4ch * 2byte = 8byte/texel), plane1..3=0
    block.extend_from_slice(&8u32.to_le_bytes());
    // Word5: bytesPlane4..7 = 0
    block.extend_from_slice(&0u32.to_le_bytes());

    // サンプル情報 x4 (R, G, B, A)。各16byte。
    const CH_RED: u8 = 0;
    const CH_GREEN: u8 = 1;
    const CH_BLUE: u8 = 2;
    const CH_ALPHA: u8 = 15;
    const QUALIFIER_FLOAT_SIGNED: u8 = 0xC0; // FLOAT(0x80) | SIGNED(0x40)

    let samples: [(u16, u8); 4] = [
        (0, CH_RED),
        (16, CH_GREEN),
        (32, CH_BLUE),
        (48, CH_ALPHA),
    ];

    for (bit_offset, channel_id) in samples {
        let channel_type = channel_id | QUALIFIER_FLOAT_SIGNED;
        let w0: u32 = (bit_offset as u32) | (15u32 << 16) | ((channel_type as u32) << 24);
        block.extend_from_slice(&w0.to_le_bytes());
        block.extend_from_slice(&0u32.to_le_bytes()); // samplePosition0..3
        block.extend_from_slice(&(-1.0f32).to_bits().to_le_bytes()); // sampleLower
        block.extend_from_slice(&(1.0f32).to_bits().to_le_bytes()); // sampleUpper
    }

    debug_assert_eq!(block.len(), 88);

    let dfd_total_size = 4 + block.len() as u32;
    buf.extend_from_slice(&dfd_total_size.to_le_bytes());
    buf.extend_from_slice(&block);
    buf
}

// ---------------------------------------------------------------------
// KTX2ファイル書き出し
// ---------------------------------------------------------------------

/// mip0 (face_sizeの1辺) からmipN-1まで、各ミップにつき6面ぶんの生バイト列
/// (RGBA16F, face_size*face_size*8byte, 面順は +X,-X,+Y,-Y,+Z,-Z) を渡す。
pub fn write_ktx2_cubemap_rgba16f(
    path: &str,
    face_size: u32,
    mips: &[[Vec<u8>; 6]], // mips[level][face] = RGBA16F生バイト列
) -> std::io::Result<()> {
    let level_count = mips.len() as u32;
    assert!(level_count >= 1, "at least 1 mip level is required");

    let dfd = build_dfd_rgba16f();
    let dfd_len = dfd.len() as u32;

    let header_size = 80u32;
    let level_index_size = 24u32 * level_count;

    let dfd_offset = header_size + level_index_size;
    let kvd_offset = dfd_offset + dfd_len;
    let kvd_len = 0u32;

    let align = |v: u64| -> u64 { (v + 7) & !7 };
    let mut data_cursor = align((kvd_offset + kvd_len) as u64);

    let mut level_entries: Vec<(u64, u64)> = Vec::with_capacity(mips.len());
    for level_faces in mips {
        let level_len: u64 = level_faces.iter().map(|f| f.len() as u64).sum();
        level_entries.push((data_cursor, level_len));
        data_cursor = align(data_cursor + level_len);
    }

    let mut out = Vec::<u8>::new();

    // header (80 bytes)
    out.extend_from_slice(&KTX2_MAGIC);
    out.extend_from_slice(&VK_FORMAT_R16G16B16A16_SFLOAT.to_le_bytes());
    out.extend_from_slice(&2u32.to_le_bytes()); // typeSize (f16 = 2byte)
    out.extend_from_slice(&face_size.to_le_bytes()); // pixelWidth
    out.extend_from_slice(&face_size.to_le_bytes()); // pixelHeight
    out.extend_from_slice(&0u32.to_le_bytes()); // pixelDepth
    out.extend_from_slice(&0u32.to_le_bytes()); // layerCount
    out.extend_from_slice(&6u32.to_le_bytes()); // faceCount
    out.extend_from_slice(&level_count.to_le_bytes()); // levelCount
    out.extend_from_slice(&0u32.to_le_bytes()); // supercompressionScheme

    out.extend_from_slice(&dfd_offset.to_le_bytes());
    out.extend_from_slice(&dfd_len.to_le_bytes());
    out.extend_from_slice(&kvd_offset.to_le_bytes());
    out.extend_from_slice(&kvd_len.to_le_bytes());
    out.extend_from_slice(&0u64.to_le_bytes()); // sgdByteOffset
    out.extend_from_slice(&0u64.to_le_bytes()); // sgdByteLength

    debug_assert_eq!(out.len() as u32, header_size);

    for (offset, length) in &level_entries {
        out.extend_from_slice(&offset.to_le_bytes());
        out.extend_from_slice(&length.to_le_bytes());
        out.extend_from_slice(&length.to_le_bytes()); // uncompressedByteLength
    }

    debug_assert_eq!(out.len() as u32, dfd_offset);

    out.extend_from_slice(&dfd);
    debug_assert_eq!(out.len() as u32, kvd_offset);
    // KVD は空

    for (level_faces, (offset, _len)) in mips.iter().zip(level_entries.iter()) {
        while (out.len() as u64) < *offset {
            out.push(0);
        }
        for face_bytes in level_faces {
            out.extend_from_slice(face_bytes);
        }
    }

    let mut file = std::fs::File::create(path)?;
    file.write_all(&out)?;
    Ok(())
}

// ---------------------------------------------------------------------
// ミップチェーン生成 (CPU側、2x2ボックスフィルタ、面ごとに独立処理)
// ---------------------------------------------------------------------

fn downsample_face_f16(src: &[u16], size: u32) -> Vec<u16> {
    let half_size = (size / 2).max(1);
    let mut dst = vec![0u16; (half_size * half_size * 4) as usize];

    for y in 0..half_size {
        for x in 0..half_size {
            for c in 0..4 {
                let get = |sx: u32, sy: u32| -> f32 {
                    let idx = ((sy * size + sx) * 4 + c) as usize;
                    f16::from_bits(src[idx]).to_f32()
                };
                let sx = (x * 2).min(size - 1);
                let sy = (y * 2).min(size - 1);
                let sx1 = (sx + 1).min(size - 1);
                let sy1 = (sy + 1).min(size - 1);

                let avg = (get(sx, sy) + get(sx1, sy) + get(sx, sy1) + get(sx1, sy1)) * 0.25;

                let didx = ((y * half_size + x) * 4 + c) as usize;
                dst[didx] = f16::from_f32(avg).to_bits();
            }
        }
    }
    dst
}

/// mip0 (6面, u16ビットパターン) からミップチェーン全体 (1x1まで) を生成する。
pub fn generate_mip_chain_f16(base_faces: [Vec<u16>; 6], face_size: u32) -> Vec<[Vec<u16>; 6]> {
    let mut chain = Vec::new();
    chain.push(base_faces.clone());

    let mut cur = base_faces;
    let mut size = face_size;
    while size > 1 {
        let next: [Vec<u16>; 6] = std::array::from_fn(|i| downsample_face_f16(&cur[i], size));
        size = (size / 2).max(1);
        chain.push(next.clone());
        cur = next;
    }
    chain
}

// ---------------------------------------------------------------------
// GPU cubemapテクスチャの読み戻し (mip0のみ、6面ぶん)
// ---------------------------------------------------------------------

pub fn readback_cubemap_rgba16f(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    face_size: u32,
) -> [Vec<u16>; 6] {
    let bytes_per_texel = 8u32; // RGBA16F
    let unpadded_bytes_per_row = face_size * bytes_per_texel;
    let align_bytes = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bytes_per_row =
        ((unpadded_bytes_per_row + align_bytes - 1) / align_bytes) * align_bytes;

    let buffer_size = (padded_bytes_per_row as u64) * (face_size as u64) * 6;
    let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cubemap_readback_buffer"),
        size: buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("cubemap_readback_encoder"),
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(face_size),
            },
        },
        wgpu::Extent3d {
            width: face_size,
            height: face_size,
            depth_or_array_layers: 6,
        },
    );
    queue.submit(Some(encoder.finish()));

    let slice = readback_buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        tx.send(res).unwrap();
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device poll failed");
    rx.recv().unwrap().expect("failed to map readback buffer");

    let data = slice.get_mapped_range();

    let mut faces: [Vec<u16>; 6] = Default::default();
    for face in 0..6usize {
        let mut face_data = vec![0u16; (face_size * face_size * 4) as usize];
        let face_offset = (padded_bytes_per_row as u64) * (face_size as u64) * (face as u64);
        for row in 0..face_size {
            let row_start = (face_offset + (padded_bytes_per_row as u64) * (row as u64)) as usize;
            let row_bytes = &data[row_start..row_start + unpadded_bytes_per_row as usize];
            let row_u16: &[u16] = bytemuck::cast_slice(row_bytes);
            let dst_start = (row * face_size * 4) as usize;
            face_data[dst_start..dst_start + row_u16.len()].copy_from_slice(row_u16);
        }
        faces[face] = face_data;
    }

    drop(data);
    readback_buffer.unmap();

    faces
}

// ---------------------------------------------------------------------
// オーケストレーション: equirect HDRI -> KTX2ファイル一発生成
// ---------------------------------------------------------------------

pub fn bake_equirect_to_ktx2(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    converter: &crate::equirect_to_cubemap::EquirectToCubemap,
    equirect_view: &wgpu::TextureView,
    face_size: u32,
    output_path: &str,
) -> std::io::Result<()> {
    let cubemap_texture = converter.convert(device, queue, equirect_view, face_size);
    let base_faces = readback_cubemap_rgba16f(device, queue, &cubemap_texture, face_size);
    let mip_chain_u16 = generate_mip_chain_f16(base_faces, face_size);

    let mips_bytes: Vec<[Vec<u8>; 6]> = mip_chain_u16
        .iter()
        .map(|faces| std::array::from_fn(|i| bytemuck::cast_slice(&faces[i]).to_vec()))
        .collect();

    write_ktx2_cubemap_rgba16f(output_path, face_size, &mips_bytes)
}
