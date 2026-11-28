//! GPU, two processes over one private `XCL_DXIL_CACHE` (#505, `XCL_SAMPLER_PRELUDE`): process
//! "make" builds a compute pipeline with three static samplers through `compute_container` (which
//! keeps its driver blob on disk where `blob_folder` says); process "load" builds the same one,
//! from that blob when it finds one, and samples a 2x2 texture between texels with each.
//!
//! What the driver does (#505, measured by upipibi and in our own runs on an RTX 3070 and an
//! RTX 5090): it numbers static sampler states in the order a process first makes them, among
//! the ones alive then (a released number goes to the next new state), and the blob holds the
//! numbers. So the blob reads wrong in a process that made other states first, unless both made
//! the same states in the same order first: the prelude.

use super::*;
use crate::gpu::recomp::pso::StaticSampler;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R32G32B32A32_FLOAT, DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC};

const ROLE: &str = "XCL_TEST_PRELUDE_ROLE";
/// states outside the list the loading process makes first (kept alive)
const OTHERS: &str = "XCL_TEST_PRELUDE_OTHERS";
/// `1`: a static sampler on a device not made by `agility::create_device`, before anything
const RAW_FIRST: &str = "XCL_TEST_PRELUDE_RAW_FIRST";
/// switch off: `keep` / `release` the list's root signatures made by hand first
const BY_HAND: &str = "XCL_TEST_PRELUDE_BY_HAND";

/// The pipeline's samplers, all in the list: point, linear (MIN_MAG_LINEAR_MIP_POINT: the tone
/// map's) and trilinear, clamped.
const SAMPLERS: [StaticSampler; 3] = [
    ([0x92, 0x00c0_0000, 0x1400_0000, 0], false),
    ([0x92, 0x00c0_0000, 0x1450_0000, 0], false),
    ([0x92, 0x00c0_0000, 0x1850_0000, 0], false),
];

/// The `k`th state outside the list: point, clamped, with a LOD bias of its own.
fn outside(k: usize) -> StaticSampler {
    ([0x92, 0x00c0_0000, 0x1400_0000 | ((k as u32 + 1) << 4), 0], false)
}

const PROGRAM: &str = "
SamplerState p : register(s0, space1);
SamplerState l : register(s1, space1);
SamplerState t : register(s2, space1);
RWByteAddressBuffer o : register(u0);
[numthreads(1, 1, 1)] void main() {
    Texture2D<float4> tex = ResourceDescriptorHeap[0];
    float2 at = float2(0.375, 0.375);
    o.Store4(0, asuint(tex.SampleLevel(p, at, 0)));
    o.Store4(16, asuint(tex.SampleLevel(l, at, 0)));
    o.Store4(32, asuint(tex.SampleLevel(t, at, 0)));
}";

/// texel (x, y) = (x, y, x + 2y, 1): at (0.375, 0.375) a point sample is texel (0, 0), a
/// filtered one a quarter of the way to each neighbour
const AT_POINT: [f32; 4] = [0.0, 0.0, 0.0, 1.0];
const FILTERED: [f32; 4] = [0.25, 0.25, 0.75, 1.0];

fn right(s: &[[f32; 4]; 3]) -> bool {
    let near = |a: [f32; 4], b: [f32; 4]| a.iter().zip(b).all(|(x, y)| (x - y).abs() <= 1.0 / 256.0);
    near(s[0], AT_POINT) && near(s[1], FILTERED) && near(s[2], FILTERED)
}

fn buffer(dev: &ID3D12Device, heap: D3D12_HEAP_TYPE, bytes: u64, flags: D3D12_RESOURCE_FLAGS, state: D3D12_RESOURCE_STATES) -> ID3D12Resource {
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: bytes,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_UNKNOWN,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: flags,
        ..Default::default()
    };
    let mut r: Option<ID3D12Resource> = None;
    unsafe { dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: heap, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut r) }.expect("buffer");
    r.unwrap()
}

fn transition(list: &ID3D12GraphicsCommandList, r: &ID3D12Resource, from: D3D12_RESOURCE_STATES, to: D3D12_RESOURCE_STATES) {
    let b = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: unsafe { std::mem::transmute_copy(r) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: from,
                StateAfter: to,
            }),
        },
    };
    unsafe { list.ResourceBarrier(&[b]) };
}

/// Run `pso` (over the root signature `rs`) once on `dev` with the 2x2 texture: its three samples.
unsafe fn sample(dev: &ID3D12Device, rs: &ID3D12RootSignature, pso: &ID3D12PipelineState) -> [[f32; 4]; 3] {
    let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
    let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
    let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
    let tex_desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: 2,
        Height: 2,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        ..Default::default()
    };
    let mut tex: Option<ID3D12Resource> = None;
    dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &tex_desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut tex).unwrap();
    let tex = tex.unwrap();
    // two rows of 256 bytes (the copy's row pitch), 32 bytes of texels each
    let upload = buffer(dev, D3D12_HEAP_TYPE_UPLOAD, 512, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
    let mut at = std::ptr::null_mut();
    upload.Map(0, None, Some(&mut at)).unwrap();
    for y in 0..2usize {
        for x in 0..2usize {
            let v = [x as f32, y as f32, (x + 2 * y) as f32, 1.0];
            std::ptr::copy_nonoverlapping(v.as_ptr(), (at as *mut u8).add(y * 256 + x * 16) as *mut f32, 4);
        }
    }
    upload.Unmap(0, None);
    let out = buffer(dev, D3D12_HEAP_TYPE_DEFAULT, 256, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    let back = buffer(dev, D3D12_HEAP_TYPE_READBACK, 256, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
    let heap: ID3D12DescriptorHeap = dev
        .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, NumDescriptors: 1, Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, NodeMask: 0 })
        .unwrap();
    let srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Texture2D: D3D12_TEX2D_SRV { MipLevels: 1, ..Default::default() } },
    };
    dev.CreateShaderResourceView(&tex, Some(&srv), heap.GetCPUDescriptorHandleForHeapStart());
    let src = D3D12_TEXTURE_COPY_LOCATION {
        pResource: std::mem::transmute_copy(&upload),
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT { Offset: 0, Footprint: D3D12_SUBRESOURCE_FOOTPRINT { Format: DXGI_FORMAT_R32G32B32A32_FLOAT, Width: 2, Height: 2, Depth: 1, RowPitch: 256 } },
        },
    };
    let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&tex), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 } };
    list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
    transition(&list, &tex, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    list.SetDescriptorHeaps(&[Some(heap.clone())]);
    list.SetComputeRootSignature(rs);
    list.SetComputeRootUnorderedAccessView(0, out.GetGPUVirtualAddress());
    list.SetPipelineState(pso);
    list.Dispatch(1, 1, 1);
    transition(&list, &out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list.CopyBufferRegion(&back, 0, &out, 0, 48);
    list.Close().unwrap();
    queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
    let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
    queue.Signal(&fence, 1).unwrap();
    while fence.GetCompletedValue() < 1 {
        std::thread::yield_now();
    }
    let mut at = std::ptr::null_mut();
    back.Map(0, None, Some(&mut at)).unwrap();
    let v = std::slice::from_raw_parts(at as *const f32, 12);
    let s = [[v[0], v[1], v[2], v[3]], [v[4], v[5], v[6], v[7]], [v[8], v[9], v[10], v[11]]];
    back.Unmap(0, None);
    s
}

/// `1`: the graphics pipeline (`graphics_child`) instead of the compute one
const GRAPHICS: &str = "XCL_TEST_PRELUDE_GRAPHICS";
/// `1` (graphics): the blob in the usual folder whatever `blob_folder` says (the driver's own
/// behaviour, past the #505 rule)
const RAW_BLOB: &str = "XCL_TEST_PRELUDE_RAW_BLOB";

/// The tone map's shape: a full-screen triangle whose pixel samples a 2x2x2 LUT with each
/// sampler, one render target each.
const GRAPHICS_PROGRAM: &str = "
SamplerState p : register(s0, space1);
SamplerState l : register(s1, space1);
SamplerState t : register(s2, space1);
struct Out { float4 a : SV_Target0; float4 b : SV_Target1; float4 c : SV_Target2; };
float4 vs(uint id : SV_VertexID) : SV_Position {
    return float4(id == 1 ? 3.0 : -1.0, id == 2 ? -3.0 : 1.0, 0.0, 1.0);
}
Out ps() {
    Texture3D<float4> lut = ResourceDescriptorHeap[0];
    float3 at = float3(0.375, 0.375, 0.375);
    Out o;
    o.a = lut.SampleLevel(p, at, 0);
    o.b = lut.SampleLevel(l, at, 0);
    o.c = lut.SampleLevel(t, at, 0);
    return o;
}";

/// LUT texel (x, y, z) = (x, y, z, 1): filtered at (0.375, 0.375, 0.375) it is 0.25 in each
const LUT_FILTERED: [f32; 4] = [0.25, 0.25, 0.25, 1.0];

fn lut_right(s: &[[f32; 4]; 3]) -> bool {
    let near = |a: [f32; 4], b: [f32; 4]| a.iter().zip(b).all(|(x, y)| (x - y).abs() <= 1.0 / 256.0);
    near(s[0], AT_POINT) && near(s[1], LUT_FILTERED) && near(s[2], LUT_FILTERED)
}

/// Draw `pso` once into three 1x1 targets with the LUT: the three pixels.
unsafe fn draw(dev: &ID3D12Device, rs: &ID3D12RootSignature, pso: &ID3D12PipelineState) -> [[f32; 4]; 3] {
    let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
    let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
    let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
    let texture = |dim, w: u64, h: u32, d: u16, flags, state| {
        let desc = D3D12_RESOURCE_DESC { Dimension: dim, Width: w, Height: h, DepthOrArraySize: d, MipLevels: 1, Format: DXGI_FORMAT_R32G32B32A32_FLOAT, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, Flags: flags, ..Default::default() };
        let mut r: Option<ID3D12Resource> = None;
        dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut r).unwrap();
        r.unwrap()
    };
    let lut = texture(D3D12_RESOURCE_DIMENSION_TEXTURE3D, 2, 2, 2, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
    // rows of 256 bytes, slices of 512 (the copy's pitches)
    let upload = buffer(dev, D3D12_HEAP_TYPE_UPLOAD, 1024, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
    let mut at = std::ptr::null_mut();
    upload.Map(0, None, Some(&mut at)).unwrap();
    for z in 0..2usize {
        for y in 0..2usize {
            for x in 0..2usize {
                let v = [x as f32, y as f32, z as f32, 1.0];
                std::ptr::copy_nonoverlapping(v.as_ptr(), (at as *mut u8).add(z * 512 + y * 256 + x * 16) as *mut f32, 4);
            }
        }
    }
    upload.Unmap(0, None);
    let targets: Vec<ID3D12Resource> = (0..3).map(|_| texture(D3D12_RESOURCE_DIMENSION_TEXTURE2D, 1, 1, 1, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, D3D12_RESOURCE_STATE_RENDER_TARGET)).collect();
    let back = buffer(dev, D3D12_HEAP_TYPE_READBACK, 3 * 256, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
    let heap: ID3D12DescriptorHeap = dev
        .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, NumDescriptors: 1, Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, NodeMask: 0 })
        .unwrap();
    let srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE3D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Texture3D: D3D12_TEX3D_SRV { MipLevels: 1, ..Default::default() } },
    };
    dev.CreateShaderResourceView(&lut, Some(&srv), heap.GetCPUDescriptorHandleForHeapStart());
    let rtvs: ID3D12DescriptorHeap = dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV, NumDescriptors: 3, ..Default::default() }).unwrap();
    let step = dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV) as usize;
    let base = rtvs.GetCPUDescriptorHandleForHeapStart();
    let handles: Vec<D3D12_CPU_DESCRIPTOR_HANDLE> = (0..3).map(|k| D3D12_CPU_DESCRIPTOR_HANDLE { ptr: base.ptr + k * step }).collect();
    for (t, h) in targets.iter().zip(&handles) {
        dev.CreateRenderTargetView(t, None, *h);
    }
    let src = D3D12_TEXTURE_COPY_LOCATION {
        pResource: std::mem::transmute_copy(&upload),
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT { Offset: 0, Footprint: D3D12_SUBRESOURCE_FOOTPRINT { Format: DXGI_FORMAT_R32G32B32A32_FLOAT, Width: 2, Height: 2, Depth: 2, RowPitch: 256 } },
        },
    };
    let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&lut), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 } };
    list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
    transition(&list, &lut, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    list.SetDescriptorHeaps(&[Some(heap.clone())]);
    list.SetGraphicsRootSignature(rs);
    list.SetPipelineState(pso);
    list.OMSetRenderTargets(3, Some(handles.as_ptr()), false, None);
    list.RSSetViewports(&[D3D12_VIEWPORT { Width: 1.0, Height: 1.0, MaxDepth: 1.0, ..Default::default() }]);
    list.RSSetScissorRects(&[windows::Win32::Foundation::RECT { left: 0, top: 0, right: 1, bottom: 1 }]);
    list.IASetPrimitiveTopology(windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    list.DrawInstanced(3, 1, 0, 0);
    for (k, t) in targets.iter().enumerate() {
        transition(&list, t, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
        let to = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::transmute_copy(&back),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT { Offset: 256 * k as u64, Footprint: D3D12_SUBRESOURCE_FOOTPRINT { Format: DXGI_FORMAT_R32G32B32A32_FLOAT, Width: 1, Height: 1, Depth: 1, RowPitch: 256 } },
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(t), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 } };
        list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
    }
    list.Close().unwrap();
    queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
    let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
    queue.Signal(&fence, 1).unwrap();
    while fence.GetCompletedValue() < 1 {
        std::thread::yield_now();
    }
    let mut at = std::ptr::null_mut();
    back.Map(0, None, Some(&mut at)).unwrap();
    let px = |k: usize| -> [f32; 4] { std::array::from_fn(|c| *((at as *const u8).add(256 * k) as *const f32).add(c)) };
    let s = [px(0), px(1), px(2)];
    back.Unmap(0, None);
    s
}

/// The graphics child: the pipeline made from the blob in its folder when one is there (else
/// built, and its blob kept there), drawn once. One `PRELUDE-CHILD` line.
unsafe fn graphics_child(role: &str, dev: &ID3D12Device) {
    let raw = dev.as_raw();
    let rs_raw = abi_rs_for(raw, &SAMPLERS).expect("root signature with the samplers");
    let rs = ID3D12RootSignature::from_raw_borrowed(&rs_raw).unwrap();
    let vs = crate::gpu::recomp::dxc::compile(GRAPHICS_PROGRAM, "vs", "vs_6_6", &[]).expect("DXC vs");
    let ps = crate::gpu::recomp::dxc::compile(GRAPHICS_PROGRAM, "ps", "ps_6_6", &[]).expect("DXC ps");
    let folder = if std::env::var(RAW_BLOB).is_ok_and(|v| v == "1") { Some(String::new()) } else { blob_folder(raw, &SAMPLERS, false) };
    let file = folder.as_deref().map(|f| in_folder(crate::gpu::recomp::dxc::cache_dir().join("driver").join("gfx-prelude-test.pso"), f));
    let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
        pRootSignature: std::mem::transmute_copy(rs),
        VS: D3D12_SHADER_BYTECODE { pShaderBytecode: vs.as_ptr().cast(), BytecodeLength: vs.len() },
        PS: D3D12_SHADER_BYTECODE { pShaderBytecode: ps.as_ptr().cast(), BytecodeLength: ps.len() },
        SampleMask: u32::MAX,
        RasterizerState: D3D12_RASTERIZER_DESC { FillMode: D3D12_FILL_MODE_SOLID, CullMode: D3D12_CULL_MODE_NONE, DepthClipEnable: true.into(), ..Default::default() },
        PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        NumRenderTargets: 3,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        ..Default::default()
    };
    for k in 0..3 {
        desc.RTVFormats[k] = DXGI_FORMAT_R32G32B32A32_FLOAT;
        desc.BlendState.RenderTarget[k].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
    }
    let blob = file.as_deref().and_then(|f| std::fs::read(f).ok());
    let from_blob = blob.as_ref().and_then(|b| {
        desc.CachedPSO = D3D12_CACHED_PIPELINE_STATE { pCachedBlob: b.as_ptr().cast(), CachedBlobSizeInBytes: b.len() };
        dev.CreateGraphicsPipelineState::<ID3D12PipelineState>(&desc).ok()
    });
    let reused = from_blob.is_some();
    let pso = match from_blob {
        Some(p) => p,
        None => {
            desc.CachedPSO = D3D12_CACHED_PIPELINE_STATE::default();
            let p: ID3D12PipelineState = dev.CreateGraphicsPipelineState(&desc).expect("graphics pipeline");
            if let (Some(f), Ok(b)) = (file.as_deref(), p.GetCachedBlob()) {
                driver_cache::write(f, std::slice::from_raw_parts(b.GetBufferPointer() as *const u8, b.GetBufferSize()));
            }
            p
        }
    };
    let kept = file.as_deref().is_some_and(|f| f.exists());
    let s = draw(dev, rs, &pso);
    println!("PRELUDE-CHILD role={role} from_blob={reused} folder={folder:?} kept={kept} right={} samples={s:?}", lut_right(&s));
}

/// A child: the device (with the prelude when the switch is on), what the case makes first, the
/// pipeline through `compute_container` (or `graphics_child`), its samples. One `PRELUDE-CHILD` line.
fn child(role: &str) {
    let others: usize = std::env::var(OTHERS).ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let by_hand = std::env::var(BY_HAND).unwrap_or_default();
    unsafe {
        if std::env::var(RAW_FIRST).is_ok_and(|v| v == "1") {
            let mut raw: Option<ID3D12Device> = None;
            windows::Win32::Graphics::Direct3D12::D3D12CreateDevice(None, windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0, &mut raw).expect("raw device");
            std::mem::forget(crate::gpu::recomp::pso::abi_root_signature_with(&raw.unwrap(), true, &[outside(99)]).expect("a static sampler before the prelude"));
        }
        let dev = crate::gpu::agility::create_device(None, windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0).expect("device");
        let raw = dev.as_raw();
        let made: Vec<ID3D12RootSignature> = match by_hand.as_str() {
            "keep" | "release" => crate::gpu::sampler_prelude::STATES.iter().map(|s| crate::gpu::recomp::pso::abi_root_signature_with(&dev, true, std::slice::from_ref(s)).unwrap()).collect(),
            _ => Vec::new(),
        };
        if by_hand == "release" {
            drop(made);
        } else {
            std::mem::forget(made);
        }
        for k in 0..others {
            std::mem::forget(crate::gpu::recomp::pso::abi_root_signature_with(&dev, true, &[outside(k)]).unwrap());
        }
        if std::env::var(GRAPHICS).is_ok_and(|v| v == "1") {
            return graphics_child(role, &dev);
        }
        let rs = abi_rs_for(raw, &SAMPLERS).expect("root signature with the samplers");
        let dxil = crate::gpu::recomp::dxc::compile(PROGRAM, "main", "cs_6_6", &[]).expect("DXC");
        let create: PsoFn = std::mem::transmute(windows::core::Interface::vtable(&dev).CreateComputePipelineState);
        let iid = pc_iid(Kind::PipelineState);
        let mut out = std::ptr::null_mut();
        let (hr, _, from_blob) = compute_container(raw, create, &D3D12_COMPUTE_PIPELINE_STATE_DESC::default(), rs, &dxil, crate::gpu::shader::fnv64(&dxil), &iid, &mut out, ComputeBuild::Whole);
        assert!(hr >= 0 && !out.is_null(), "compute_container: {hr:#x}");
        let folder = blob_folder(raw, &SAMPLERS, true);
        let kept = folder.as_deref().and_then(|f| driver_cache::path_for(&dxil, 0, false).map(|p| in_folder(p, f))).is_some_and(|p| p.exists());
        let pso = ID3D12PipelineState::from_raw(out);
        let rs = ID3D12RootSignature::from_raw_borrowed(&rs).unwrap();
        let s = sample(&dev, rs, &pso);
        println!("PRELUDE-CHILD role={role} from_blob={from_blob} folder={folder:?} kept={kept} right={} samples={s:?}", right(&s));
    }
}

struct Seen {
    from_blob: bool,
    kept: bool,
    right: bool,
    line: String,
    stderr: String,
}

fn spawn(test: &str, role: &str, cache: &std::path::Path, env: &[(&str, &str)]) -> Seen {
    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", &format!("hle::d3d12x::real::sampler_prelude_tests::{test}"), "--nocapture", "--test-threads=1"])
        .env_remove("XCL_SAMPLER_PRELUDE")
        .envs(env.iter().copied())
        .env(ROLE, role)
        .env("XCL_DXIL_CACHE", cache)
        .env("XCL_PSO_DISK_CACHE", "1")
        // (this pipeline builds fast: keep its blob all the same)
        .env("XCL_PSO_DISK_CACHE_MIN_MS", "0")
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(out.status.success(), "{role} failed ({:?}):\n{stdout}\n{stderr}", out.status);
    let line = stdout.lines().find_map(|l| l.find("PRELUDE-CHILD ").map(|at| l[at..].to_string())).unwrap_or_else(|| panic!("no PRELUDE-CHILD line:\n{stdout}"));
    let flag = |name: &str| line.contains(&format!(" {name}=true"));
    Seen { from_blob: flag("from_blob"), kept: flag("kept"), right: flag("right"), line, stderr }
}

/// The parent of a case: skip without an NVIDIA GPU or DXC; `make` and `load` with their
/// environments over a fresh cache: what each saw.
fn case(test: &str, make: &[(&str, &str)], load: &[(&str, &str)]) -> Option<(Seen, Seen)> {
    if let Ok(role) = std::env::var(ROLE) {
        child(&role);
        return None;
    }
    let dev = match crate::gpu::agility::create_device(None, windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0) {
        Ok(d) => d,
        Err(_) => {
            crate::gpu::recomp::runner::skip_test("no GPU");
            return None;
        }
    };
    if adapter_vendor(dev.as_raw()) != Some(0x10DE) {
        crate::gpu::recomp::runner::skip_test("not NVIDIA (#505 and the prelude are NVIDIA's)");
        return None;
    }
    if crate::gpu::recomp::dxc::location().is_err() {
        crate::gpu::recomp::runner::skip_test("no DXC");
        return None;
    }
    let cache = std::env::temp_dir().join(format!("xcl-prelude-{}-{test}", std::process::id()));
    let _ = std::fs::remove_dir_all(&cache);
    std::fs::create_dir_all(&cache).unwrap();
    let made = spawn(test, "make", &cache, make);
    let loaded = spawn(test, "load", &cache, load);
    let _ = std::fs::remove_dir_all(&cache);
    eprintln!("{test}:\n  {}\n  {}", made.line, loaded.line);
    assert!(made.right, "the making process samples right: {}", made.line);
    Some((made, loaded))
}

const ON: (&str, &str) = ("XCL_SAMPLER_PRELUDE", "1");

/// With the prelude in both processes the blob is kept and taken, and samples right after the
/// loading process made 10 states outside the list.
#[test]
fn gpu_prelude_in_both_processes_keeps_the_blob_right() {
    let Some((made, loaded)) = case("gpu_prelude_in_both_processes_keeps_the_blob_right", &[ON], &[ON, (OTHERS, "10")]) else { return };
    assert!(made.kept && loaded.from_blob, "the blob is kept (in the prelude's folder) and taken: {}", loaded.line);
    assert!(loaded.right, "{}", loaded.line);
}

/// Without the prelude (compute keeps its blobs on this line) the same blob reads wrong: the
/// bug the prelude is for. (If a driver stops doing this, this test says so.)
#[test]
fn gpu_without_the_prelude_the_blob_reads_wrong_after_other_states() {
    let Some((_, loaded)) = case("gpu_without_the_prelude_the_blob_reads_wrong_after_other_states", &[], &[(OTHERS, "10")]) else { return };
    assert!(loaded.from_blob, "{}", loaded.line);
    assert!(!loaded.right, "the driver read the blob right after 10 other states: #505's numbering is gone on it ({})", loaded.line);
}

/// A blob made with the prelude is not taken by a process without it, nor one made without it by
/// a process with it (other folders): each builds its own and samples right.
#[test]
fn gpu_a_blob_is_never_taken_across_the_prelude() {
    let Some((_, loaded)) = case("gpu_a_blob_is_never_taken_across_the_prelude", &[ON], &[]) else { return };
    assert!(!loaded.from_blob && loaded.right, "made with the prelude, loaded without: {}", loaded.line);
    let Some((_, loaded)) = case("gpu_a_blob_is_never_taken_across_the_prelude", &[], &[ON]) else { return };
    assert!(!loaded.from_blob && loaded.right, "made without the prelude, loaded with it: {}", loaded.line);
}

/// A static sampler made before the prelude (a device not from `agility::create_device`) is
/// warned of loudly, and that process neither keeps nor takes such blobs: it samples right.
#[test]
fn gpu_a_static_sampler_before_the_prelude_turns_the_blobs_off() {
    let Some((_, loaded)) = case("gpu_a_static_sampler_before_the_prelude_turns_the_blobs_off", &[ON], &[ON, (RAW_FIRST, "1")]) else { return };
    assert!(!loaded.from_blob && loaded.right, "{}", loaded.line);
    assert!(loaded.stderr.contains("XCL_SAMPLER_PRELUDE: !!!"), "the loud warning:\n{}", loaded.stderr);
}

/// Why the prelude's root signatures stay alive: made and RELEASED before 10 other states, its
/// numbers go to those, and the blob of a process that kept them reads wrong. (Switch off; the
/// list made by hand.)
#[test]
fn gpu_a_released_prelude_gives_its_numbers_away() {
    let Some((_, loaded)) = case("gpu_a_released_prelude_gives_its_numbers_away", &[(BY_HAND, "keep")], &[(BY_HAND, "keep"), (OTHERS, "10")]) else { return };
    assert!(loaded.from_blob && loaded.right, "kept in both: {}", loaded.line);
    let Some((_, loaded)) = case("gpu_a_released_prelude_gives_its_numbers_away", &[(BY_HAND, "keep")], &[(BY_HAND, "release"), (OTHERS, "10")]) else { return };
    assert!(loaded.from_blob && !loaded.right, "released before the 10: {}", loaded.line);
}

const GFX: (&str, &str) = (GRAPHICS, "1");
const RAW: (&str, &str) = (RAW_BLOB, "1");

/// Graphics (the tone map's shape): past the #505 rule (`RAW_BLOB`), a pipeline made from
/// another process's blob reads its LUT samplers wrong after 10 other states, as compute does.
#[test]
fn gpu_graphics_blob_reads_wrong_after_other_states_without_the_prelude() {
    let Some((made, loaded)) = case("gpu_graphics_blob_reads_wrong_after_other_states_without_the_prelude", &[GFX, RAW], &[GFX, RAW, (OTHERS, "10")]) else { return };
    assert!(made.kept && loaded.from_blob, "{}", loaded.line);
    assert!(!loaded.right, "the driver read the graphics blob right after 10 other states ({})", loaded.line);
}

/// Graphics with the prelude in both processes: the blob is kept (in the prelude's folder),
/// taken, and samples right after 10 states outside the list.
#[test]
fn gpu_graphics_prelude_in_both_processes_keeps_the_blob_right() {
    let Some((made, loaded)) = case("gpu_graphics_prelude_in_both_processes_keeps_the_blob_right", &[GFX, ON], &[GFX, ON, (OTHERS, "10")]) else { return };
    assert!(made.kept && loaded.from_blob, "{}", loaded.line);
    assert!(loaded.right, "{}", loaded.line);
}

/// Graphics without the prelude: the #505 rule keeps no blob on NVIDIA, the loading process
/// builds its own and samples right.
#[test]
fn gpu_graphics_without_the_prelude_keeps_no_blob() {
    let Some((made, loaded)) = case("gpu_graphics_without_the_prelude_keeps_no_blob", &[GFX], &[GFX, (OTHERS, "10")]) else { return };
    assert!(!made.kept && !loaded.from_blob && loaded.right, "{}", loaded.line);
}
