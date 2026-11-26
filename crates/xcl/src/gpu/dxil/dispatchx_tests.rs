use super::kernel::{self, KernelOptions};
use crate::gpu::{rdna2::encode, recomp::{self, runner, CsInterface}};

// An unguarded store at the 3-D global thread index: only EXEC can suppress excess lanes.
fn stores(threads: [u32; 3], groups: [u32; 3]) -> Vec<u8> {
    let mut code = Vec::new();
    let mut emit = |name, enc, fields: &[(&str, u64)], literal| {
        code.extend(encode(name, enc, fields, literal).unwrap());
    };
    for k in 0..3u64 {
        emit("V_MUL_LO_U32", "ENC_VOP3", &[("VDST", 3 + k), ("SRC0", k), ("SRC1", 255)], Some(threads[k as usize]));
        emit("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 3 + k), ("SRC0", 259 + k), ("VSRC1", k)], None);
    }
    emit("V_MUL_LO_U32", "ENC_VOP3", &[("VDST", 5), ("SRC0", 261), ("SRC1", 255)], Some(threads[1] * groups[1]));
    emit("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 4), ("SRC0", 261), ("VSRC1", 4)], None);
    emit("V_MUL_LO_U32", "ENC_VOP3", &[("VDST", 4), ("SRC0", 260), ("SRC1", 255)], Some(threads[0] * groups[0]));
    emit("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 3), ("SRC0", 260), ("VSRC1", 3)], None);
    emit("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 6), ("SRC0", 130), ("VSRC1", 3)], None);
    emit("V_MOV_B32", "ENC_VOP1", &[("VDST", 7), ("SRC0", 128)], None);
    emit("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 8), ("SRC0", 129), ("VSRC1", 3)], None);
    emit("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 6), ("DATA", 8), ("SADDR", 0x7D)], None);
    emit("S_ENDPGM", "ENC_SOPP", &[], None);
    code
}

fn interface(threads: [u32; 3]) -> CsInterface {
    CsInterface { threads, user_sgprs: 0, tgid: [true; 3], tg_size: false, tid_components: 3 }
}

pub(crate) fn wave32_store_container(groups: [u32; 3]) -> Vec<u8> {
    use crate::gpu::xbox::tests::{fssm, sample};
    let mut header = fssm(&stores([32, 1, 1], groups), &[(6, 1)]);
    header[0xd4] = 1 << 5; // wave32
    for (word, value) in header[0x138..0x148].chunks_exact_mut(4).zip([0x1380u32, 32, 1, 1]) {
        word.copy_from_slice(&value.to_le_bytes());
    }
    sample(5, &[header])
}

fn masked_stores(wave64: bool, split: bool) {
    if recomp::dxc::location().is_err() { runner::skip_test("no DXC"); return; }
    let supported = if wave64 && !split { runner::wave64_supported() } else { runner::wave32_supported() };
    if supported != Ok(true) { runner::skip_test("required wave size unavailable"); return; }
    let x = if wave64 { 64 } else { 32 };
    let square = if wave64 { [8, 8, 1] } else { [8, 4, 1] };
    let cube = if wave64 { [4, 4, 4] } else { [4, 4, 2] };
    let options = KernelOptions { wave64, split, mem_windows: 1, sampler_base: 0 };
    for (threads, groups, limits) in [
        ([x, 1, 1], [128 / x, 1, 1], [100, 0, 0]),
        ([x, 1, 1], [128 / x, 1, 1], [10, 0, 0]),
        ([x, 1, 1], [128 / x, 1, 1], [0, 0, 0]),
        (square, [2, 2, 1], [13, square[1] + 1, 0]),
        (cube, [2, 2, 2], [7, 6, cube[2] * 2 - 1]),
    ] {
        let shape = std::array::from_fn::<_, 3, _>(|k| threads[k] * groups[k]);
        let size = shape.iter().product::<u32>() as usize;
        let code = stores(threads, groups);
        let cs = interface(threads);
        let mut io = vec![0; recomp::KERNEL_SGPRS];
        for k in 0..3 { io[recomp::IO_THREAD_LIMIT[k]] = limits[k]; }
        let untouched = 0xDEAD_BEEF;
        let input = vec![untouched; size];
        let base = kernel::translate_compute_on(&code, options, &cs, true).unwrap();
        let plain = runner::dispatch(&base.dxil, groups, &io, &input).unwrap();
        assert_eq!(plain, (1..=size as u32).collect::<Vec<_>>(), "base ignores thread limits");
        let masked = kernel::translate_compute_with_thread_limits(&code, options, &cs, true).unwrap();
        let got = runner::dispatch(&masked.dxil, groups, &io, &input).unwrap();
        let mut expected = input;
        for z in 0..shape[2] {
            for y in 0..shape[1] {
                for x in 0..shape[0] {
                    if [x, y, z].iter().zip(limits).all(|(&v, limit)| limit == 0 || v < limit) {
                        let i = ((z * shape[1] + y) * shape[0] + x) as usize;
                        expected[i] = i as u32 + 1;
                    }
                }
            }
        }
        assert_eq!(got, expected, "threads={threads:?}, groups={groups:?}, limits={limits:?}");
    }
}

#[test]
fn dispatchx_wave32_masks_excess_lanes() { masked_stores(false, false); }
#[test]
fn dispatchx_split_wave64_masks_excess_lanes() { masked_stores(true, true); }
#[test]
fn dispatchx_native_wave64_masks_excess_lanes() { masked_stores(true, false); }

#[test]
fn dispatchx_base_translation_stays_byte_identical() {
    if recomp::dxc::location().is_err() { runner::skip_test("no DXC"); return; }
    let cs = interface([64, 1, 1]);
    let code = stores(cs.threads, [2, 1, 1]);
    for (wave64, split) in [(false, false), (true, true), (true, false)] {
        let options = KernelOptions { wave64, split, mem_windows: 1, sampler_base: 0 };
        let base = kernel::translate_compute_on(&code, options, &cs, true).unwrap().dxil;
        // The window's red/green harness records original-base bytes before installing the
        // implementation, then compares them here (same toolchain, no brittle golden hash).
        if let Some(dir) = std::env::var_os("XCL_TEST_DISPATCHX_BASELINE") {
            let path = std::path::PathBuf::from(dir).join(format!("{wave64}-{split}.dxil"));
            if std::env::var_os("XCL_TEST_DISPATCHX_RECORD").is_some() {
                std::fs::write(&path, &base).unwrap();
            } else {
                assert_eq!(std::fs::read(path).unwrap(), base, "original base translation changed");
            }
        }
        let _masked = kernel::translate_compute_with_thread_limits(&code, options, &cs, true).unwrap();
        assert_eq!(kernel::translate_compute_on(&code, options, &cs, true).unwrap().dxil, base,
            "building a masked entry must not mutate subsequent base translations");
    }
}
