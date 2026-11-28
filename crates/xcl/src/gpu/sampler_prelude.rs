//! `XCL_SAMPLER_PRELUDE=1` (#505, experimental, off by default): every device XCL makes
//! (`agility::create_device`: the title's, the preparation workers', the tests') gets, before
//! anything else, one root signature per static sampler state of the fixed list `STATES`, in
//! that order.
//!
//! On NVIDIA a pipeline made from another process's driver blob reads its static samplers by
//! the numbers the driver gave their states in the process that made the blob: the driver
//! numbers states in the order a process first makes them (#505: upipibi's two-process case,
//! and `static_sampler_blob_tests`). With the prelude first in every process, the states in
//! the list have the same numbers everywhere, so such a pipeline may keep and take its blob
//! (`covers`). A pipeline with a state outside the list keeps the #505 rule (no disk blob on
//! NVIDIA), and so does every pipeline in a process where any static sampler came before the
//! prelude. The blobs of pipelines the prelude covers are kept apart (`dir`), under the list's
//! hash: none made without it, or with another list, is ever taken.

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::ID3D12Device;

use super::recomp::pso::StaticSampler;

/// The 18 distinct static sampler states (`Translation::static_samplers`: the console S#
/// words and whether it compares) of the retail title's 2,682 translated compute programs
/// (corpus-v13's `translated\v2`, read 2026-11-28), sorted by their words. The pixel shaders'
/// states were not in that cache: a state met outside the list is logged (`covers`).
pub const STATES: [StaticSampler; 18] = [
    ([0x0000_0000, 0x00c0_0000, 0x1400_0000, 0], false),
    ([0x0000_0000, 0x00c0_0000, 0x1450_0000, 0], false),
    ([0x0000_0000, 0x00c0_0000, 0x1850_0000, 0], false),
    ([0x0000_0000, 0x00c0_0000, 0x1850_3f80, 0], false),
    ([0x0000_0012, 0x00c0_0000, 0x1850_0000, 0], false),
    ([0x0000_0036, 0x00c0_0000, 0x1400_0000, 0], false),
    ([0x0000_0092, 0x00c0_0000, 0x1400_0000, 0], false),
    ([0x0000_0092, 0x00c0_0000, 0x1450_0000, 0], false),
    ([0x0000_0092, 0x00c0_0000, 0x1850_0000, 0], false),
    ([0x0000_00b6, 0x00c0_0000, 0x1850_0000, 0x8000_0000], false),
    ([0x0000_0180, 0x00c0_0000, 0x1450_0000, 0], false),
    ([0x0000_01b6, 0x00c0_0000, 0x1400_0000, 0], false),
    ([0x0000_01b6, 0x00c0_0000, 0x1850_0000, 0], false),
    ([0x0000_0400, 0x00c0_0000, 0x18f0_0000, 0], false),
    ([0x0000_0492, 0x00c0_0000, 0x18f0_0000, 0], false),
    ([0x0000_0692, 0x00c0_0000, 0x18f0_0000, 0], false),
    ([0x0000_3092, 0x00c0_0000, 0x1850_0000, 0], true),
    ([0x0000_30b6, 0x00c0_0000, 0x1850_0000, 0x8000_0000], true),
];

/// The switch.
pub fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_SAMPLER_PRELUDE").is_ok_and(|v| v.trim() == "1"))
}

/// The list's digest: blobs the prelude covers are kept under it (`dir`).
pub fn hash() -> u64 {
    hash_of(&STATES)
}

fn hash_of(states: &[StaticSampler]) -> u64 {
    let bytes: Vec<u8> = states.iter().flat_map(|(w, c)| w.iter().flat_map(|v| v.to_le_bytes()).chain([*c as u8])).collect();
    super::shader::fnv64(&bytes)
}

/// The folder (in the driver blob folder) of the blobs the prelude covers.
pub fn dir() -> String {
    format!("prelude-{:016x}", hash())
}

/// The prelude ran in this process before any other static sampler.
static FIRST_OK: AtomicBool = AtomicBool::new(false);
/// What made the process's first static sampler (the prelude, or what came before it).
static FIRST: Mutex<Option<&'static str>> = Mutex::new(None);

thread_local! {
    /// inside `run`: its own root signatures are not "another static sampler"
    static IN: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The prelude on the new device `dev` (`agility::create_device`). The root signatures stay
/// alive for the process (a state's number might go with its last root signature).
pub fn run(dev: &ID3D12Device) {
    if !on() {
        return;
    }
    let before = *FIRST.lock().unwrap();
    IN.with(|i| i.set(true));
    let made = STATES.iter().filter(|s| super::recomp::pso::abi_root_signature_with(dev, true, std::slice::from_ref(*s)).map(std::mem::forget).is_ok()).count();
    IN.with(|i| i.set(false));
    if made != STATES.len() {
        crate::xlog_warn!("D3D12.X: XCL_SAMPLER_PRELUDE: only {made} of {} prelude root signatures made: no blob of a pipeline with static samplers is kept or taken in this process (#505 rule)", STATES.len());
        return;
    }
    match before {
        None => {
            *FIRST.lock().unwrap() = Some("the prelude");
            FIRST_OK.store(true, Ordering::Relaxed);
            crate::xlog_info!("D3D12.X: XCL_SAMPLER_PRELUDE: the prelude ({} static sampler states, list {:016x}) ran before any other static sampler of this process: pipelines whose states are all in it keep their driver blobs (in driver\\{})", STATES.len(), hash(), dir());
        }
        // (another device of this process: the prelude was first already, or never can be)
        Some(_) if FIRST_OK.load(Ordering::Relaxed) => {}
        Some(who) => crate::xlog_warn!(
            "D3D12.X: XCL_SAMPLER_PRELUDE: !!! the first static sampler of this process came from {who} BEFORE the prelude: its numbers differ from other processes', so no blob of a pipeline with static samplers is kept or taken in this process (#505 rule) !!!"
        ),
    }
}

/// A static sampler is made now by `who` (XCL's ABI root signatures, the present blit, a
/// title's root signature): the process's first is logged with whether the prelude came first.
pub fn note(who: &'static str) {
    if !on() || IN.with(|i| i.get()) {
        return;
    }
    let mut first = FIRST.lock().unwrap();
    if first.is_none() {
        *first = Some(who);
        crate::xlog_warn!("D3D12.X: XCL_SAMPLER_PRELUDE: !!! the first static sampler of this process is made by {who} with no prelude before it (no device from agility::create_device yet): blobs of pipelines with static samplers are not kept or taken in this process !!!");
    }
}

/// Whether a pipeline with the static sampler states `states` may keep and take a driver blob
/// another process made: the prelude is on, it ran first in this process, and every state is
/// in the list. Each state met outside the list is logged once.
pub fn covers(states: &[StaticSampler]) -> bool {
    if !on() || !FIRST_OK.load(Ordering::Relaxed) {
        return false;
    }
    let outside: Vec<&StaticSampler> = states.iter().filter(|s| !STATES.contains(s)).collect();
    if !outside.is_empty() {
        static SEEN: Mutex<Option<HashSet<StaticSampler>>> = Mutex::new(None);
        let mut seen = SEEN.lock().unwrap();
        let seen = seen.get_or_insert_with(HashSet::new);
        for s in outside.iter().copied() {
            if seen.len() < 64 && seen.insert(*s) {
                crate::xlog_info!("D3D12.X: XCL_SAMPLER_PRELUDE: static sampler state outside the list: [{:#010x}, {:#010x}, {:#010x}, {:#010x}] compare={} (its pipelines keep the #505 rule)", s.0[0], s.0[1], s.0[2], s.0[3], s.1);
            }
        }
    }
    outside.is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The list is what the blob folder is named by: a change of any word or of its order
    /// names another folder; it has no state twice.
    #[test]
    fn the_list_names_its_blob_folder() {
        let mut other = STATES;
        other.swap(0, 1);
        assert_ne!(hash_of(&other), hash());
        other = STATES;
        other[17].1 = false;
        assert_ne!(hash_of(&other), hash());
        assert_eq!(dir(), format!("prelude-{:016x}", hash()));
        let distinct: HashSet<StaticSampler> = STATES.iter().copied().collect();
        assert_eq!(distinct.len(), STATES.len());
        // (#505's tone map LUT sampler: MIN_MAG_LINEAR_MIP_POINT, clamp)
        assert!(STATES.contains(&([0x92, 0x00c0_0000, 0x1450_0000, 0], false)));
    }
}
