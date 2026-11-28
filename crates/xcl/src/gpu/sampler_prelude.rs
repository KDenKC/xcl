//! `XCL_SAMPLER_PRELUDE=1` (#505, experimental, off by default): every device XCL makes
//! (`agility::create_device`: the title's, the preparation workers', the tests') gets, before
//! anything else, one root signature per static sampler state of this process's list (`list`):
//! the fixed `STATES`, then the states earlier processes added to the list's file, in order.
//!
//! On NVIDIA the driver gives each static sampler description a place in a per-process table,
//! in the order a process first makes them, among the ones alive then (a released place goes to
//! the next new description); the register and the shader visibility do not count, and samplers
//! in a descriptor heap take none. The places are compiled into the pipeline's code, so a blob
//! made in another process reads the wrong samplers unless that process numbered them alike
//! (#505: upipibi, CharlyRien, `real::sampler_prelude_tests`). With the prelude first in every
//! process, the list's states have the same places everywhere, so a pipeline whose states are
//! all in it may keep and take its blob (`covers`). One with a state outside it keeps the #505
//! rule (no disk blob on NVIDIA) in that process, and the state is appended to the list's file
//! for the next ones; every pipeline keeps the rule in a process where a static sampler came
//! before the prelude. A pipeline's blob is kept apart, under the hash of the list up to its last
//! state (`folder_for`): none made without the prelude, or with another list, is ever taken, and
//! a list that grows leaves the earlier folders valid.

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

/// The fixed list's digest: the folder of the blobs whose states are all among `STATES`.
pub fn hash() -> u64 {
    hash_of(&STATES)
}

fn hash_of(states: &[StaticSampler]) -> u64 {
    let bytes: Vec<u8> = states.iter().flat_map(|(w, c)| w.iter().flat_map(|v| v.to_le_bytes()).chain([*c as u8])).collect();
    super::shader::fnv64(&bytes)
}

/// The folder (in the driver blob folder) of the blobs whose states are all among `STATES`.
pub fn dir() -> String {
    format!("prelude-{:016x}", hash())
}

/// States past `STATES` the list takes at most (the file's further lines are not read).
const EXTRA_MOST: usize = 64;

/// The list grows: a state a process meets outside it is appended to this file (one state a line,
/// `extra_line`), and the next processes make it in their prelude after `STATES`, in the file's
/// order. Append only, so every list read from it is a prefix of every later one.
fn extra_file() -> std::path::PathBuf {
    super::recomp::dxc::cache_dir().join("driver").join("sampler-prelude-extra.txt")
}

fn extra_line((w, c): &StaticSampler) -> String {
    format!("{:08x} {:08x} {:08x} {:08x} {}\n", w[0], w[1], w[2], w[3], *c as u8)
}

/// The extra states of the file's `text`, in order: each once, none of `STATES`, malformed
/// lines skipped, `EXTRA_MOST` at most.
fn parse_extra(text: &str) -> Vec<StaticSampler> {
    let mut out: Vec<StaticSampler> = Vec::new();
    for line in text.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        let [a, b, c, d, cmp] = f[..] else { continue };
        let w: Option<Vec<u32>> = [a, b, c, d].iter().map(|x| u32::from_str_radix(x, 16).ok()).collect();
        let (Some(w), Ok(cmp)) = (w, cmp.parse::<u8>()) else { continue };
        let s: StaticSampler = ([w[0], w[1], w[2], w[3]], cmp != 0);
        if cmp <= 1 && !STATES.contains(&s) && !out.contains(&s) {
            out.push(s);
            if out.len() == EXTRA_MOST {
                break;
            }
        }
    }
    out
}

/// This process's list: `STATES`, then the extra states the file held when the prelude first
/// ran (read once: a state appended later is the next process's).
pub fn list() -> &'static [StaticSampler] {
    static LIST: OnceLock<Vec<StaticSampler>> = OnceLock::new();
    LIST.get_or_init(|| STATES.iter().copied().chain(parse_extra(&std::fs::read_to_string(extra_file()).unwrap_or_default())).collect())
}

/// The folder of a pipeline with the states `states`, all in `list`: the hash of the list up to
/// the last of them, never shorter than `STATES`. A list grown later names the same folder for
/// it, so blobs made with a shorter list stay valid. None: a state is not in `list`.
fn folder_in(list: &[StaticSampler], states: &[StaticSampler]) -> Option<String> {
    let last = states.iter().map(|s| list.iter().position(|l| l == s)).collect::<Option<Vec<usize>>>()?.into_iter().max().unwrap_or(0);
    let end = last.max(STATES.len() - 1);
    Some(format!("prelude-{:016x}", hash_of(&list[..=end])))
}

/// The folder of the blobs of a pipeline whose states `covers` this process's list.
pub fn folder_for(states: &[StaticSampler]) -> Option<String> {
    folder_in(list(), states)
}

/// `s` onto the file, unless it is there already (another process may have just added it).
fn append_extra(s: &StaticSampler) -> bool {
    use std::io::Write;
    let path = extra_file();
    let had = std::fs::read_to_string(&path).unwrap_or_default();
    let known = parse_extra(&had);
    if known.contains(s) || known.len() >= EXTRA_MOST {
        return false;
    }
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    // (one write of one whole line, in append mode)
    std::fs::OpenOptions::new().create(true).append(true).open(&path).and_then(|mut f| f.write_all(extra_line(s).as_bytes())).is_ok()
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
    let list = list();
    IN.with(|i| i.set(true));
    let made = list.iter().filter(|s| super::recomp::pso::abi_root_signature_with(dev, true, std::slice::from_ref(*s)).map(std::mem::forget).is_ok()).count();
    IN.with(|i| i.set(false));
    if made != list.len() {
        crate::xlog_warn!("D3D12.X: XCL_SAMPLER_PRELUDE: only {made} of {} prelude root signatures made: no blob of a pipeline with static samplers is kept or taken in this process (#505 rule)", list.len());
        return;
    }
    match before {
        None => {
            *FIRST.lock().unwrap() = Some("the prelude");
            FIRST_OK.store(true, Ordering::Relaxed);
            crate::xlog_info!("D3D12.X: XCL_SAMPLER_PRELUDE: the prelude ({} static sampler states: {} fixed, {} from {}) ran before any other static sampler of this process: pipelines whose states are all in it keep their driver blobs (in driver\\prelude-<hash of the list up to their last state>)", list.len(), STATES.len(), list.len() - STATES.len(), extra_file().display());
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
/// in this process's list. A state met outside it is appended to the list's file once (its
/// pipelines keep the #505 rule in this process; the next process makes it in its prelude).
pub fn covers(states: &[StaticSampler]) -> bool {
    if !on() || !FIRST_OK.load(Ordering::Relaxed) {
        return false;
    }
    let list = list();
    let outside: Vec<&StaticSampler> = states.iter().filter(|s| !list.contains(s)).collect();
    if !outside.is_empty() {
        static SEEN: Mutex<Option<HashSet<StaticSampler>>> = Mutex::new(None);
        let mut seen = SEEN.lock().unwrap();
        let seen = seen.get_or_insert_with(HashSet::new);
        for s in outside.iter().copied() {
            if seen.len() < EXTRA_MOST && seen.insert(*s) {
                let added = append_extra(s);
                crate::xlog_info!("D3D12.X: XCL_SAMPLER_PRELUDE: static sampler state outside the list: [{:#010x}, {:#010x}, {:#010x}, {:#010x}] compare={} (its pipelines keep the #505 rule in this process; {})", s.0[0], s.0[1], s.0[2], s.0[3], s.1, if added { "added to the list for the next processes" } else { "already in the list's file, or it is full" });
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

    /// The list's file: each state once in its first place, none of the fixed list, a bad line
    /// skipped, and a line written is read back.
    #[test]
    fn the_extra_states_are_read_in_order_once() {
        let a: StaticSampler = ([0x92, 0x00c0_0000, 0x1450_0070, 0], false);
        let b: StaticSampler = ([0x36, 0x00c0_0000, 0x1850_0000, 0x8000_0000], true);
        let text = format!("{}not a state\n{}{}{}1 2 3\n00000092 00c00000 14500000 00000000 7\n", extra_line(&a), extra_line(&STATES[3]), extra_line(&b), extra_line(&a));
        assert_eq!(parse_extra(&text), vec![a, b]);
        let many: String = (0..EXTRA_MOST as u32 + 5).map(|k| extra_line(&([0x92, 0x00c0_0000, 0x1400_0000 | (k + 1) << 4, 0], false))).collect();
        assert_eq!(parse_extra(&many).len(), EXTRA_MOST);
    }

    /// A pipeline's folder is the list's hash up to its last state, never shorter than the fixed
    /// list: the fixed states name `dir`, and a longer list names the same folder for a state it
    /// already had, another for a state only it has.
    #[test]
    fn a_longer_list_keeps_the_folders_of_the_shorter() {
        let a: StaticSampler = ([0x92, 0x00c0_0000, 0x1450_0070, 0], false);
        let b: StaticSampler = ([0x36, 0x00c0_0000, 0x1850_0000, 0], false);
        let short: Vec<StaticSampler> = STATES.iter().copied().chain([a]).collect();
        let long: Vec<StaticSampler> = STATES.iter().copied().chain([a, b]).collect();
        assert_eq!(folder_in(&short, &[STATES[0], STATES[7]]), Some(dir()));
        assert_eq!(folder_in(&long, &[STATES[7]]), Some(dir()), "the fixed states' folder stays");
        let with_a = folder_in(&short, &[STATES[0], a]);
        assert_eq!(folder_in(&long, &[a, STATES[2]]), with_a, "a's folder stays when b is added after it");
        assert_ne!(with_a, Some(dir()));
        assert_ne!(folder_in(&long, &[b]), with_a, "b is past a: its own folder");
        assert_eq!(folder_in(&short, &[b]), None, "b is not in the shorter list");
    }
}
