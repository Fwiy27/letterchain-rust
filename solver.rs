// Add to Cargo.toml:
// rayon = "1.8"

use crate::game::{self, DictByLen, PackedLetters, Segments, BASE_SCORE};
use rustc_hash::FxHashMap;
use rayon::prelude::*;
use std::sync::atomic::{AtomicI32, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

const ALL_MASK: u32 = (1 << 25) - 1;
const REPORT_EVERY: usize = 250_000;
const MAX_NODES: usize = 100_000_000;

// Same supporting code as before (LetterCounts, State, etc.)
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct State {
    disabled_mask: u32,
    letters_packed: PackedLetters,
    c3: i16,
    c4: i16,
    c5: i16,
    score: i32,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct LetterCounts([u8; 26]);

impl LetterCounts {
    #[inline(always)]
    const fn new() -> Self {
        LetterCounts([0; 26])
    }

    #[inline(always)]
    fn inc(&mut self, idx: usize) {
        unsafe {
            *self.0.get_unchecked_mut(idx) = self.0.get_unchecked(idx).saturating_add(1);
        }
    }

    #[inline(always)]
    fn get(&self, idx: usize) -> u8 {
        unsafe { *self.0.get_unchecked(idx) }
    }
}

struct SegmentCaps {
    caps3: Vec<i32>,
    caps4: Vec<i32>,
    caps5: Vec<i32>,
}

impl SegmentCaps {
    fn new(segments: &Segments) -> Self {
        SegmentCaps {
            caps5: segments.seg5.iter().map(|s| 2 * BASE_SCORE[5] * s.mult_prod).collect(),
            caps4: segments.seg4.iter().map(|s| 2 * BASE_SCORE[4] * s.mult_prod).collect(),
            caps3: segments.seg3.iter().map(|s| 2 * BASE_SCORE[3] * s.mult_prod).collect(),
        }
    }
}

fn build_suffix_counts(daily_letters: &[u8]) -> Vec<LetterCounts> {
    let n = daily_letters.len();
    let mut result = Vec::with_capacity(n + 1);
    let mut counts = LetterCounts::new();

    result.push(counts);

    for i in (0..n).rev() {
        let letter_idx = (daily_letters[i] - 1) as usize;
        if letter_idx < 26 {
            counts.inc(letter_idx);
        }
        result.push(counts);
    }

    result.reverse();
    result
}

#[inline(always)]
fn filled_mask_from_packed(packed: PackedLetters) -> u32 {
    let mut mask = 0u32;
    for i in 0..25 {
        if game::get_cell(packed, i) != 0 {
            mask |= 1 << i;
        }
    }
    mask
}

#[inline]
fn max_possible_bonus(c3: i16, c4: i16, c5: i16, moves_left: usize) -> i32 {
    if moves_left == 0 {
        if c5 >= 2 && c3 == 0 && c4 == 0 { return 75; }
        if c4 >= 2 && c3 == 0 && c5 == 0 { return 50; }
        if c3 >= 3 && c4 == 0 && c5 == 0 { return 25; }
        return 0;
    }

    let mut best = 0;
    if c3 == 0 && c4 == 0 { best = best.max(75); }
    if c3 == 0 && c5 == 0 { best = best.max(50); }
    if c4 == 0 && c5 == 0 { best = best.max(25); }
    if c5 >= 2 && c3 == 0 && c4 == 0 { best = best.max(75); }
    if c4 >= 2 && c3 == 0 && c5 == 0 { best = best.max(50); }
    if c3 >= 3 && c4 == 0 && c5 == 0 { best = best.max(25); }
    best
}

#[inline]
fn quick_upper_bound(
    score: i32,
    c3: i16,
    c4: i16,
    c5: i16,
    m: usize,
    disabled_mask: u32,
    seg_caps: &SegmentCaps,
    daily_letters_len: usize,
) -> i32 {
    let disabled_count = disabled_mask.count_ones() as i32;
    let active_cells = 25 - disabled_count;
    let max_per_cell = seg_caps.caps5.iter().max().copied().unwrap_or(0);
    let optimistic_add = active_cells * max_per_cell / 5;
    let bonus = max_possible_bonus(c3, c4, c5, daily_letters_len - m);
    score + optimistic_add + bonus
}

const MOVE_ORDER: [usize; 25] = [
    12, 6, 7, 8, 11, 13, 16, 17, 18,
    1, 2, 3, 5, 9, 10, 14, 15, 19, 21, 22, 23,
    0, 4, 20, 24,
];

#[inline]
fn ordered_legal_positions(legal_mask: u32) -> impl Iterator<Item = usize> {
    MOVE_ORDER.iter().copied().filter(move |&pos| (legal_mask & (1 << pos)) != 0)
}

// PARALLEL VERSION: Each thread has its own cache
fn dfs_parallel(
    m: usize,
    state: State,
    daily_letters: &[u8],
    dict: &DictByLen,
    segments: &Segments,
    seg_caps: &SegmentCaps,
    local_nodes: &mut usize,
    local_cache: &mut FxHashMap<(usize, State), i32>,
    global_best: &AtomicI32,
    global_nodes: &AtomicUsize,
) -> i32 {
    *local_nodes += 1;

    // Periodically sync with global counters
    if *local_nodes % 10_000 == 0 {
        global_nodes.fetch_add(10_000, Ordering::Relaxed);
        *local_nodes = 0;
    }

    // Check cache
    let cache_key = (m, state);
    if let Some(&cached) = local_cache.get(&cache_key) {
        return cached;
    }

    // Load current global best for pruning
    let current_best = global_best.load(Ordering::Relaxed);

    // Quick pruning
    let quick_ub = quick_upper_bound(
        state.score, state.c3, state.c4, state.c5, m,
        state.disabled_mask, seg_caps, daily_letters.len(),
    );

    if quick_ub <= current_best {
        local_cache.insert(cache_key, i32::MIN);
        return i32::MIN;
    }

    // Terminal
    if m == daily_letters.len() {
        let final_score = game::finalize_score_packed(
            state.letters_packed, state.disabled_mask,
            state.score, state.c3 as i32, state.c4 as i32, state.c5 as i32,
        );

        // Update global best atomically
        global_best.fetch_max(final_score, Ordering::Relaxed);

        local_cache.insert(cache_key, final_score);
        return final_score;
    }

    let letter = daily_letters[m];
    let filled = filled_mask_from_packed(state.letters_packed);
    let legal_mask = (!filled) & (!state.disabled_mask) & ALL_MASK;

    if legal_mask == 0 {
        let final_score = game::finalize_score_packed(
            state.letters_packed, state.disabled_mask,
            state.score, state.c3 as i32, state.c4 as i32, state.c5 as i32,
        );
        global_best.fetch_max(final_score, Ordering::Relaxed);
        local_cache.insert(cache_key, final_score);
        return final_score;
    }

    let mut local_best = i32::MIN;

    for pos in ordered_legal_positions(legal_mask) {
        if let Some((new_packed, new_disabled, new_score, nc3, nc4, nc5)) =
            game::apply_move_packed(
                state.letters_packed,
                state.disabled_mask,
                state.score,
                state.c3 as i32,
                state.c4 as i32,
                state.c5 as i32,
                pos,
                letter,
                dict,
                segments,
            )
        {
            let new_state = State {
                disabled_mask: new_disabled,
                letters_packed: new_packed,
                c3: nc3 as i16,
                c4: nc4 as i16,
                c5: nc5 as i16,
                score: new_score,
            };

            let val = dfs_parallel(
                m + 1,
                new_state,
                daily_letters,
                dict,
                segments,
                seg_caps,
                local_nodes,
                local_cache,
                global_best,
                global_nodes,
            );

            if val > local_best {
                local_best = val;
                global_best.fetch_max(val, Ordering::Relaxed);
            }
        }
    }

    local_cache.insert(cache_key, local_best);
    local_best
}

pub struct SolveResult {
    pub best_score: i32,
    pub best_order: Vec<usize>,
}

pub fn solve_exact_parallel(
    daily_letters: &[u8],
    mult: &[i32; 25],
    dictionary_path: &str,
) -> std::io::Result<SolveResult> {
    let dict = Arc::new(DictByLen::load(dictionary_path)?);
    let segments = Arc::new(game::build_segments(mult));
    let seg_caps = Arc::new(SegmentCaps::new(&segments));

    let global_best = Arc::new(AtomicI32::new(i32::MIN));
    let global_nodes = Arc::new(AtomicUsize::new(0));

    let start_t = Instant::now();

    // Progress reporter thread
    let global_nodes_clone = Arc::clone(&global_nodes);
    let global_best_clone = Arc::clone(&global_best);
    let reporter = std::thread::spawn(move || {
        loop {
            std::thread::sleep(std::time::Duration::from_secs(5));
            let nodes = global_nodes_clone.load(Ordering::Relaxed);
            let best = global_best_clone.load(Ordering::Relaxed);
            let elapsed = start_t.elapsed().as_secs_f64();
            let nps = nodes as f64 / elapsed;

            if best > i32::MIN / 2 {
                println!(
                    "[progress] nodes={:>12} best={:>6} n/s={:>10.0} elapsed={:.1}s",
                    nodes, best, nps, elapsed
                );
            } else {
                println!(
                    "[progress] nodes={:>12} best=unset n/s={:>10.0} elapsed={:.1}s",
                    nodes, nps, elapsed
                );
            }

            if nodes > MAX_NODES {
                break;
            }
        }
    });

    // Generate all first moves
    let initial_state = State {
        disabled_mask: 0,
        letters_packed: 0,
        c3: 0,
        c4: 0,
        c5: 0,
        score: 0,
    };

    let first_letter = daily_letters[0];
    let mut first_moves = Vec::new();

    for pos in ordered_legal_positions(ALL_MASK) {
        if let Some((new_packed, new_disabled, new_score, nc3, nc4, nc5)) =
            game::apply_move_packed(
                initial_state.letters_packed,
                initial_state.disabled_mask,
                initial_state.score,
                0, 0, 0,
                pos,
                first_letter,
                &dict,
                &segments,
            )
        {
            let new_state = State {
                disabled_mask: new_disabled,
                letters_packed: new_packed,
                c3: nc3 as i16,
                c4: nc4 as i16,
                c5: nc5 as i16,
                score: new_score,
            };
            first_moves.push((pos, new_state));
        }
    }

    println!("Parallel search with {} first moves across {} threads",
             first_moves.len(), rayon::current_num_threads());

    // PARALLEL: Each thread explores a different first move
    let results: Vec<_> = first_moves
        .par_iter()
        .map(|(first_pos, first_state)| {
            let mut local_nodes = 0;
            let mut local_cache = FxHashMap::default();

            let score = dfs_parallel(
                1,
                *first_state,
                daily_letters,
                &dict,
                &segments,
                &seg_caps,
                &mut local_nodes,
                &mut local_cache,
                &global_best,
                &global_nodes,
            );

            // Sync final local nodes
            global_nodes.fetch_add(local_nodes, Ordering::Relaxed);

            (*first_pos, score, local_cache)
        })
        .collect();

    // Find best result
    let (best_first_pos, best_score, _) = results
        .iter()
        .max_by_key(|(_, score, _)| score)
        .unwrap();

    let total_nodes = global_nodes.load(Ordering::Relaxed);
    let elapsed = start_t.elapsed().as_secs_f64();

    println!("\n=== Search Complete ===");
    println!("Total nodes: {}", total_nodes);
    println!("Best score: {}", best_score);
    println!("Time: {:.1}s ({:.0} nodes/sec)", elapsed, total_nodes as f64 / elapsed);

    // Reconstruct path (simplified - just return first move for now)
    let best_order = vec![*best_first_pos];

    Ok(SolveResult {
        best_score: *best_score,
        best_order,
    })
}

pub fn solve(letters: &[char], mult: &[i32; 25], dict_path: &str) -> std::io::Result<()> {
    let letter_bytes: Vec<u8> = letters
        .iter()
        .map(|&c| {
            let upper = c.to_ascii_uppercase();
            if upper.is_ascii_alphabetic() {
                (upper as u8) - b'A' + 1
            } else {
                0
            }
        })
        .collect();

    let result = solve_exact_parallel(&letter_bytes, mult, dict_path)?;

    println!("Best score: {}", result.best_score);
    println!("Best order: {:?}", result.best_order);

    Ok(())
}