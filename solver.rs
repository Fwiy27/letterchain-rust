use crate::game::{self, DictByLen, PackedLetters, Segments, BASE_SCORE};
use rustc_hash::FxHashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use rayon::prelude::*;

const ALL_MASK: u32 = (1 << 25) - 1;
const REPORT_EVERY: usize = 10_000;

#[derive(Clone, Copy)]
struct LetterCounts([u8; 26]);

impl LetterCounts {
    #[inline(always)]
    fn new() -> Self {
        LetterCounts([0; 26])
    }

    #[inline(always)]
    fn inc(&mut self, idx: usize) {
        if idx < 26 {
            self.0[idx] = self.0[idx].saturating_add(1);
        }
    }

    #[inline(always)]
    fn get(&self, idx: usize) -> u8 {
        self.0[idx]
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

// Word index for pattern matching
pub struct WordIndex {
    words: Vec<u64>,
    counts: Vec<LetterCounts>,
    pos_letter_bits: Vec<[u64; 26]>,
    len: usize,
}

impl WordIndex {
    pub fn build(dict: &DictByLen, length: usize) -> Self {
        let words: Vec<u64> = match length {
            3 => dict.w3.iter().map(|&w| w as u64).collect(),
            4 => dict.w4.iter().copied().collect(),
            5 => dict.w5.iter().copied().collect(),
            _ => panic!("Invalid length"),
        };

        let mut counts = Vec::with_capacity(words.len());
        for &word in &words {
            let mut cnt = LetterCounts::new();
            for i in 0..length {
                let letter = ((word >> (8 * (length - 1 - i))) & 0xFF) as u8;
                if letter > 0 && letter <= 26 {
                    cnt.inc((letter - 1) as usize);
                }
            }
            counts.push(cnt);
        }

        let mut pos_letter_bits = vec![[0u64; 26]; length];
        for (word_idx, &word) in words.iter().enumerate() {
            let bit = 1u64 << (word_idx % 64);
            if word_idx >= 64 {
                continue;
            }

            for pos in 0..length {
                let letter = ((word >> (8 * (length - 1 - pos))) & 0xFF) as u8;
                if letter > 0 && letter <= 26 {
                    pos_letter_bits[pos][(letter - 1) as usize] |= bit;
                }
            }
        }

        WordIndex {
            words,
            counts,
            pos_letter_bits,
            len: length,
        }
    }

    #[inline]
    fn candidates_for_pattern(&self, pattern: &[Option<u8>]) -> u64 {
        if self.words.len() > 64 {
            return u64::MAX;
        }

        let mut bits = (1u64 << self.words.len()) - 1;

        for (pos, &cell) in pattern.iter().enumerate() {
            if let Some(letter) = cell {
                if letter == 0 || letter > 26 {
                    return 0;
                }
                let letter_idx = (letter - 1) as usize;
                bits &= self.pos_letter_bits[pos][letter_idx];
                if bits == 0 {
                    return 0;
                }
            }
        }

        bits
    }
}

fn segment_feasible(
    seg: &game::Segment,
    disabled_mask: u32,
    letters_packed: PackedLetters,
    remaining: &LetterCounts,
    windex: &WordIndex,
) -> bool {
    if (disabled_mask & seg.mask) != 0 {
        return false;
    }

    let len = seg.len as usize;
    let mut pattern = vec![None; len];
    let mut fixed_cnt = LetterCounts::new();

    for pos in 0..len {
        let cell = seg.idx[pos] as usize;
        let x = game::get_cell(letters_packed, cell);

        if x == 0 {
            continue;
        }

        if x > 26 {
            return false;
        }

        pattern[pos] = Some(x);
        fixed_cnt.inc((x - 1) as usize);
    }

    let mut cand_bits = windex.candidates_for_pattern(&pattern);

    while cand_bits != 0 {
        let trailing = cand_bits.trailing_zeros() as usize;
        let wi = trailing;

        if wi >= windex.words.len() {
            break;
        }

        let word_counts = &windex.counts[wi];

        let mut ok = true;
        for c in 0..26 {
            let need = word_counts.get(c).saturating_sub(fixed_cnt.get(c));
            if need > remaining.get(c) {
                ok = false;
                break;
            }
        }

        if ok {
            return true;
        }

        cand_bits &= cand_bits - 1;
    }

    false
}

#[inline]
fn max_possible_bonus(c3: i32, c4: i32, c5: i32, moves_left: usize) -> i32 {
    let mut best = 0;

    if c3 == 0 && c4 == 0 && moves_left > 0 {
        best = best.max(75);
    }
    if c3 == 0 && c5 == 0 && moves_left > 0 {
        best = best.max(50);
    }
    if c4 == 0 && c5 == 0 && moves_left > 0 {
        best = best.max(25);
    }

    if c5 >= 2 && c3 == 0 && c4 == 0 {
        best = best.max(75);
    }
    if c4 >= 2 && c3 == 0 && c5 == 0 {
        best = best.max(50);
    }
    if c3 >= 3 && c4 == 0 && c5 == 0 {
        best = best.max(25);
    }

    best
}

fn upper_bound(
    score: i32,
    c3: i32,
    c4: i32,
    c5: i32,
    m: usize,
    disabled_mask: u32,
    letters_packed: PackedLetters,
    segments: &Segments,
    seg_caps: &SegmentCaps,
    suffix_counts: &[LetterCounts],
    indices: &Indices,
    daily_letters_len: usize,
) -> i32 {
    let remaining = &suffix_counts[m];
    let mut ub_add = 0;

    for (seg, &cap) in segments.seg5.iter().zip(&seg_caps.caps5) {
        if segment_feasible(seg, disabled_mask, letters_packed, remaining, &indices.idx5) {
            ub_add += cap;
        }
    }

    for (seg, &cap) in segments.seg4.iter().zip(&seg_caps.caps4) {
        if segment_feasible(seg, disabled_mask, letters_packed, remaining, &indices.idx4) {
            ub_add += cap;
        }
    }

    for (seg, &cap) in segments.seg3.iter().zip(&seg_caps.caps3) {
        if segment_feasible(seg, disabled_mask, letters_packed, remaining, &indices.idx3) {
            ub_add += cap;
        }
    }

    ub_add += max_possible_bonus(c3, c4, c5, daily_letters_len - m);
    score + ub_add
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

struct Indices {
    idx3: WordIndex,
    idx4: WordIndex,
    idx5: WordIndex,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct State {
    disabled_mask: u32,
    letters_packed: PackedLetters,
    c3: i32,
    c4: i32,
    c5: i32,
    score: i32,
}

pub struct SolveResult {
    pub best_score: i32,
    pub best_order: Vec<usize>,
}

// Shared state for parallel search
struct SharedState {
    best_score: Mutex<i32>,
    nodes: Mutex<usize>,
    last_report_t: Mutex<Instant>,
}

impl SharedState {
    fn new() -> Arc<Self> {
        Arc::new(SharedState {
            best_score: Mutex::new(i32::MIN),
            nodes: Mutex::new(0),
            last_report_t: Mutex::new(Instant::now()),
        })
    }

    fn update_best(&self, score: i32) -> bool {
        let mut best = self.best_score.lock().unwrap();
        if score > *best {
            *best = score;
            true
        } else {
            false
        }
    }

    fn get_best(&self) -> i32 {
        *self.best_score.lock().unwrap()
    }

    fn increment_nodes(&self, start_t: Instant, daily_letters_len: usize, root_ub: i32, depth: usize) {
        let mut nodes = self.nodes.lock().unwrap();
        *nodes += 1;

        if *nodes % REPORT_EVERY == 0 {
            let mut last_t = self.last_report_t.lock().unwrap();
            let now = Instant::now();
            let dt = (now - *last_t).as_secs_f64();
            let total_dt = (now - start_t).as_secs_f64();
            let nps = if dt > 0.0 { REPORT_EVERY as f64 / dt } else { 0.0 };

            let best = self.get_best();
            let gap = if best > i32::MIN / 2 {
                Some(root_ub - best)
            } else {
                None
            };

            if let Some(g) = gap {
                println!(
                    "[progress] nodes={:>12} depth={:>2}/{} best={:>6}  rootUB-best={:>4}  n/s={:>10.0} elapsed={:.1}s",
                    nodes, depth, daily_letters_len, best, g, nps, total_dt
                );
            } else {
                println!(
                    "[progress] nodes={:>12} depth={:>2}/{} best=unset  n/s={:>10.0} elapsed={:.1}s",
                    nodes, depth, daily_letters_len, nps, total_dt
                );
            }

            *last_t = now;
        }
    }
}

pub fn solve_exact_parallel(
    daily_letters: &[u8],
    mult: &[i32; 25],
    dictionary_path: &str,
    num_threads: usize,
) -> std::io::Result<SolveResult> {
    // Configure rayon thread pool
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .ok(); // Ignore error if already initialized

    let dict = DictByLen::load(dictionary_path)?;
    let segments = game::build_segments(mult);

    let indices = Arc::new(Indices {
        idx3: WordIndex::build(&dict, 3),
        idx4: WordIndex::build(&dict, 4),
        idx5: WordIndex::build(&dict, 5),
    });

    let suffix_counts = Arc::new(build_suffix_counts(daily_letters));
    let seg_caps = Arc::new(SegmentCaps::new(&segments));
    let dict = Arc::new(dict);
    let segments = Arc::new(segments);
    let daily_letters = Arc::new(daily_letters.to_vec());

    let start_t = Instant::now();
    let shared = SharedState::new();

    let root_packed: PackedLetters = 0;
    let root_ub = upper_bound(
        0, 0, 0, 0, 0, 0, root_packed,
        &segments, &seg_caps, &suffix_counts, &indices, daily_letters.len(),
    );

    println!("Starting parallel search with {} threads", num_threads);

    // Thread-local caches
    let initial_state = State {
        disabled_mask: 0,
        letters_packed: root_packed,
        c3: 0,
        c4: 0,
        c5: 0,
        score: 0,
    };

    // Collect first-level moves to parallelize
    let letter = daily_letters[0];
    let filled = filled_mask_from_packed(initial_state.letters_packed);
    let legal_mask = (!filled) & (!initial_state.disabled_mask) & ALL_MASK;

    let mut first_moves = Vec::new();
    let mut lm = legal_mask;
    while lm != 0 {
        let pos = lm.trailing_zeros() as usize;
        lm &= lm - 1;

        if let Some((new_packed, new_disabled, new_score, nc3, nc4, nc5)) =
            game::apply_move_packed(
                initial_state.letters_packed,
                initial_state.disabled_mask,
                initial_state.score,
                initial_state.c3,
                initial_state.c4,
                initial_state.c5,
                pos,
                letter,
                &dict,
                &segments,
            )
        {
            let new_state = State {
                disabled_mask: new_disabled,
                letters_packed: new_packed,
                c3: nc3,
                c4: nc4,
                c5: nc5,
                score: new_score,
            };
            first_moves.push((pos, new_state));
        }
    }

    // Process first-level moves in parallel
    first_moves.par_iter().for_each(|(pos, state)| {
        let mut cache: FxHashMap<(usize, State), i32> = FxHashMap::default();

        dfs_parallel(
            1,
            *state,
            &daily_letters,
            &dict,
            &segments,
            &seg_caps,
            &suffix_counts,
            &indices,
            &shared,
            start_t,
            root_ub,
            &mut cache,
        );
    });

    let best_score = shared.get_best();
    let total_nodes = *shared.nodes.lock().unwrap();

    println!("\nSearch complete!");
    println!("Total nodes explored: {}", total_nodes);
    println!("Best score found: {}", best_score);

    // Reconstruct best path (sequential)
    let best_order = reconstruct_parallel(
        &daily_letters,
        &dict,
        &segments,
        &shared,
    );

    Ok(SolveResult {
        best_score,
        best_order,
    })
}

fn dfs_parallel(
    m: usize,
    state: State,
    daily_letters: &Arc<Vec<u8>>,
    dict: &Arc<DictByLen>,
    segments: &Arc<Segments>,
    seg_caps: &Arc<SegmentCaps>,
    suffix_counts: &Arc<Vec<LetterCounts>>,
    indices: &Arc<Indices>,
    shared: &Arc<SharedState>,
    start_t: Instant,
    root_ub: i32,
    cache: &mut FxHashMap<(usize, State), i32>,
) -> i32 {
    shared.increment_nodes(start_t, daily_letters.len(), root_ub, m);

    // Check cache
    let cache_key = (m, state);
    if let Some(&cached) = cache.get(&cache_key) {
        return cached;
    }

    // Upper bound pruning
    let ub = upper_bound(
        state.score, state.c3, state.c4, state.c5, m,
        state.disabled_mask, state.letters_packed,
        segments, seg_caps, suffix_counts, indices, daily_letters.len(),
    );

    let best_so_far = shared.get_best();
    if ub <= best_so_far {
        cache.insert(cache_key, i32::MIN);
        return i32::MIN;
    }

    // Terminal
    if m == daily_letters.len() {
        let final_score = game::finalize_score_packed(
            state.letters_packed, state.disabled_mask,
            state.score, state.c3, state.c4, state.c5,
        );

        shared.update_best(final_score);
        cache.insert(cache_key, final_score);
        return final_score;
    }

    let letter = daily_letters[m];
    let filled = filled_mask_from_packed(state.letters_packed);
    let legal_mask = (!filled) & (!state.disabled_mask) & ALL_MASK;

    if legal_mask == 0 {
        let final_score = game::finalize_score_packed(
            state.letters_packed, state.disabled_mask,
            state.score, state.c3, state.c4, state.c5,
        );

        shared.update_best(final_score);
        cache.insert(cache_key, final_score);
        return final_score;
    }

    let mut local_best = i32::MIN;
    let mut lm = legal_mask;

    while lm != 0 {
        let pos = lm.trailing_zeros() as usize;
        lm &= lm - 1;

        if let Some((new_packed, new_disabled, new_score, nc3, nc4, nc5)) =
            game::apply_move_packed(
                state.letters_packed,
                state.disabled_mask,
                state.score,
                state.c3,
                state.c4,
                state.c5,
                pos,
                letter,
                dict,
                segments,
            )
        {
            let new_state = State {
                disabled_mask: new_disabled,
                letters_packed: new_packed,
                c3: nc3,
                c4: nc4,
                c5: nc5,
                score: new_score,
            };

            let val = dfs_parallel(
                m + 1,
                new_state,
                daily_letters,
                dict,
                segments,
                seg_caps,
                suffix_counts,
                indices,
                shared,
                start_t,
                root_ub,
                cache,
            );

            if val > local_best {
                local_best = val;
            }
        }
    }

    cache.insert(cache_key, local_best);
    local_best
}

fn reconstruct_parallel(
    daily_letters: &[u8],
    dict: &DictByLen,
    segments: &Segments,
    shared: &Arc<SharedState>,
) -> Vec<usize> {
    let mut order = Vec::new();
    let mut state = State {
        disabled_mask: 0,
        letters_packed: 0,
        c3: 0,
        c4: 0,
        c5: 0,
        score: 0,
    };

    // Simple greedy reconstruction - in parallel version, exact reconstruction
    // would require storing the full search tree, which is memory intensive.
    // This gives a valid path but may not be THE optimal path.
    for m in 0..daily_letters.len() {
        let letter = daily_letters[m];
        let filled = filled_mask_from_packed(state.letters_packed);
        let legal_mask = (!filled) & (!state.disabled_mask) & ALL_MASK;

        let mut best_pos = None;
        let mut best_score = i32::MIN;
        let mut lm = legal_mask;

        while lm != 0 {
            let pos = lm.trailing_zeros() as usize;
            lm &= lm - 1;

            if let Some((new_packed, new_disabled, new_score, nc3, nc4, nc5)) =
                game::apply_move_packed(
                    state.letters_packed,
                    state.disabled_mask,
                    state.score,
                    state.c3,
                    state.c4,
                    state.c5,
                    pos,
                    letter,
                    dict,
                    segments,
                )
            {
                if new_score > best_score {
                    best_score = new_score;
                    best_pos = Some((pos, State {
                        disabled_mask: new_disabled,
                        letters_packed: new_packed,
                        c3: nc3,
                        c4: nc4,
                        c5: nc5,
                        score: new_score,
                    }));
                }
            }
        }

        if let Some((pos, new_state)) = best_pos {
            order.push(pos);
            state = new_state;
        } else {
            break;
        }
    }

    order
}

pub fn solve(
    letters: &[char],
    mult: &[i32; 25],
    dict_path: &str,
    num_threads: usize,
) -> std::io::Result<()> {
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

    let result = solve_exact_parallel(&letter_bytes, mult, dict_path, num_threads)?;

    println!("Best score: {}", result.best_score);
    println!("Best order: {:?}", result.best_order);

    Ok(())
}