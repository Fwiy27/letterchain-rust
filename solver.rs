use crate::game::{self, DictByLen, PackedLetters, Segments, BASE_SCORE};
use rustc_hash::FxHashMap;
use std::time::Instant;

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
    words: Vec<u64>, // Packed words
    counts: Vec<LetterCounts>,
    // pos_letter_bits[position][letter] = bitmask of words
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
            let bit = 1u64 << (word_idx % 64); // Limited to 64 words per bit position
            if word_idx >= 64 {
                continue; // Simple limitation for this implementation
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
            // Simplified: if more than 64 words, just return all set
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

        cand_bits &= cand_bits - 1; // Clear lowest bit
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

    // Check 5-letter segments
    for (seg, &cap) in segments.seg5.iter().zip(&seg_caps.caps5) {
        if segment_feasible(seg, disabled_mask, letters_packed, remaining, &indices.idx5) {
            ub_add += cap;
        }
    }

    // Check 4-letter segments
    for (seg, &cap) in segments.seg4.iter().zip(&seg_caps.caps4) {
        if segment_feasible(seg, disabled_mask, letters_packed, remaining, &indices.idx4) {
            ub_add += cap;
        }
    }

    // Check 3-letter segments
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

pub fn solve_exact(
    daily_letters: &[u8], // 1-26 for A-Z
    mult: &[i32; 25],
    dictionary_path: &str,
) -> std::io::Result<SolveResult> {
    let dict = DictByLen::load(dictionary_path)?;
    let segments = game::build_segments(mult);

    let indices = Indices {
        idx3: WordIndex::build(&dict, 3),
        idx4: WordIndex::build(&dict, 4),
        idx5: WordIndex::build(&dict, 5),
    };

    let suffix_counts = build_suffix_counts(daily_letters);
    let seg_caps = SegmentCaps::new(&segments);

    let mut nodes = 0usize;
    let start_t = Instant::now();
    let mut last_report_t = start_t;
    let mut best_score = i32::MIN;

    let root_packed: PackedLetters = 0;
    let root_ub = upper_bound(
        0, 0, 0, 0, 0, 0, root_packed,
        &segments, &seg_caps, &suffix_counts, &indices, daily_letters.len(),
    );

    let mut cache: FxHashMap<(usize, State), i32> = FxHashMap::default();

    fn dfs(
        m: usize,
        state: State,
        daily_letters: &[u8],
        dict: &DictByLen,
        segments: &Segments,
        seg_caps: &SegmentCaps,
        suffix_counts: &[LetterCounts],
        indices: &Indices,
        nodes: &mut usize,
        best_score: &mut i32,
        last_report_t: &mut Instant,
        start_t: Instant,
        root_ub: i32,
        cache: &mut FxHashMap<(usize, State), i32>,
    ) -> i32 {
        *nodes += 1;

        if *nodes % REPORT_EVERY == 0 {
            let now = Instant::now();
            let dt = (now - *last_report_t).as_secs_f64();
            let total_dt = (now - start_t).as_secs_f64();
            let nps = if dt > 0.0 { REPORT_EVERY as f64 / dt } else { 0.0 };

            let gap = if *best_score > i32::MIN / 2 {
                Some(root_ub - *best_score)
            } else {
                None
            };

            if let Some(g) = gap {
                println!(
                    "[progress] nodes={:>12} depth={:>2}/{} best={:>6}  rootUB-best={:>4}  n/s={:>10.0} elapsed={:.1}s",
                    nodes, m, daily_letters.len(), best_score, g, nps, total_dt
                );
            } else {
                println!(
                    "[progress] nodes={:>12} depth={:>2}/{} best=unset  n/s={:>10.0} elapsed={:.1}s",
                    nodes, m, daily_letters.len(), nps, total_dt
                );
            }

            *last_report_t = now;
        }

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

        if ub <= *best_score {
            cache.insert(cache_key, i32::MIN);
            return i32::MIN;
        }

        // Terminal
        if m == daily_letters.len() {
            let final_score = game::finalize_score_packed(
                state.letters_packed, state.disabled_mask,
                state.score, state.c3, state.c4, state.c5,
            );

            if final_score > *best_score {
                *best_score = final_score;
            }

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

            if final_score > *best_score {
                *best_score = final_score;
            }

            cache.insert(cache_key, final_score);
            return final_score;
        }

        let mut local_best = i32::MIN;
        let mut lm = legal_mask;

        while lm != 0 {
            let pos = lm.trailing_zeros() as usize;
            lm &= lm - 1; // Clear lowest bit

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

                let val = dfs(
                    m + 1,
                    new_state,
                    daily_letters,
                    dict,
                    segments,
                    seg_caps,
                    suffix_counts,
                    indices,
                    nodes,
                    best_score,
                    last_report_t,
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

    let initial_state = State {
        disabled_mask: 0,
        letters_packed: root_packed,
        c3: 0,
        c4: 0,
        c5: 0,
        score: 0,
    };

    let best_final = dfs(
        0,
        initial_state,
        daily_letters,
        &dict,
        &segments,
        &seg_caps,
        &suffix_counts,
        &indices,
        &mut nodes,
        &mut best_score,
        &mut last_report_t,
        start_t,
        root_ub,
        &mut cache,
    );

    // Reconstruct best path
    fn reconstruct(
        daily_letters: &[u8],
        dict: &DictByLen,
        segments: &Segments,
        cache: &FxHashMap<(usize, State), i32>,
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

        for m in 0..daily_letters.len() {
            let letter = daily_letters[m];
            let filled = filled_mask_from_packed(state.letters_packed);
            let legal_mask = (!filled) & (!state.disabled_mask) & ALL_MASK;

            let mut best_pos = None;
            let mut best_val = i32::MIN;
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

                    let val = cache.get(&(m + 1, new_state)).copied().unwrap_or(i32::MIN);

                    if val > best_val {
                        best_val = val;
                        best_pos = Some((pos, new_state));
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

    let best_order = reconstruct(daily_letters, &dict, &segments, &cache);

    Ok(SolveResult {
        best_score: best_final,
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

    let result = solve_exact(&letter_bytes, mult, dict_path)?;

    println!("Best score: {}", result.best_score);
    println!("Best order: {:?}", result.best_order);

    Ok(())
}