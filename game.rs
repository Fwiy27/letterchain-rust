use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

// Base scores for word lengths
pub const BASE_SCORE: [i32; 6] = [0, 0, 0, 4, 7, 13];

// Packed state representation: 5 bits per cell (0 = empty, 1-26 = A-Z)
// We use u128 to fit all 25 cells (25 * 5 = 125 bits)
pub type PackedLetters = u128;

const CELL_MASK: u128 = 0x1F; // 5 bits
const ALL_MASK: u32 = (1 << 25) - 1;

#[inline(always)]
pub fn get_cell(packed: PackedLetters, pos: usize) -> u8 {
    ((packed >> (pos * 5)) & CELL_MASK) as u8
}

#[inline(always)]
fn set_cell(packed: PackedLetters, pos: usize, val: u8) -> PackedLetters {
    let shift = pos * 5;
    (packed & !(CELL_MASK << shift)) | ((val as u128) << shift)
}

#[derive(Clone)]
pub struct DictByLen {
    pub w3: rustc_hash::FxHashSet<u32>, // Packed 3-letter words
    pub w4: rustc_hash::FxHashSet<u64>, // Packed 4-letter words
    pub w5: rustc_hash::FxHashSet<u64>, // Packed 5-letter words
}

#[inline(always)]
fn pack_word_3(chars: &[u8]) -> u32 {
    ((chars[0] as u32) << 16) | ((chars[1] as u32) << 8) | (chars[2] as u32)
}

#[inline(always)]
fn pack_word_4(chars: &[u8]) -> u64 {
    ((chars[0] as u64) << 24) | ((chars[1] as u64) << 16) |
    ((chars[2] as u64) << 8) | (chars[3] as u64)
}

#[inline(always)]
fn pack_word_5(chars: &[u8]) -> u64 {
    ((chars[0] as u64) << 32) | ((chars[1] as u64) << 24) |
    ((chars[2] as u64) << 16) | ((chars[3] as u64) << 8) | (chars[4] as u64)
}

impl DictByLen {
    pub fn load<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let mut w3 = rustc_hash::FxHashSet::default();
        let mut w4 = rustc_hash::FxHashSet::default();
        let mut w5 = rustc_hash::FxHashSet::default();

        for line in reader.lines() {
            let word = line?;
            let upper = word.trim().to_uppercase();
            let bytes: Vec<u8> = upper.bytes().map(|b| b - b'A' + 1).collect();

            match bytes.len() {
                3 => { w3.insert(pack_word_3(&bytes)); }
                4 => { w4.insert(pack_word_4(&bytes)); }
                5 => { w5.insert(pack_word_5(&bytes)); }
                _ => {}
            }
        }

        Ok(DictByLen { w3, w4, w5 })
    }
}

#[derive(Clone, Copy)]
pub struct Segment {
    pub idx: [u8; 5],  // Max length is 5, unused slots are 0xFF
    pub len: u8,
    pub mask: u32,
    pub mult_prod: i32,
}

impl Segment {
    #[inline(always)]
    fn new(indices: &[usize], mult: &[i32]) -> Self {
        let mut idx = [0xFF; 5];
        let mut mask = 0u32;
        let mut mult_prod = 1i32;

        for (i, &cell) in indices.iter().enumerate() {
            idx[i] = cell as u8;
            mask |= 1 << cell;
            mult_prod *= mult[cell];
        }

        Segment {
            idx,
            len: indices.len() as u8,
            mask,
            mult_prod,
        }
    }
}

pub struct Segments {
    pub seg3: Vec<Segment>,
    pub seg4: Vec<Segment>,
    pub seg5: Vec<Segment>,
}

pub fn build_segments(mult: &[i32; 25]) -> Segments {
    let seg5_idx: &[&[usize]] = &[
        // Rows
        &[0,1,2,3,4], &[5,6,7,8,9], &[10,11,12,13,14], &[15,16,17,18,19], &[20,21,22,23,24],
        // Columns
        &[0,5,10,15,20], &[1,6,11,16,21], &[2,7,12,17,22], &[3,8,13,18,23], &[4,9,14,19,24],
        // Diagonals
        &[0,6,12,18,24], &[4,8,12,16,20],
    ];

    let seg4_idx: &[&[usize]] = &[
        // Row windows
        &[0,1,2,3], &[1,2,3,4],
        &[5,6,7,8], &[6,7,8,9],
        &[10,11,12,13], &[11,12,13,14],
        &[15,16,17,18], &[16,17,18,19],
        &[20,21,22,23], &[21,22,23,24],
        // Column windows
        &[0,5,10,15], &[5,10,15,20],
        &[1,6,11,16], &[6,11,16,21],
        &[2,7,12,17], &[7,12,17,22],
        &[3,8,13,18], &[8,13,18,23],
        &[4,9,14,19], &[9,14,19,24],
        // Diagonal windows
        &[0,6,12,18], &[1,7,13,19], &[5,11,17,23], &[6,12,18,24],
        &[3,7,11,15], &[4,8,12,16], &[8,12,16,20], &[9,13,17,21],
    ];

    let seg3_idx: &[&[usize]] = &[
        // Row windows
        &[0,1,2], &[1,2,3], &[2,3,4],
        &[5,6,7], &[6,7,8], &[7,8,9],
        &[10,11,12], &[11,12,13], &[12,13,14],
        &[15,16,17], &[16,17,18], &[17,18,19],
        &[20,21,22], &[21,22,23], &[22,23,24],
        // Column windows
        &[0,5,10], &[5,10,15], &[10,15,20],
        &[1,6,11], &[6,11,16], &[11,16,21],
        &[2,7,12], &[7,12,17], &[12,17,22],
        &[3,8,13], &[8,13,18], &[13,18,23],
        &[4,9,14], &[9,14,19], &[14,19,24],
        // Diagonal windows
        &[0,6,12], &[1,7,13], &[2,8,14],
        &[5,11,17], &[6,12,18], &[7,13,19],
        &[10,16,22], &[11,17,23], &[12,18,24],
        &[2,6,10], &[3,7,11], &[4,8,12],
        &[7,11,15], &[8,12,16], &[9,13,17],
        &[12,16,20], &[13,17,21], &[14,18,22],
    ];

    Segments {
        seg5: seg5_idx.iter().map(|&idx| Segment::new(idx, mult)).collect(),
        seg4: seg4_idx.iter().map(|&idx| Segment::new(idx, mult)).collect(),
        seg3: seg3_idx.iter().map(|&idx| Segment::new(idx, mult)).collect(),
    }
}

#[inline(always)]
pub fn finalize_score_packed(
    letters_packed: PackedLetters,
    disabled_mask: u32,
    score: i32,
    count3: i32,
    count4: i32,
    count5: i32,
) -> i32 {
    let mut final_score = score;

    // Exclusive-length bonuses
    if count5 >= 2 && count3 == 0 && count4 == 0 {
        final_score += 75;
    } else if count4 >= 2 && count3 == 0 && count5 == 0 {
        final_score += 50;
    } else if count3 >= 3 && count4 == 0 && count5 == 0 {
        final_score += 25;
    }

    // Unused tile penalty
    let mut leftover = 0;
    for i in 0..25 {
        if get_cell(letters_packed, i) != 0 && (disabled_mask & (1 << i)) == 0 {
            leftover += 1;
        }
    }
    final_score - 4 * leftover
}

#[inline(always)]
pub fn apply_move_packed(
    letters_packed: PackedLetters,
    disabled_mask: u32,
    score: i32,
    count3: i32,
    count4: i32,
    count5: i32,
    pos: usize,
    letter_byte: u8, // 1-26 for A-Z
    dict: &DictByLen,
    segments: &Segments,
) -> Option<(PackedLetters, u32, i32, i32, i32, i32)> {
    // Check legality
    if pos >= 25 || (disabled_mask & (1 << pos)) != 0 || get_cell(letters_packed, pos) != 0 {
        return None;
    }

    let new_packed = set_cell(letters_packed, pos, letter_byte);
    let mut new_score = score;
    let mut new_count3 = count3;
    let mut new_count4 = count4;
    let mut new_count5 = count5;
    let mut to_disable = 0u32;

    // Check length-5 segments
    for seg in &segments.seg5 {
        if (disabled_mask & seg.mask) != 0 {
            continue;
        }

        let cells = [
            get_cell(new_packed, seg.idx[0] as usize),
            get_cell(new_packed, seg.idx[1] as usize),
            get_cell(new_packed, seg.idx[2] as usize),
            get_cell(new_packed, seg.idx[3] as usize),
            get_cell(new_packed, seg.idx[4] as usize),
        ];

        if cells.iter().any(|&c| c == 0) {
            continue;
        }

        let word = pack_word_5(&cells);
        let rev_word = pack_word_5(&[cells[4], cells[3], cells[2], cells[1], cells[0]]);

        let mut scored = false;
        if dict.w5.contains(&word) {
            new_score += BASE_SCORE[5] * seg.mult_prod;
            new_count5 += 1;
            scored = true;
        }
        if dict.w5.contains(&rev_word) {
            new_score += BASE_SCORE[5] * seg.mult_prod;
            new_count5 += 1;
            scored = true;
        }

        if scored {
            to_disable |= seg.mask;
        }
    }

    // Check length-4 segments
    for seg in &segments.seg4 {
        if (disabled_mask & seg.mask) != 0 {
            continue;
        }

        let cells = [
            get_cell(new_packed, seg.idx[0] as usize),
            get_cell(new_packed, seg.idx[1] as usize),
            get_cell(new_packed, seg.idx[2] as usize),
            get_cell(new_packed, seg.idx[3] as usize),
        ];

        if cells.iter().any(|&c| c == 0) {
            continue;
        }

        let word = pack_word_4(&cells);
        let rev_word = pack_word_4(&[cells[3], cells[2], cells[1], cells[0]]);

        let mut scored = false;
        if dict.w4.contains(&word) {
            new_score += BASE_SCORE[4] * seg.mult_prod;
            new_count4 += 1;
            scored = true;
        }
        if dict.w4.contains(&rev_word) {
            new_score += BASE_SCORE[4] * seg.mult_prod;
            new_count4 += 1;
            scored = true;
        }

        if scored {
            to_disable |= seg.mask;
        }
    }

    // Check length-3 segments
    for seg in &segments.seg3 {
        if (disabled_mask & seg.mask) != 0 {
            continue;
        }

        let cells = [
            get_cell(new_packed, seg.idx[0] as usize),
            get_cell(new_packed, seg.idx[1] as usize),
            get_cell(new_packed, seg.idx[2] as usize),
        ];

        if cells.iter().any(|&c| c == 0) {
            continue;
        }

        let word = pack_word_3(&cells);
        let rev_word = pack_word_3(&[cells[2], cells[1], cells[0]]);

        let mut scored = false;
        if dict.w3.contains(&word) {
            new_score += BASE_SCORE[3] * seg.mult_prod;
            new_count3 += 1;
            scored = true;
        }
        if dict.w3.contains(&rev_word) {
            new_score += BASE_SCORE[3] * seg.mult_prod;
            new_count3 += 1;
            scored = true;
        }

        if scored {
            to_disable |= seg.mask;
        }
    }

    Some((new_packed, disabled_mask | to_disable, new_score, new_count3, new_count4, new_count5))
}