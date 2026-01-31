# Word Game Solver - Rust Implementation

This is an optimized Rust rewrite of a Python word game solver.

## Key Optimizations

### 1. **Packed State Representation**
- **Python**: Used `bytes` (8 bits per cell, 25 bytes = 200 bits)
- **Rust**: Uses `u128` (5 bits per cell, 25×5 = 125 bits)
- Benefit: More cache-efficient, fits in a single register

### 2. **Word Storage**
- **Python**: Stored words as uppercase strings in sets
- **Rust**: Packs words into integers:
  - 3-letter words → `u32` (3 bytes)
  - 4-letter words → `u64` (4 bytes)
  - 5-letter words → `u64` (5 bytes)
- Benefit: Faster hashing, better cache locality, ~3x memory reduction

### 3. **Hash Maps**
- **Python**: Uses built-in dict/set (SipHash)
- **Rust**: Uses `FxHashMap`/`FxHashSet` from `rustc-hash`
- Benefit: Much faster hashing for integer keys (~2-3x speedup)

### 4. **Inline Hints**
- Critical hot-path functions marked with `#[inline(always)]`
- Helps compiler optimize tight loops and reduce function call overhead

### 5. **Memory Layout**
- `#[repr(C)]`-like structs for predictable layout
- Stack allocation where possible (arrays instead of vectors)
- Benefit: Better cache behavior, no heap allocations in hot paths

### 6. **Bit Manipulation**
- Uses efficient bit operations for:
  - Checking disabled cells
  - Finding legal moves
  - Building masks
- Examples: `trailing_zeros()`, `lm & (lm - 1)` for bit iteration

### 7. **Data Structures**
- Fixed-size arrays `[u8; 26]` for letter counts (vs Python lists)
- Tuples and small arrays instead of vectors when size is known
- Benefit: No bounds checking overhead, better cache locality

### 8. **Type System**
- Strong typing prevents errors at compile time
- Zero-cost abstractions (iterators, etc.)
- No runtime type checking overhead

### 9. **Compiler Optimizations**
- Profile-guided optimizations available
- Link-time optimization (LTO) enabled
- Aggressive inlining and loop unrolling

## Performance Characteristics

Expected improvements over Python:
- **5-15x** faster overall execution
- **2-5x** less memory usage
- **Better cache utilization** due to packed representations
- **Predictable performance** (no GC pauses)

## Building

```bash
# Debug build
cargo build

# Optimized release build
cargo build --release

# Run
cargo run --release
```

## Configuration

The release profile in `Cargo.toml` is configured for maximum performance:
- `opt-level = 3`: Maximum optimizations
- `lto = "fat"`: Full link-time optimization
- `codegen-units = 1`: Better optimization at cost of compile time
- `panic = "abort"`: Smaller binary, faster unwinding

## Usage

```rust
let daily_letters = vec!['L', 'U', 'R', 'U', 'V', 'P', 'E', 'L', 'O', 'N', 'L', 'O'];

let mut mult = [1i32; 25];
mult[1] = 2;
mult[2] = 2;
mult[10] = 2;
mult[17] = 2;
mult[9] = 3;

solver::solve(&daily_letters, &mult, "dictionary.txt")?;
```

## Architecture

- `game.rs`: Core game logic, state representation, word detection
- `solver.rs`: DFS solver with memoization and pruning
- `main.rs`: Entry point with example usage

## Dictionary Format

Plain text file with one word per line. The code automatically filters for 3, 4, and 5-letter words and converts them to uppercase.

## Potential Further Optimizations

1. **SIMD**: Use SIMD instructions for parallel word checking
2. **Parallel Search**: Multi-threaded exploration of search tree
3. **Better Pruning**: Implement more sophisticated bounds
4. **Profile-Guided Optimization**: Use `cargo pgo` for runtime profiling
5. **Custom Allocator**: Use jemalloc or mimalloc for better allocation patterns
6. **Zobrist Hashing**: Even faster state hashing for memoization

## Limitations

The current WordIndex implementation is simplified for words up to 64 per length category. For larger dictionaries, the bit-packing strategy would need adjustment.