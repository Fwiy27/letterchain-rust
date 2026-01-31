mod game;
mod solver;

fn main() -> std::io::Result<()> {
    let daily_letters = vec!['L', 'U', 'R', 'U', 'V', 'P', 'E', 'L', 'O', 'N', 'L', 'O'];

    let mut mult = [1i32; 25];
    mult[1] = 2;
    mult[2] = 2;
    mult[10] = 2;
    mult[17] = 2;
    mult[9] = 3;

    solver::solve(&daily_letters, &mult, "dictionary.txt")?;

    Ok(())
}