use std::{collections::VecDeque, thread, time::Instant};

use cli_table::Cell;
use colored::Colorize;
use log::{debug, error};

use crate::{
    cli::{parse_cli_tiles, Cli, Tiling},
    cli_tables::{print_args_table, print_title},
    matrix_multiplication::{algorithms::Algorithm, matrix_multiplication},
    random_filled_square_matrix_of_size,
};

/// Benchmarks the execution time of a given matrix multiplication algorithm.
/// Returns the execution time in milliseconds, or `None` if an error occurred.
/// If an error occurs, the error is logged and printed to the console.
pub fn time_algorithm(algorithm: Algorithm, a: &Vec<Vec<i32>>, b: &Vec<Vec<i32>>) -> Option<u128> {
    let start = Instant::now();
    let res = matrix_multiplication(&a, &b, algorithm);
    let end = Instant::now();
    match res {
        Ok(_) => Some(end.duration_since(start).as_millis()),
        Err(err) => {
            println!("{} {}", "Error:".bright_red(), err);
            error!("Error: {}", err);
            None
        }
    }
}

pub fn matrix_multiplication_benchmark(cli: &Cli) {
    let n = cli.size;
    let iterations = cli.iterations;
    let available_threads = thread::available_parallelism().unwrap().get();
    let mut threads = cli.threads.unwrap_or(available_threads);
    let tile_size = cli.tile_size;

    let parallel_only: bool = cli.parallel_only;
    let skip_ijk = cli.skip_ijk;

    // collect execution times for different matrix multiplication methods

    let mut sequential_ijk_times = Vec::new();
    let mut sequential_ikj_times = Vec::new();
    let mut parallel_i_loop_times = Vec::new();
    let mut parallel_tiling_times = Vec::new();

    print_title("Welcome to Matrix Multiplication Benchmark!");

    if threads > available_threads {
        println!(
            "{} {} threads are available, but {} threads were requested. Using {} threads instead.",
            "Warning:".bright_yellow(),
            available_threads,
            threads,
            available_threads
        );
        threads = available_threads;
    }

    let table = vec![
        vec!["Matrix size".cell(), n.to_string().cell()],
        vec!["Number of threads".cell(), threads.to_string().cell()],
        vec!["Number of iterations".cell(), iterations.to_string().cell()],
        vec!["Parallel only".cell(), parallel_only.to_string().cell()],
        vec!["Tile size".cell(), tile_size.to_string().cell()],
    ];
    print_args_table(table);

    let mut algorithms_to_benchmark = VecDeque::with_capacity(4);
    if parallel_only == false {
        if skip_ijk == false {
            algorithms_to_benchmark.push_back(Algorithm::SequentialIjk);
        }
        algorithms_to_benchmark.push_back(Algorithm::SequentialIkj);
    }
    algorithms_to_benchmark.push_back(Algorithm::ParallelILoop(threads));
    algorithms_to_benchmark.push_back(Algorithm::ParallelTiling(threads, tile_size));

    print_title("Benchmarking!");

    for i in 0..iterations {
        println!("starting iteration {} of {}", i + 1, iterations);

        let a = random_filled_square_matrix_of_size!(n);
        let b = random_filled_square_matrix_of_size!(n);

        for algorithm in algorithms_to_benchmark.iter() {
            let time = time_algorithm(*algorithm, &a, &b).unwrap_or_default();
            match algorithm {
                Algorithm::SequentialIjk => sequential_ijk_times.push(time),
                Algorithm::SequentialIkj => sequential_ikj_times.push(time),
                Algorithm::ParallelILoop(_) => parallel_i_loop_times.push(time),
                Algorithm::ParallelTiling(_, _) => parallel_tiling_times.push(time),
            }
            println!("finished algorithm {:?}", algorithm);
            debug!("finished algorithm {:?}", algorithm);
        }
        println!();
    }

    // calculate average execution times

    let sequential_ijk_average = sequential_ijk_times.iter().sum::<u128>() / iterations as u128;
    let sequential_ikj_average = sequential_ikj_times.iter().sum::<u128>() / iterations as u128;
    let parallel_ijk_average = parallel_i_loop_times.iter().sum::<u128>() / iterations as u128;
    let parallel_tiling_average = parallel_tiling_times.iter().sum::<u128>() / iterations as u128;

    print_title("Benchmark Results");
    if parallel_only == false {
        if skip_ijk == false {
            println!("sequential ijk average: {} ms", sequential_ijk_average);
        }
        println!("sequential ikj average: {} ms", sequential_ikj_average);
    }
    println!("parallel i-loop average: {} ms", parallel_ijk_average);
    println!("parallel tiling average: {} ms", parallel_tiling_average);
}

/// Subprogram benchmarking the performance of different tiling strategies.
pub fn tiling_benchmark(cli: &Tiling) {
    let n = cli.size;
    let iterations = cli.iterations;
    let available_threads = thread::available_parallelism().unwrap().get();
    let threads = cli.threads.unwrap_or(available_threads);
    let tiles = match parse_cli_tiles(&cli.tiles) {
        Ok(tiles) => tiles,
        Err(err) => {
            println!("{}", err);
            error!("{}", err);
            return;
        }
    };

    print_title("Welcome to Tiling Benchmark!");
    let table = vec![
        vec!["Matrix size".cell(), n.to_string().cell()],
        vec!["Number of threads".cell(), threads.to_string().cell()],
        vec!["Number of iterations".cell(), iterations.to_string().cell()],
        vec!["Tiles".cell(), format!("{:?}", tiles).cell()],
    ];
    print_args_table(table);

    let mut times: Vec<Vec<u128>> = Vec::with_capacity(tiles.len());

    for _ in 0..tiles.len() {
        times.push(Vec::with_capacity(iterations));
    }

    print_title("Benchmarking!");

    for i in 0..iterations {
        println!("starting iteration {} of {}", i + 1, iterations);

        let a = random_filled_square_matrix_of_size!(n);
        let b = random_filled_square_matrix_of_size!(n);

        for j in 0..tiles.len() {
            let tile = tiles[j];
            let time = time_algorithm(Algorithm::ParallelTiling(threads, tile), &a, &b)
                .unwrap_or_default();
            times[j].push(time);
            println!("finished tile size {}", tile);
            debug!("finished tile size {}", tile);
        }

        println!();
    }

    print_title("Benchmark Results");
    for i in 0..tiles.len() {
        let tile = tiles[i];
        let average = times[i].iter().sum::<u128>() / iterations as u128;
        println!(
            "parallel tiling average for tile size {}: {} ms",
            tile, average
        );
    }
}
