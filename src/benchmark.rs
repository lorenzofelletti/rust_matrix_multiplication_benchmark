use std::{collections::HashMap, thread, time::Instant, vec};

use cli_table::Cell;
use colored::Colorize;
use log::{debug, error};

use crate::{
    cli::{parse_cli_tiles, Cli, Tiling},
    cli_tables::{print_args_table, print_benchmark_results_table, print_title},
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
            println!(
                "{} while executing algorithm: {}",
                "Error".bright_red(),
                err
            );
            error!("Error while executing algorithm: {}", err);
            None
        }
    }
}

/// Runs one iteration of the benchmark suite.
fn run_benchmark_iteration(algorithms: &[Algorithm], size: usize) -> HashMap<Algorithm, u128> {
    let a = random_filled_square_matrix_of_size!(size);
    let b = random_filled_square_matrix_of_size!(size);
    let mut results = HashMap::with_capacity(algorithms.len());
    for algorithm in algorithms {
        let time = time_algorithm(*algorithm, &a, &b).unwrap_or_default();
        println!("Finished algorithm: {:?} in {:?} ms", algorithm, time);
        debug!("Finished algorithm: {:?} in {:?} ms", algorithm, time);
        results.insert(*algorithm, time);
    }
    results
}

/// Runs the benchmark suite for a given number of iterations.
fn run_benchmark(
    algorithms: &[Algorithm],
    iterations: usize,
    size: usize,
) -> HashMap<Algorithm, Vec<u128>> {
    let mut results: HashMap<Algorithm, Vec<u128>> = HashMap::with_capacity(algorithms.len());
    for i in 0..iterations {
        println!("Running iteration {}/{}", i + 1, iterations);
        debug!("Running iteration {}/{}", i + 1, iterations);
        let iteration_results = run_benchmark_iteration(algorithms, size);
        for (algorithm, time) in iteration_results {
            results.entry(algorithm).or_insert_with(Vec::new).push(time);
        }
    }
    results
}

/// Runs the benchmark on the specified algorithms for the specified number of iterations, and prints
/// the results.
///
/// # Arguments
///
/// * `algorithms` - The algorithms to benchmark.
/// * `iterations` - The number of iterations to run the benchmark for.
fn benchmark_and_print_results(algorithms: &[Algorithm], iterations: usize, size: usize) {
    print_title("Benchmarking!");

    let results = run_benchmark(&algorithms, iterations, size)
        .iter()
        .map(|(algorithm, times)| {
            let average = times.iter().sum::<u128>() / iterations as u128;
            (*algorithm, average)
        })
        .collect::<HashMap<_, _>>();

    print_title("Benchmark Results");

    let benchmark_results_table = results
        .iter()
        .map(|(algorithm, time)| vec![algorithm.to_string().cell(), time.to_string().cell()])
        .collect::<Vec<_>>();
    print_benchmark_results_table(benchmark_results_table);
}

pub fn matrix_multiplication_benchmark(cli: &Cli) {
    let n = cli.size;
    let iterations = cli.iterations;
    let available_threads = thread::available_parallelism().unwrap().get();
    let threads = cli.threads.unwrap_or(available_threads);
    let parallel_only: bool = cli.parallel_only;
    let tile_size = cli.tile_size;
    let skip_ijk = cli.skip_sequential_ijk;

    print_title("Welcome to Matrix Multiplication Benchmark!");

    let table = vec![
        vec!["Matrix size".cell(), n.to_string().cell()],
        vec!["Number of threads".cell(), threads.to_string().cell()],
        vec!["Number of iterations".cell(), iterations.to_string().cell()],
        vec!["Parallel only".cell(), parallel_only.to_string().cell()],
        vec!["Tile size".cell(), tile_size.to_string().cell()],
    ];
    print_args_table(table);

    let mut algorithms = Vec::with_capacity(4);
    if parallel_only == false {
        if skip_ijk == false {
            algorithms.push(Algorithm::SequentialIjk);
        }
        algorithms.push(Algorithm::SequentialIkj);
    }
    algorithms.push(Algorithm::ParallelILoop(threads));
    algorithms.push(Algorithm::ParallelTiling(threads, tile_size));

    benchmark_and_print_results(&algorithms, iterations, n);
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

    let algorithms = tiles
        .iter()
        .map(|tile| Algorithm::ParallelTiling(threads, *tile))
        .collect::<Vec<_>>();

    benchmark_and_print_results(&algorithms, iterations, n);
}
