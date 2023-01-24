extern crate core;

use std::{collections::VecDeque, thread, time::Instant};

use clap::Parser;
use cli::Tiling;
use log::{debug, error};

use crate::{
    cli::Cli, matrix_multiplication::algorithms::Algorithm,
    matrix_multiplication::matrix_multiplication,
};

mod cli;
mod matrix_multiplication;
mod thread_pool;

/// Benchmarks the execution time of a given matrix multiplication algorithm.
/// Returns the execution time in milliseconds, or `None` if an error occurred.
/// If an error occurs, the error is logged and printed to the console.
fn time_algorithm(algorithm: Algorithm, a: &Vec<Vec<i32>>, b: &Vec<Vec<i32>>) -> Option<u128> {
    let start = Instant::now();
    let res = matrix_multiplication(&a, &b, algorithm);
    let end = Instant::now();
    match res {
        Ok(_) => Some(end.duration_since(start).as_millis()),
        Err(err) => {
            println!("Error while executing algorithm: {}", err);
            error!("Error while executing algorithm: {}", err);
            None
        }
    }
}

fn matrix_multiplication_benchmark(cli: &Cli) {
    let n = cli.size;
    let iterations = cli.iterations;
    let available_threads = thread::available_parallelism().unwrap().get();
    let threads = cli.threads.unwrap_or(available_threads);
    let parallel_only: bool = cli.parallel_only;
    let tile_size = cli.tile_size;

    // collect execution times for different matrix multiplication methods

    let mut sequential_ijk_times = Vec::new();
    let mut sequential_ikj_times = Vec::new();
    let mut parallel_i_loop_times = Vec::new();
    let mut parallel_tiling_times = Vec::new();

    println!("Welcome to Matrix Multiplication Benchmark!");
    println!("Matrix size: {}", n);
    println!("Number of threads: {}", threads);
    println!("Number of iterations: {}", iterations);
    println!("Parallel only: {}", parallel_only);
    println!("Tile size: {}", tile_size);

    let mut algorithms_to_benchmark = VecDeque::with_capacity(4);
    if parallel_only == false {
        algorithms_to_benchmark.push_back(Algorithm::SequentialIjk);
        algorithms_to_benchmark.push_back(Algorithm::SequentialIkj);
    }
    algorithms_to_benchmark.push_back(Algorithm::ParallelILoop(threads));
    algorithms_to_benchmark.push_back(Algorithm::ParallelTiling(threads, tile_size));

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

    println!("Benchmark Results");
    if parallel_only == false {
        println!("sequential ijk average: {} ms", sequential_ijk_average);
        println!("sequential ikj average: {} ms", sequential_ikj_average);
    }
    println!("parallel i-loop average: {} ms", parallel_ijk_average);
    println!("parallel tiling average: {} ms", parallel_tiling_average);
}

fn parse_cli_tiles(tiles_string: &String) -> Result<Vec<usize>, String> {
    let spit_tiles: Vec<_> = tiles_string.split(",").collect();

    let tiles = spit_tiles
        .iter()
        .map(|tile| tile.parse::<usize>())
        .collect::<Result<Vec<_>, _>>();

    match tiles {
        Ok(tiles) => Ok(tiles),
        Err(_) => Err("tiles must be a positive integer".to_string()),
    }
}

/// Subprogram benchmarking the performance of different tiling strategies.
fn tiling_benchmark(cli: &Tiling) {
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

    println!("Welcome to Tiling Benchmark!");
    println!("Subprogram: tiling");
    println!("Matrix size: {}", n);
    println!("Number of threads: {}", threads);
    println!("Number of iterations: {}", iterations);
    println!("Tiles: {:?}", tiles);

    let mut times: Vec<Vec<u128>> = Vec::with_capacity(tiles.len());

    for _ in 0..tiles.len() {
        times.push(Vec::with_capacity(iterations));
    }

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

    println!("Benchmark Results");
    for i in 0..tiles.len() {
        let tile = tiles[i];
        let average = times[i].iter().sum::<u128>() / iterations as u128;
        println!(
            "parallel tiling average for tile size {}: {} ms",
            tile, average
        );
    }
}

fn main() {
    let cli = Cli::parse();

    match &cli.subcommands {
        Some(cli::Commands::OsThreads) => {
            print!(
                "number of os threads: {}",
                thread::available_parallelism().unwrap()
            );
        }
        Some(cli::Commands::Tiling(args)) => {
            tiling_benchmark(&args);
        }
        None => {
            matrix_multiplication_benchmark(&cli);
        }
    }
}
