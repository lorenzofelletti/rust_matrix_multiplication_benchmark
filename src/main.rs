extern crate core;

use std::{thread, time::Instant};

use clap::Parser;
use log::debug;

use crate::{
    cli::Cli, matrix_multiplication::algorithms::Algorithm,
    matrix_multiplication::matrix_multiplication,
};

mod cli;
mod matrix_multiplication;
mod thread_pool;

fn matrix_multiplication_benchmark(cli: &Cli) {
    let n = cli.size;
    let iterations = cli.iterations;
    let available_threads = thread::available_parallelism().unwrap().get();
    let threads = cli.threads.unwrap_or(available_threads);
    let parallel_only: bool = cli.parallel_only;

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

    for i in 0..iterations {
        println!("starting iteration {} of {}", i + 1, iterations);

        let a = random_filled_square_matrix_of_size!(n);
        let b = random_filled_square_matrix_of_size!(n);

        if parallel_only == false {
            let start = Instant::now();
            let _c = matrix_multiplication(&a, &b, Algorithm::SequentialIjk);
            let end = Instant::now();
            sequential_ijk_times.push(end.duration_since(start).as_millis());

            println!("finished sequential ijk");
            debug!("finished sequential ijk");

            let start = Instant::now();
            let _c = matrix_multiplication(&a, &b, Algorithm::SequentialIkj);
            let end = Instant::now();
            sequential_ikj_times.push(end.duration_since(start).as_millis());

            println!("finished sequential ikj");
            debug!("finished sequential ikj");
        }

        let start = Instant::now();
        let _c = matrix_multiplication(&a, &b, Algorithm::ParallelILoop(threads));
        let end = Instant::now();
        parallel_i_loop_times.push(end.duration_since(start).as_millis());

        println!("finished parallel i-loop");
        debug!("finished parallel i-loop");

        let start = Instant::now();
        let _c = matrix_multiplication(&a, &b, Algorithm::ParallelTiling(threads, 32));
        let end = Instant::now();
        parallel_tiling_times.push(end.duration_since(start).as_millis());

        println!("finished parallel tiling");
        debug!("finished parallel tiling");

        println!();
    }

    // calculate average execution times

    let sequential_ijk_average = sequential_ijk_times.iter().sum::<u128>() / iterations as u128;
    let sequential_ikj_average = sequential_ikj_times.iter().sum::<u128>() / iterations as u128;
    let parallel_ijk_average = parallel_i_loop_times.iter().sum::<u128>() / iterations as u128;
    let parallel_tiling_average = parallel_tiling_times.iter().sum::<u128>() / iterations as u128;

    // print results

    println!("Benchmark Results");
    if parallel_only == false {
        println!("sequential ijk average: {} ms", sequential_ijk_average);
        println!("sequential ikj average: {} ms", sequential_ikj_average);
    }
    println!("parallel i-loop average: {} ms", parallel_ijk_average);
    println!("parallel tiling average: {} ms", parallel_tiling_average);
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
        None => {
            matrix_multiplication_benchmark(&cli);
        }
    }
}
