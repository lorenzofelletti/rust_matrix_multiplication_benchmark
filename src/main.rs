extern crate core;

use std::thread;

use benchmark::{matrix_multiplication_benchmark, tiling_benchmark};
use clap::Parser;

use crate::cli::Cli;

mod benchmark;
mod cli;
mod cli_tables;
mod matrix_multiplication;
mod thread_pool;

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
