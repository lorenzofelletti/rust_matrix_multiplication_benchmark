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
    env_logger::init_from_env(env_logger::Env::default().default_filter_or("info"));

    let cli = Cli::parse();

    match &cli.subcommands {
        Some(cli::Commands::OsThreads) => {
            println!(
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
