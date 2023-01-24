use clap::{Args, Parser, Subcommand};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
/// Matrix Multiplication Benchmark
pub struct Cli {
    #[arg(default_value_t = 128)]
    /// Size of the matrix
    pub size: usize,

    #[arg(short, long, default_value_t = 5)]
    /// Number of iterations to run the benchmark
    pub iterations: usize,

    #[arg(short, long)]
    /// Number of threads to use for parallel matrix multiplication [default: number of logical cores]
    pub threads: Option<usize>,

    #[arg(short, long, action = clap::ArgAction::SetTrue)]
    /// Only run parallel matrix multiplication
    pub parallel_only: bool,

    #[arg(long, default_value_t = 32)]
    /// Tile size for parallel tiling algorithm
    pub tile_size: usize,

    #[command(subcommand)]
    pub subcommands: Option<Commands>,
}

#[derive(Subcommand)]
pub enum Commands {
    #[command(name = "os_threads")]
    /// Print the number of available OS threads
    OsThreads,
    #[command(name = "tiling")]
    /// Run benchmark suite for parallel tiling algorithm
    Tiling(Tiling),
}

#[derive(Args)]
pub struct Tiling {
    #[arg(default_value_t = 128)]
    /// Size of the matrix
    pub size: usize,

    #[arg(short, long, default_value_t = 5)]
    /// Number of iterations to run the benchmark
    pub iterations: usize,

    #[arg(long)]
    /// Number of threads to use for parallel matrix multiplication [default: number of logical cores]
    pub threads: Option<usize>,

    #[arg(short, long, default_value_t = ("16,32,64".to_string()))]
    /// Tile sizes to test. Separate multiple values with commas.
    pub tiles: String,
}
