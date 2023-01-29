use clap::{Args, Parser, Subcommand};

const ABOUT: &str = "Matrix Multiplication Benchmark \n
A benchmark suite for evaluating the performance of different matrix multiplication algorithms. \n
There are two main types of algorithms: \n
    1. Sequential: Any algorithm that uses a single thread to perform the matrix multiplication. \n
    2. Parallel: Algorithms that take advantage of multiple threads to perform the matrix multiplication. \n
Both these types can take great advantage of cache locality and SIMD instructions.";

#[derive(Parser)]
#[command(author, version, about = ABOUT, long_about = None)]
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

    #[arg(short, long, action = clap::ArgAction::SetTrue)]
    /// Skip sequential ijk algorithm
    pub skip_sequential_ijk: bool,

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

const TILES_DEFAULTS: &str = "16,32,64";

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

    #[arg(short, long, default_value_t = String::from(TILES_DEFAULTS))]
    /// Tile sizes to test. Separate multiple values with commas.
    pub tiles: String,
}

pub fn parse_cli_tiles(tiles_string: &String) -> Result<Vec<usize>, String> {
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
