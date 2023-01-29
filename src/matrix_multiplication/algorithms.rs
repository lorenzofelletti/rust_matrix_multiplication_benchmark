/// Enum representing available matrix multiplication algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Algorithm {
    SequentialIjk,
    SequentialIkj,
    /// Parallel algorithm using a loop over i
    ///
    /// # Arguments
    ///
    /// * `usize` - number of threads to use
    ParallelILoop(usize),
    /// Parallel algorithm using tiling
    ///
    /// # Arguments
    ///
    /// * `usize` - number of threads to use
    /// * `usize` - tile size
    ParallelTiling(usize, usize),
}

impl std::fmt::Display for Algorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Algorithm::SequentialIjk => write!(f, "Sequential IJK"),
            Algorithm::SequentialIkj => write!(f, "Sequential IKJ"),
            Algorithm::ParallelILoop(threads) => write!(f, "Parallel I Loop ({} threads)", threads),
            Algorithm::ParallelTiling(threads, tile_size) => {
                write!(
                    f,
                    "Parallel Tiling ({} threads, {} tile size)",
                    threads, tile_size
                )
            }
        }
    }
}
