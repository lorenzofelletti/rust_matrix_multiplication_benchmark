/// Enum representing available matrix multiplication algorithms
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
