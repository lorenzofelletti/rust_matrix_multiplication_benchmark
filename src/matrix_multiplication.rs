use std::mem;

use thread_pool::ThreadPool;
use types::{MatrixRowMutPtr, MatrixRowPtr};

use crate::{thread_pool, zero_filled_square_matrix_of_size};

use self::{
    algorithms::Algorithm,
    sanitize::{sanitize_matrices, SanitizeError, size_multiple_of_tile_size},
    types::SquareMatrixPtr,
};

pub mod algorithms;
pub mod generate;
mod sanitize;
mod types;

pub fn matrix_multiplication(
    a: &Vec<Vec<i32>>,
    b: &Vec<Vec<i32>>,
    algorithm: Algorithm,
) -> Result<Vec<Vec<i32>>, SanitizeError> {
    sanitize_matrices(a, b)?;

    let size = a.len();

    match algorithm {
        Algorithm::SequentialIjk => matrix_multiplication_sequential_ijk(a, b, size),
        Algorithm::SequentialIkj => matrix_multiplication_sequential_ikj(a, b, size),
        Algorithm::ParallelILoop(threads) => {
            matrix_multiplication_parallel_i_loop(a, b, size, threads)
        }
        Algorithm::ParallelTiling(threads, tile_size) => {
            size_multiple_of_tile_size(size, tile_size)?;

            let res = matrix_multiplication_parallel_tiling(a, b, size, tile_size, threads)?;
            let c: Vec<Vec<i32>> = res.chunks(size).map(|row| row.to_vec()).collect();
            Ok(c)
        }
    }
}

fn matrix_multiplication_sequential_ijk(
    a: &Vec<Vec<i32>>,
    b: &Vec<Vec<i32>>,
    size: usize,
) -> Result<Vec<Vec<i32>>, SanitizeError> {
    let mut c = zero_filled_square_matrix_of_size!(size);

    let b = SquareMatrixPtr::new(b);

    for i in 0..size {
        let a_i = MatrixRowPtr(a[i].as_ptr());
        let mut c_i = MatrixRowMutPtr(c[i].as_mut_ptr());
        for j in 0..size {
            for k in 0..size {
                let b_k = b.get_row(k);
                unsafe {
                    *c_i.add(j) += *a_i.add(k) * *b_k.add(j);
                }
            }
        }
    }

    Ok(c)
}

fn matrix_multiplication_sequential_ikj(
    a: &Vec<Vec<i32>>,
    b: &Vec<Vec<i32>>,
    size: usize,
) -> Result<Vec<Vec<i32>>, SanitizeError> {
    let mut c = zero_filled_square_matrix_of_size!(size);

    let b = SquareMatrixPtr::new(b);

    for i in 0..size {
        let a_i = MatrixRowPtr(a[i].as_ptr());
        let mut c_i = MatrixRowMutPtr(c[i].as_mut_ptr());
        for k in 0..size {
            let b_k = b.get_row(k);
            for j in 0..size {
                unsafe {
                    *c_i.add(j) += *a_i.add(k) * *b_k.add(j);
                }
            }
        }
    }

    Ok(c)
}

fn matrix_multiplication_parallel_i_loop(
    a: &Vec<Vec<i32>>,
    b: &Vec<Vec<i32>>,
    size: usize,
    preferred_number_of_threads: usize,
) -> Result<Vec<Vec<i32>>, SanitizeError> {
    let mut c = zero_filled_square_matrix_of_size!(size);

    let pool = ThreadPool::new(preferred_number_of_threads);

    for i in 0..size {
        let a_i = MatrixRowPtr(a[i].as_ptr());
        let mut c_i = MatrixRowMutPtr(c[i].as_mut_ptr());
        let b = SquareMatrixPtr::new(b);

        unsafe {
            pool.execute(move || {
                for k in 0..size {
                    let b_k = b.get_row(k);
                    for j in 0..size {
                        *c_i.add(j) += *a_i.add(k) * *b_k.add(j);
                    }
                }
            });
        }
    }

    ThreadPool::terminate(pool);

    Ok(c)
}

fn matrix_multiplication_parallel_tiling(
    a: &Vec<Vec<i32>>,
    b: &Vec<Vec<i32>>,
    size: usize,
    tile_size: usize,
    threads: usize,
) -> Result<Vec<i32>, SanitizeError> {
    // check that matrix size is a multiple of tile size
    let mut c: Vec<i32> = Vec::with_capacity(size * size);
    let res_length = size * size;

    let a: Vec<i32> = a.into_iter().flatten().map(|x| *x).collect::<Vec<_>>();
    let b: Vec<i32> = b.into_iter().flatten().map(|x| *x).collect::<Vec<_>>();

    let a = MatrixRowPtr(a.as_ptr());
    let b = MatrixRowPtr(b.as_ptr());
    let c_ptr = MatrixRowMutPtr(c.as_mut_ptr());
    // forget c so that it is not dropped
    mem::forget(c);

    let pool = ThreadPool::new(threads);

    for l in (0..size).step_by(tile_size) {
        for w in (0..size).step_by(tile_size) {
            pool.execute(move || {
                let a = a;
                let b = b;
                let mut c_ptr = c_ptr;
                for kh in (0..size).step_by(tile_size) {
                    for i in 0..tile_size {
                        for k in 0..tile_size {
                            for j in 0..tile_size {
                                unsafe {
                                    *c_ptr.add((l + i) * size + w + j) += *a
                                        .add((l + i) * size + kh + k)
                                        * *b.add((kh + k) * size + w + j);
                                }
                            }
                        }
                    }
                }
            });
        }
    }

    ThreadPool::terminate(pool);

    let c: Vec<i32>;
    unsafe {
        c = Vec::from_raw_parts(c_ptr.0, res_length, res_length);
    }

    //let mut res: Vec<i32> = Vec::with_capacity(res_length);

    /*for i in 0..res_length {
        res.push(c[i]);
    }*/

    Ok(c) //res) //res)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{num::NonZeroUsize, thread};

    fn get_a() -> Vec<Vec<i32>> {
        vec![vec![1, 2], vec![3, 4]]
    }

    fn get_b() -> Vec<Vec<i32>> {
        vec![vec![5, 6], vec![7, 8]]
    }

    fn get_c() -> Vec<Vec<i32>> {
        vec![vec![19, 22], vec![43, 50]]
    }

    #[test]
    fn test_matrix_multiplication_sequential_ijk() {
        let a = get_a();
        let b = get_b();

        let c = matrix_multiplication_sequential_ijk(&a, &b, a.len()).unwrap();

        assert_eq!(c, get_c());
    }

    #[test]
    fn test_matrix_multiplication_sequential_ikj() {
        let a = get_a();
        let b = get_b();

        let c = matrix_multiplication_sequential_ikj(&a, &b, a.len()).unwrap();

        assert_eq!(c, get_c());
    }

    #[test]
    fn test_matrix_multiplication_parallel_i_loop() {
        let a = get_a();
        let b = get_b();

        let threads: usize = thread::available_parallelism()
            .unwrap_or(NonZeroUsize::new(1).unwrap())
            .into();

        let c = matrix_multiplication_parallel_i_loop(&a, &b, a.len(), threads).unwrap();

        assert_eq!(c, get_c());
    }

    #[test]
    fn test_matrix_multiplication_parallel_tiling() {
        let a = get_a();
        let b = get_b();

        let threads: usize = thread::available_parallelism()
            .unwrap_or(NonZeroUsize::new(1).unwrap())
            .into();

        let c = matrix_multiplication_parallel_tiling(&a, &b, a.len(), 1, threads).unwrap();

        assert_eq!(c, get_c().into_iter().flatten().collect::<Vec<_>>())
    }
}
