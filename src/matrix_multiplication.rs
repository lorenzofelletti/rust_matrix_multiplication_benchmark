use std::mem;

use thread_pool::ThreadPool;
use types::MatrixRowPtr;

use crate::thread_pool;

use self::{
    algorithms::Algorithm,
    sanitize::{extra_sanitization_steps_for_tiling_algorithm, sanitize_matrices, SanitizeError},
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
    match algorithm {
        Algorithm::ParallelTiling(_, tile_size) => {
            extra_sanitization_steps_for_tiling_algorithm(a.len(), tile_size)?
        }
        _ => (),
    };

    let size = a.len();

    let a = a.into_iter().flatten().map(|x| *x).collect::<Vec<i32>>();
    let b = b.into_iter().flatten().map(|x| *x).collect::<Vec<i32>>();

    let c = match algorithm {
        Algorithm::SequentialIjk => matrix_multiplication_sequential_ijk(&a, &b, size),
        Algorithm::SequentialIkj => matrix_multiplication_sequential_ikj(&a, &b, size),
        Algorithm::ParallelILoop(threads) => {
            matrix_multiplication_parallel_i_loop(&a, &b, size, threads)
        }
        Algorithm::ParallelTiling(threads, tile_size) => {
            matrix_multiplication_parallel_tiling(&a, &b, size, tile_size, threads)
        }
    };
    Ok(c.into_iter()
        .map(|x| x.into_iter().map(|x| x as i32).collect())
        .collect())
}

fn matrix_multiplication_sequential_ijk(
    a: &Vec<i32>,
    b: &Vec<i32>,
    size: usize,
) -> Result<Vec<i32>, SanitizeError> {
    let mut c: Vec<i32> = vec![0; size * size];

    let a = a.as_ptr();
    let b = b.as_ptr();
    let c_ptr = c.as_mut_ptr();

    for i in 0..size {
        for j in 0..size {
            for k in 0..size {
                unsafe {
                    *c_ptr.add(i * size + j) += *a.add(i * size + k) * *b.add(k * size + j);
                }
            }
        }
    }

    Ok(c)
}

fn matrix_multiplication_sequential_ikj(
    a: &Vec<i32>,
    b: &Vec<i32>,
    size: usize,
) -> Result<Vec<i32>, SanitizeError> {
    let mut c: Vec<i32> = vec![0; size * size];

    let a = a.as_ptr();
    let b = b.as_ptr();
    let c_ptr = c.as_mut_ptr();

    for i in 0..size {
        for k in 0..size {
            for j in 0..size {
                unsafe {
                    *c_ptr.add(i * size + j) += *a.add(i * size + k) * *b.add(k * size + j);
                }
            }
        }
    }

    Ok(c)
}

fn matrix_multiplication_parallel_i_loop(
    a: &Vec<i32>,
    b: &Vec<i32>,
    size: usize,
    preferred_number_of_threads: usize,
) -> Result<Vec<i32>, SanitizeError> {
    let out_vec_len = size * size;
    let mut c: Vec<i32> = vec![0; out_vec_len];

    let pool = ThreadPool::new(preferred_number_of_threads);

    let a = MatrixRowPtr(a.as_ptr());
    let b = MatrixRowPtr(b.as_ptr());
    let c_ptr = MatrixRowPtr(c.as_mut_ptr());

    for i in 0..size {
        pool.execute(move || {
            let a = a;
            let b = b;
            let c_ptr = c_ptr;
            for k in 0..size {
                for j in 0..size {
                    unsafe {
                        *c_ptr.add_mut(i * size + j) += *a.add(i * size + k) * *b.add(k * size + j);
                    }
                }
            }
        });
    }

    ThreadPool::terminate(pool);

    Ok(c)
}

fn matrix_multiplication_parallel_tiling(
    a: &Vec<i32>,
    b: &Vec<i32>,
    size: usize,
    tile_size: usize,
    threads: usize,
) -> Result<Vec<i32>, SanitizeError> {
    let out_vec_len = size * size;
    let mut c: Vec<i32> = vec![0; out_vec_len];

    let a = MatrixRowPtr(a.as_ptr());
    let b = MatrixRowPtr(b.as_ptr());
    let c_ptr = MatrixRowPtr(c.as_mut_ptr());

    mem::forget(c); // forgets c so that it is not dropped, avoiding double free

    let pool = ThreadPool::new(threads);

    for l in (0..size).step_by(tile_size) {
        for w in (0..size).step_by(tile_size) {
            pool.execute(move || {
                let a = a;
                let b = b;
                let c_ptr = c_ptr;
                for kh in (0..size).step_by(tile_size) {
                    for i in 0..tile_size {
                        for k in 0..tile_size {
                            for j in 0..tile_size {
                                unsafe {
                                    *c_ptr.add_mut((l + i) * size + w + j) += *a
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
        c = Vec::from_raw_parts(c_ptr.0, out_vec_len, out_vec_len);
    }
    Ok(c)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{num::NonZeroUsize, thread};

    fn get_a() -> Vec<i32> {
        vec![1, 2, 3, 4]
    }

    fn get_b() -> Vec<i32> {
        vec![5, 6, 7, 8]
    }

    fn get_c() -> Vec<i32> {
        vec![19, 22, 43, 50]
    }

    fn get_size() -> usize {
        2
    }

    #[test]
    fn test_matrix_multiplication_sequential_ijk() {
        let a = get_a();
        let b = get_b();

        let c = matrix_multiplication_sequential_ijk(&a, &b, get_size()).unwrap();

        assert_eq!(c, get_c());
    }

    #[test]
    fn test_matrix_multiplication_sequential_ikj() {
        let a = get_a();
        let b = get_b();

        let c = matrix_multiplication_sequential_ikj(&a, &b, get_size()).unwrap();

        assert_eq!(c, get_c());
    }

    #[test]
    fn test_matrix_multiplication_parallel_i_loop() {
        let a = get_a();
        let b = get_b();

        let threads: usize = thread::available_parallelism()
            .unwrap_or(NonZeroUsize::new(1).unwrap())
            .into();

        let c = matrix_multiplication_parallel_i_loop(&a, &b, get_size(), threads).unwrap();

        assert_eq!(c, get_c());
    }

    #[test]
    fn test_matrix_multiplication_parallel_tiling() {
        let a = get_a();
        let b = get_b();

        let threads: usize = thread::available_parallelism()
            .unwrap_or(NonZeroUsize::new(1).unwrap())
            .into();

        let c = matrix_multiplication_parallel_tiling(&a, &b, get_size(), 1, threads).unwrap();

        assert_eq!(c, get_c())
    }
}
