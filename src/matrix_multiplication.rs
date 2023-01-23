use thread_pool::ThreadPool;
use types::{MatrixRowMutPtr, MatrixRowPtr};

use crate::{thread_pool, zero_filled_square_matrix_of_size};

use self::{
    algorithms::Algorithm,
    sanitize::{sanitize_matrices, SanitizeError},
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

#[cfg(test)]
mod tests {
    use std::{num::NonZeroUsize, thread};

    fn get_a() -> Vec<Vec<i32>> {
        vec![vec![1, 2], vec![3, 4]]
    }

    fn get_b() -> Vec<Vec<i32>> {
        vec![vec![5, 6], vec![7, 8]]
    }

    #[test]
    fn test_matrix_multiplication_sequential_ijk() {
        let a = get_a();
        let b = get_b();

        let c = super::matrix_multiplication_sequential_ijk(&a, &b, a.len()).unwrap();

        assert_eq!(c, vec![vec![19, 22], vec![43, 50]]);
    }

    #[test]
    fn test_matrix_multiplication_sequential_ikj() {
        let a = get_a();
        let b = get_b();

        let c = super::matrix_multiplication_sequential_ikj(&a, &b, a.len()).unwrap();

        assert_eq!(c, vec![vec![19, 22], vec![43, 50]]);
    }

    #[test]
    fn test_matrix_multiplication_parallel_i_loop() {
        let a = get_a();
        let b = get_b();

        let threads: usize = thread::available_parallelism()
            .unwrap_or(NonZeroUsize::new(1).unwrap())
            .into();

        let c = super::matrix_multiplication_parallel_i_loop(&a, &b, a.len(), threads).unwrap();

        assert_eq!(c, vec![vec![19, 22], vec![43, 50]]);
    }
}
