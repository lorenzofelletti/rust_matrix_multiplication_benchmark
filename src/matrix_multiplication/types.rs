/// Struct holding pointers to `MatrixRowPtr` type
#[derive(Clone)]
pub struct SquareMatrixPtr(pub Vec<MatrixRowPtr<*const i32>>);

impl SquareMatrixPtr {
    /// Create new `SquareMatrixPtr` from `Vec<Vec<i32>>`
    pub fn new(matrix: &Vec<Vec<i32>>) -> SquareMatrixPtr {
        let mut matrix_ptr = Vec::new();

        for row in matrix {
            matrix_ptr.push(MatrixRowPtr(row.as_ptr()));
        }

        SquareMatrixPtr(matrix_ptr)
    }

    /// Get row by index
    ///
    /// # Panics
    ///
    /// Panics if `row` is out of bounds
    pub fn get_row(&self, row: usize) -> &MatrixRowPtr<*const i32> {
        let size = self.0.len();
        if row > size {
            panic!("Row index out of bounds");
        }

        &self.0[row]
    }
}

unsafe impl Send for SquareMatrixPtr {}

/// Struct holding mutable pointers to `i32` type.
/// It represents a row of a matrix that can be modified
#[derive(Clone, Copy)]
pub struct MatrixRowPtr<T>(pub T);

impl MatrixRowPtr<*mut i32> {
    /// Get value by index
    ///
    /// # Arguments
    ///
    /// * `offset` - index of value
    ///
    /// # Safety
    ///
    /// This function is unsafe because it dereferences a raw pointer, and it
    /// is the caller's responsibility to ensure that the pointer is valid.
    pub unsafe fn add_mut(&mut self, offset: usize) -> &mut i32 {
        &mut *self.0.add(offset)
    }
}

impl MatrixRowPtr<*const i32> {
    /// Get value by index
    ///
    /// # Arguments
    ///
    /// * `offset` - index of value
    ///
    /// # Safety
    ///
    /// This function is unsafe because it dereferences a raw pointer, and it
    /// is the caller's responsibility to ensure that the pointer is valid.
    pub unsafe fn add(&self, offset: usize) -> &i32 {
        &*self.0.add(offset)
    }
}

unsafe impl<T> Send for MatrixRowPtr<T> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_square_matrix_ptr() {
        let a = vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]];
        let a_ptr = SquareMatrixPtr::new(&a);

        unsafe {
            let a_0 = a_ptr.get_row(0);
            assert_eq!(*a_0.add(0), 1);
            assert_eq!(*a_0.add(1), 2);
            assert_eq!(*a_0.add(2), 3);

            let a_1 = a_ptr.get_row(1);
            assert_eq!(*a_1.add(0), 4);
            assert_eq!(*a_1.add(1), 5);
            assert_eq!(*a_1.add(2), 6);

            let a_2 = a_ptr.get_row(2);
            assert_eq!(*a_2.add(0), 7);
            assert_eq!(*a_2.add(1), 8);
            assert_eq!(*a_2.add(2), 9);
        }
    }

    #[test]
    fn test_get_row_out_of_bounds() {
        let a = vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]];
        let a_ptr = SquareMatrixPtr::new(&a);

        assert!(std::panic::catch_unwind(|| {
            a_ptr.get_row(3);
        })
        .is_err());
    }

    #[test]
    fn test_matrix_row_ptr_add() {
        let a = vec![1, 2, 3];
        let a_ptr = MatrixRowPtr(a.as_ptr());

        unsafe {
            assert_eq!(*a_ptr.add(0), 1);
            assert_eq!(*a_ptr.add(1), 2);
            assert_eq!(*a_ptr.add(2), 3);
        }
    }

    #[test]
    fn test_matrix_row_mut_ptr_add() {
        let mut a = vec![1, 2, 3];
        let mut a_ptr = MatrixRowPtr(a.as_mut_ptr());

        unsafe {
            assert_eq!(*a_ptr.add_mut(0), 1);
            assert_eq!(*a_ptr.add_mut(1), 2);
            assert_eq!(*a_ptr.add_mut(2), 3);
        }
    }
}
