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
    pub unsafe fn add_mut(self, offset: usize) -> &'static mut i32 {
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
        let a_ptr = MatrixRowPtr(a.as_mut_ptr());

        unsafe {
            assert_eq!(*a_ptr.add_mut(0), 1);
            assert_eq!(*a_ptr.add_mut(1), 2);
            assert_eq!(*a_ptr.add_mut(2), 3);
        }
    }
}
