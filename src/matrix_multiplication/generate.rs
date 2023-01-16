const MAX_ABS_VALUE_DEFAULT: i32 = 11; // 11 results in a matrix with values from -10 to 10

/// Generates a square matrix of size `size` filled with zeros or random values between -10 and 10
///
/// # Arguments
///
/// * `size` - The size of the matrix
/// * `random_values` - If true, the matrix will be filled with random values between -10 and 10
///
/// # Returns
///
/// A square matrix of size `size` as a `Vec<Vec<i32>>`
/// ```
pub fn generate_square_matrix_of_size(size: usize, random_values: bool, max_abs_value: Option<i32>) -> Vec<Vec<i32>> {
    let mut matrix = Vec::with_capacity(size);

    let modulo = max_abs_value.unwrap_or(MAX_ABS_VALUE_DEFAULT);

    if modulo < 1 {
        panic!("max_abs_value must be greater than 0");
    }

    for _ in 0..size {
        let mut row = Vec::with_capacity(size);
        for _ in 0..size {
            if random_values {
                // random between -10 and 10
                row.push(rand::random::<i32>() % modulo);
            } else {
                row.push(0);
            }
        }
        matrix.push(row);
    }

    matrix
}

/// Generates a square matrix of size `size` filled with zeros
#[macro_export]
macro_rules! zero_filled_square_matrix_of_size {
    ($size: expr) => {
        $crate::matrix_multiplication::generate::generate_square_matrix_of_size($size, false, None)
    };
}

/// Generates a square matrix of size `size` filled with random values between -10 and 10 (inclusive)
/// if no `max_abs_value` is provided, otherwise between -`max_abs_value` and `max_abs_value` (inclusive)
#[macro_export]
macro_rules! random_filled_square_matrix_of_size {
    ($size: expr) => {
        $crate::matrix_multiplication::generate::generate_square_matrix_of_size($size, true, None)
    };
    ($size: expr, $max_abs_value: expr) => {
        $crate::matrix_multiplication::generate::generate_square_matrix_of_size($size, true, Some($max_abs_value))
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_square_matrix_of_size() {
        let matrix = zero_filled_square_matrix_of_size!(10);
        assert_eq!(matrix.len(), 10);
        assert_eq!(matrix[0].len(), 10);
        for row in matrix {
            for value in row {
                assert_eq!(value, 0);
            }
        }
    }

    #[test]
    fn test_generate_square_matrix_of_size_random_default_abs() {
        let matrix = random_filled_square_matrix_of_size!(10);
        assert_eq!(matrix.len(), 10);
        assert_eq!(matrix[0].len(), 10);
        assert!(matrix[0][0] >= -MAX_ABS_VALUE_DEFAULT && matrix[0][0] <= MAX_ABS_VALUE_DEFAULT);
        assert!(matrix[9][9] >= -MAX_ABS_VALUE_DEFAULT && matrix[9][9] <= MAX_ABS_VALUE_DEFAULT);
    }

    #[test]
    fn test_generate_square_matrix_of_size_random_custom_abs() {
        let max_abs_value = 5;
        let matrix = random_filled_square_matrix_of_size!(10, max_abs_value);
        assert_eq!(matrix.len(), 10);
        assert_eq!(matrix[0].len(), 10);
        assert!(matrix[0][0] >= -max_abs_value && matrix[0][0] <= max_abs_value);
        assert!(matrix[9][9] >= -max_abs_value && matrix[9][9] <= max_abs_value);
    }
}
