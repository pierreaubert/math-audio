use approx::assert_relative_eq;
use math_audio_autodiff::{
    matrix::{Matrix, MatrixType},
    module::DiffModule,
    tensor::DiffTensor,
};
use ndarray::Array3;
use num_complex::Complex;

const NFFT: usize = 512;

#[test]
fn orthogonal_matrix_stays_orthogonal_with_zero_params() {
    let n = 4;
    let matrix = Matrix::new(NFFT, n, n, MatrixType::Orthogonal).unwrap();
    let m = matrix.build_matrix().unwrap();
    let identity = m.t().dot(&m);
    for i in 0..n {
        for j in 0..n {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_relative_eq!(identity[[i, j]].re, expected, epsilon = 1e-6);
        }
    }
}

#[test]
fn dense_matrix_gradient_matches_finite_difference() {
    let n_bins = NFFT / 2 + 1;
    let mut matrix = Matrix::new(NFFT, 2, 1, MatrixType::Dense).unwrap();
    matrix.param[[0, 0]] = 0.5;
    matrix.param[[1, 0]] = -0.3;

    let input = DiffTensor::from_array(
        Array3::<Complex<f64>>::from_elem((1, n_bins, 1), Complex::new(1.0, 0.2)).into_dyn(),
    );
    let target = DiffTensor::from_array(
        Array3::<Complex<f64>>::from_elem((1, n_bins, 2), Complex::new(0.4, -0.1)).into_dyn(),
    );

    let eps = 1e-6;
    let mut numeric = ndarray::Array2::<f64>::zeros((2, 1));
    for i in 0..2 {
        matrix.param[[i, 0]] += eps;
        let out_plus = matrix.forward(&input).unwrap();
        let loss_plus = (&out_plus.data - &target.data)
            .iter()
            .map(|x| x.norm_sqr())
            .sum::<f64>();
        matrix.param[[i, 0]] -= 2.0 * eps;
        let out_minus = matrix.forward(&input).unwrap();
        let loss_minus = (&out_minus.data - &target.data)
            .iter()
            .map(|x| x.norm_sqr())
            .sum::<f64>();
        numeric[[i, 0]] = (loss_plus - loss_minus) / (2.0 * eps);
        matrix.param[[i, 0]] += eps;
    }

    matrix.zero_grad();
    let out = matrix.forward(&input).unwrap();
    let diff = &out.data - &target.data;
    let grad = DiffTensor::from_array(diff.into_owned() * 2.0);
    matrix.backward(&input, &out, &grad).unwrap();

    for i in 0..2 {
        assert_relative_eq!(matrix.param_grad[[i, 0]], numeric[[i, 0]], epsilon = 1e-5);
    }
}

#[test]
fn orthogonal_matrix_gradient_matches_finite_difference() {
    let n_bins = NFFT / 2 + 1;
    let n = 3;
    let mut matrix = Matrix::new(NFFT, n, n, MatrixType::Orthogonal).unwrap();
    // Non-zero raw parameters; matrix_exp_skew uses raw - raw.t().
    matrix.param[[0, 1]] = 0.4;
    matrix.param[[0, 2]] = -0.2;
    matrix.param[[1, 2]] = 0.3;

    let input = DiffTensor::from_array(
        Array3::<Complex<f64>>::from_elem((1, n_bins, n), Complex::new(1.0, 0.2)).into_dyn(),
    );
    let target = DiffTensor::from_array(
        Array3::<Complex<f64>>::from_elem((1, n_bins, n), Complex::new(0.4, -0.1)).into_dyn(),
    );

    let eps = 1e-6;
    let mut numeric = ndarray::Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            matrix.param[[i, j]] += eps;
            let out_plus = matrix.forward(&input).unwrap();
            let loss_plus = (&out_plus.data - &target.data)
                .iter()
                .map(|x| x.norm_sqr())
                .sum::<f64>();
            matrix.param[[i, j]] -= 2.0 * eps;
            let out_minus = matrix.forward(&input).unwrap();
            let loss_minus = (&out_minus.data - &target.data)
                .iter()
                .map(|x| x.norm_sqr())
                .sum::<f64>();
            numeric[[i, j]] = (loss_plus - loss_minus) / (2.0 * eps);
            matrix.param[[i, j]] += eps;
        }
    }

    matrix.zero_grad();
    let out = matrix.forward(&input).unwrap();
    let diff = &out.data - &target.data;
    let grad = DiffTensor::from_array(diff.into_owned() * 2.0);
    matrix.backward(&input, &out, &grad).unwrap();

    for i in 0..n {
        for j in 0..n {
            assert_relative_eq!(matrix.param_grad[[i, j]], numeric[[i, j]], epsilon = 1e-5);
        }
    }
}

#[test]
fn orthogonal_matrix_gradient_is_accurate_for_large_parameters() {
    let n_bins = NFFT / 2 + 1;
    let mut matrix = Matrix::new(NFFT, 2, 2, MatrixType::Orthogonal).unwrap();
    matrix.param[[0, 1]] = 50.0;
    matrix.param[[1, 0]] = -20.0;
    let input = DiffTensor::from_array(
        Array3::<Complex<f64>>::from_elem((1, n_bins, 2), Complex::new(0.7, -0.2)).into_dyn(),
    );
    let mut target_data = Array3::<Complex<f64>>::from_elem((1, n_bins, 2), Complex::new(0.1, 0.3));
    for bin in 0..n_bins {
        target_data[[0, bin, 1]] = Complex::new(-0.4, 0.2);
    }
    let target = DiffTensor::from_array(target_data.into_dyn());
    let output = matrix.forward(&input).unwrap();
    let grad_output = DiffTensor::from_array((&output.data - &target.data) * 2.0);
    matrix.zero_grad();
    matrix.backward(&input, &output, &grad_output).unwrap();

    let eps = 1e-4;
    for (i, j) in [(0, 1), (1, 0)] {
        let mut plus = matrix.clone();
        plus.param[[i, j]] += eps;
        let plus_output = plus.forward(&input).unwrap();
        let plus_loss = plus_output
            .data
            .iter()
            .zip(target.data.iter())
            .map(|(actual, expected)| (actual - expected).norm_sqr())
            .sum::<f64>();
        let mut minus = matrix.clone();
        minus.param[[i, j]] -= eps;
        let minus_output = minus.forward(&input).unwrap();
        let minus_loss = minus_output
            .data
            .iter()
            .zip(target.data.iter())
            .map(|(actual, expected)| (actual - expected).norm_sqr())
            .sum::<f64>();
        let numerical = (plus_loss - minus_loss) / (2.0 * eps);
        assert_relative_eq!(
            matrix.param_grad[[i, j]],
            numerical,
            epsilon = 1e-5,
            max_relative = 1e-5
        );
    }
}
