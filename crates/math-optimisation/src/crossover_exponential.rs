use ndarray::Array1;
use rand::Rng;

#[allow(dead_code)]
pub(crate) fn exponential_crossover<R: Rng + ?Sized>(
    target: &Array1<f64>,
    mutant: &Array1<f64>,
    cr: f64,
    rng: &mut R,
) -> Array1<f64> {
    let mut trial = target.clone();
    exponential_crossover_into(
        trial.as_slice_mut().expect("contiguous"),
        target.as_slice().expect("contiguous"),
        mutant.as_slice().expect("contiguous"),
        cr,
        rng,
    );
    trial
}

pub(crate) fn exponential_crossover_into<R: Rng + ?Sized>(
    out: &mut [f64],
    target: &[f64],
    mutant: &[f64],
    cr: f64,
    rng: &mut R,
) {
    let n = out.len();
    debug_assert_eq!(n, target.len());
    debug_assert_eq!(n, mutant.len());
    out.copy_from_slice(target);
    let mut j = rng.random_range(0..n);
    let mut l = 0usize;
    // ensure at least one parameter from mutant
    loop {
        out[j] = mutant[j];
        l += 1;
        j = (j + 1) % n;
        if rng.random::<f64>() >= cr || l >= n {
            break;
        }
    }
}
