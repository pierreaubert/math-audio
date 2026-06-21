use ndarray::Array1;
use rand::Rng;

#[allow(dead_code)]
pub(crate) fn binomial_crossover<R: Rng + ?Sized>(
    target: &Array1<f64>,
    mutant: &Array1<f64>,
    cr: f64,
    rng: &mut R,
) -> Array1<f64> {
    let mut trial = target.clone();
    binomial_crossover_into(trial.as_slice_mut().expect("contiguous"), target.as_slice().expect("contiguous"), mutant.as_slice().expect("contiguous"), cr, rng);
    trial
}

pub(crate) fn binomial_crossover_into<R: Rng + ?Sized>(
    out: &mut [f64],
    target: &[f64],
    mutant: &[f64],
    cr: f64,
    rng: &mut R,
) {
    let n = out.len();
    debug_assert_eq!(n, target.len());
    debug_assert_eq!(n, mutant.len());
    let jrand = rng.random_range(0..n);
    out.copy_from_slice(target);
    for j in 0..n {
        if j == jrand || rng.random::<f64>() < cr {
            out[j] = mutant[j];
        }
    }
}
