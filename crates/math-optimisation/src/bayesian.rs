//! Bayesian optimisation for expensive bounded continuous objectives.
//!
//! The implementation is intentionally domain-agnostic. It models objective
//! values with a Gaussian-process surrogate over normalised `[0, 1]^n`
//! coordinates, uses a Matérn-5/2 kernel with ARD lengthscales, and proposes
//! one or more candidates per iteration through EI or Monte-Carlo batch
//! q-EI/qEHVI acquisition.

use crate::CallbackAction;
use crate::error::{DEError, Result};
use ndarray::Array1;

mod bayes_opt_config;
mod cholesky;
mod constrained;
mod consts;
mod derive;
mod evaluate;
mod expected;
mod gaussian_process;
mod hypervolume;
mod hypervolume_samples;
mod misc;
mod normal;
mod pareto;
mod select;
mod solve;
mod stop;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_stop;
mod types;

pub use bayes_opt_config::*;
pub use constrained::{
    BayesOptConstraint, BayesOptConstraintFn, ConstrainedBayesOptCallback,
    ConstrainedBayesOptConfig, ConstrainedBayesOptIntermediate, ConstrainedBayesOptReport,
    constrained_bayesian_optimization,
};
pub use types::*;

use bayes_opt_config::candidate_pool;
use bayes_opt_config::fit_gp;
use bayes_opt_config::initial_design;
use bayes_opt_config::validate_config;
use derive::derive_candidate_pool_size;
use derive::derive_initial_samples;
use derive::derive_lengthscales;
use evaluate::evaluate_and_store;
use evaluate::evaluate_multi_and_store;
use gaussian_process::max_posterior_std;
use misc::denormalize;
use misc::make_rng;
use misc::reference_point;
use pareto::pareto_solutions;
use pareto::pareto_values;
use select::select_batch;
use select::select_ehvi_batch;

/// Minimise `f` with Gaussian-process Bayesian optimisation.
///
/// The objective receives parameters in the original coordinate system.
/// Internally the surrogate is fit on normalised `[0, 1]^n` coordinates.
pub fn bayesian_optimization<F>(f: &F, mut config: BayesOptConfig) -> Result<BayesOptReport>
where
    F: Fn(&Array1<f64>) -> f64 + Sync,
{
    validate_config(&config)?;
    let n = config.bounds.len();
    let batch_size = config.batch_size.max(1);
    let initial_samples = derive_initial_samples(n, &config).min(config.maxeval);
    let candidate_pool_size = derive_candidate_pool_size(n, &config);
    let lengthscales = derive_lengthscales(n, &config)?;
    let fixed_lengthscales = config.lengthscales.is_some();
    let mut rng = make_rng(config.seed);

    let mut samples = initial_design(&config, initial_samples);
    let mut xs_norm = Vec::with_capacity(config.maxeval);
    let mut ys = Vec::with_capacity(config.maxeval);
    let mut best_x = Array1::zeros(n);
    let mut best_y = f64::INFINITY;

    evaluate_and_store(
        f,
        &mut samples,
        &config,
        &mut xs_norm,
        &mut ys,
        &mut best_x,
        &mut best_y,
    );

    let mut nit = 0usize;
    let mut final_std = f64::INFINITY;
    let mut message = String::from("evaluation budget reached");
    let mut success = false;

    while ys.len() < config.maxeval {
        let gp = fit_gp(&xs_norm, &ys, &lengthscales, fixed_lengthscales, &config)?;
        let candidates = candidate_pool(&config, candidate_pool_size, &xs_norm, &mut rng);
        if candidates.is_empty() {
            message = String::from("candidate pool exhausted");
            break;
        }
        final_std = max_posterior_std(&gp, &candidates, &config.bounds, &config.parallel);

        if config.posterior_std_threshold > 0.0
            && final_std <= config.posterior_std_threshold
            && ys.len() >= initial_samples
        {
            success = true;
            message = format!(
                "posterior std {:.3e} below threshold {:.3e}",
                final_std, config.posterior_std_threshold
            );
            break;
        }

        let remaining = config.maxeval - ys.len();
        let q = batch_size.min(remaining);
        let mut selected = select_batch(&gp, &candidates, best_y, q, &config, &mut rng);
        evaluate_and_store(
            f,
            &mut selected,
            &config,
            &mut xs_norm,
            &mut ys,
            &mut best_x,
            &mut best_y,
        );

        nit += 1;
        if let Some(callback) = config.callback.as_mut() {
            let payload = BayesOptIntermediate {
                x: best_x.clone(),
                fun: best_y,
                iter: nit,
                nfev: ys.len(),
                posterior_std: final_std,
            };
            if matches!(callback(&payload), CallbackAction::Stop) {
                success = true;
                message = String::from("stopped by callback");
                break;
            }
        }
    }

    if nit > 0 && !success {
        success = true;
    }

    Ok(BayesOptReport {
        x: best_x,
        fun: best_y,
        success,
        message,
        nfev: ys.len(),
        nit,
        posterior_std: final_std,
    })
}

/// Minimise a vector objective with Monte-Carlo EHVI.
///
/// This uses one independent GP per objective, samples the joint posterior
/// over each proposed batch, and scores candidates by expected hypervolume
/// improvement. All objectives are minimised.
pub fn bayesian_multi_objective<F>(f: &F, config: BayesOptConfig) -> Result<BayesOptParetoReport>
where
    F: Fn(&Array1<f64>) -> Vec<f64> + Sync,
{
    bayesian_multi_objective_with_stop(f, config, &|| false)
}

/// Minimizes vector objectives with cooperative cancellation during EHVI optimization.
///
/// `should_stop` must be fast and safe to call concurrently. The first `true`
/// is latched. Checks occur before objective admission, between GP likelihood
/// trials, and between EHVI candidates. A running objective or numerical kernel
/// is allowed to finish; this API provides no hard latency bound. Parallel
/// evaluations already admitted are drained and included in the returned report.
/// A stopped report has `success == false` and message `"stop requested"`.
/// The scalar iteration callback in `config` is not used for vector objectives.
///
/// # Errors
/// Returns an error for invalid configuration or a failed surrogate fit.
pub fn bayesian_multi_objective_with_stop<F>(
    f: &F,
    config: BayesOptConfig,
    should_stop: &(dyn Fn() -> bool + Sync),
) -> Result<BayesOptParetoReport>
where
    F: Fn(&Array1<f64>) -> Vec<f64> + Sync,
{
    let stop = stop::StopCheck::new(should_stop);
    validate_config(&config)?;
    let n = config.bounds.len();
    let batch_size = config.batch_size.max(1);
    let initial_samples = derive_initial_samples(n, &config).min(config.maxeval);
    let candidate_pool_size = derive_candidate_pool_size(n, &config).clamp(32, 256);
    let lengthscales = derive_lengthscales(n, &config)?;
    let fixed_lengthscales = config.lengthscales.is_some();
    let mut rng = make_rng(config.seed);

    let mut initial = initial_design(&config, initial_samples);
    let mut xs_norm = Vec::with_capacity(config.maxeval);
    let mut values = Vec::with_capacity(config.maxeval);
    evaluate_multi_and_store(f, &mut initial, &config, &mut xs_norm, &mut values, &stop);

    let mut nit = 0usize;
    'optimization: while values.len() < config.maxeval && !stop.requested() {
        let m = values.first().map(|v| v.len()).unwrap_or(0);
        if m == 0 {
            return Err(DEError::InvalidConfig {
                message: "multi-objective function returned an empty objective vector".into(),
            });
        }

        let mut gps = Vec::with_capacity(m);
        for j in 0..m {
            let yj = values.iter().map(|v| v[j]).collect::<Vec<_>>();
            let Some(gp) = bayes_opt_config::fit_gp_with_stop(
                &xs_norm,
                &yj,
                &lengthscales,
                fixed_lengthscales,
                &config,
                &stop,
            )?
            else {
                break 'optimization;
            };
            gps.push(gp);
        }
        if stop.requested() {
            break;
        }

        let candidates = candidate_pool(&config, candidate_pool_size, &xs_norm, &mut rng);
        if candidates.is_empty() {
            break;
        }
        let current_front = pareto_values(&values);
        let reference = reference_point(&values);
        let remaining = config.maxeval - values.len();
        let q = batch_size.min(remaining);
        let mut selected = select_ehvi_batch(
            &gps,
            &candidates,
            select::EhviFront {
                values: &current_front,
                reference: &reference,
            },
            q,
            &config,
            &mut rng,
            &stop,
        );
        if stop.requested() {
            break;
        }
        let expected = values.len() + selected.len();
        evaluate_multi_and_store(f, &mut selected, &config, &mut xs_norm, &mut values, &stop);
        if values.len() != expected {
            break;
        }
        nit += 1;
    }

    stop.requested();
    let population = xs_norm
        .iter()
        .zip(values.iter())
        .map(|(x, objectives)| BayesParetoSolution {
            x: denormalize(x, &config.bounds),
            objectives: objectives.clone(),
        })
        .collect::<Vec<_>>();
    let pareto_front = pareto_solutions(&population);

    Ok(BayesOptParetoReport {
        pareto_front,
        population,
        nfev: values.len(),
        nit,
        success: nit > 0 && !stop.observed(),
        message: if stop.observed() {
            String::from("stop requested")
        } else if nit > 0 {
            String::from("evaluation budget reached")
        } else {
            String::from("initial design consumed evaluation budget")
        },
    })
}
