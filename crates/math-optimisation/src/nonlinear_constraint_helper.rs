use super::deconfig::DEConfig;
use super::types::VectorConstraintFn;
use ndarray::Array1;
use std::sync::Arc;

/// SciPy-like nonlinear constraint helper: vector-valued fun(x) with lb <= fun(x) <= ub.
#[derive(Clone)]
pub struct NonlinearConstraintHelper {
    /// Vector-valued constraint function.
    pub fun: VectorConstraintFn,
    /// Lower bounds for each constraint component.
    pub lb: Array1<f64>,
    /// Upper bounds for each constraint component.
    pub ub: Array1<f64>,
}

impl NonlinearConstraintHelper {
    /// Apply helper by emitting penalty closures per component.
    /// lb <= f_i(x) <= ub becomes two inequalities: f_i(x)-ub <= 0 and lb - f_i(x) <= 0.
    /// If lb==ub, emit an equality penalty for f_i(x)-lb.
    pub fn apply_to(&self, cfg: &mut DEConfig, weight_ineq: f64, weight_eq: f64) {
        let f = self.fun.clone();
        let lb = self.lb.clone();
        let ub = self.ub.clone();
        let m = lb.len().min(ub.len());
        for i in 0..m {
            let l = lb[i];
            let u = ub[i];
            let tol = 1e-12 * (l.abs() + u.abs()).max(1.0);
            // An absent bound must not turn an infinite tolerance into equality.
            if l.is_finite() && u.is_finite() && (u - l).abs() <= tol {
                let fi = f.clone();
                cfg.penalty_eq.push((
                    Arc::new(move |x: &Array1<f64>| {
                        let y = (fi)(x);
                        y[i] - l
                    }),
                    weight_eq,
                ));
            } else {
                if u != f64::INFINITY {
                    let fi_u = f.clone();
                    cfg.penalty_ineq.push((
                        Arc::new(move |x: &Array1<f64>| {
                            let y = (fi_u)(x);
                            y[i] - u
                        }),
                        weight_ineq,
                    ));
                }
                if l != f64::NEG_INFINITY {
                    let fi_l = f.clone();
                    cfg.penalty_ineq.push((
                        Arc::new(move |x: &Array1<f64>| {
                            let y = (fi_l)(x);
                            l - y[i]
                        }),
                        weight_ineq,
                    ));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DEConfigBuilder;

    fn penalties(lower: f64, upper: f64) -> DEConfig {
        let helper = NonlinearConstraintHelper {
            fun: Arc::new(|x: &Array1<f64>| x.clone()),
            lb: Array1::from_vec(vec![lower]),
            ub: Array1::from_vec(vec![upper]),
        };
        let mut config = DEConfigBuilder::new().build().expect("valid defaults");
        helper.apply_to(&mut config, 7.0, 11.0);
        config
    }

    #[test]
    fn upper_only_bound_has_finite_signed_residual() {
        let config = penalties(f64::NEG_INFINITY, 10.0);
        assert!(config.penalty_eq.is_empty());
        assert_eq!(config.penalty_ineq.len(), 1);
        let (residual, weight) = &config.penalty_ineq[0];
        assert_eq!(*weight, 7.0);
        assert_eq!(residual(&Array1::from_vec(vec![2.0])), -8.0);
        assert_eq!(residual(&Array1::from_vec(vec![12.0])), 2.0);
    }

    #[test]
    fn lower_only_bound_has_finite_signed_residual() {
        let config = penalties(1.0, f64::INFINITY);
        assert!(config.penalty_eq.is_empty());
        assert_eq!(config.penalty_ineq.len(), 1);
        let (residual, _) = &config.penalty_ineq[0];
        assert_eq!(residual(&Array1::from_vec(vec![2.0])), -1.0);
        assert_eq!(residual(&Array1::from_vec(vec![0.0])), 1.0);
    }

    #[test]
    fn unbounded_component_adds_no_penalty() {
        let config = penalties(f64::NEG_INFINITY, f64::INFINITY);
        assert!(config.penalty_eq.is_empty());
        assert!(config.penalty_ineq.is_empty());
    }

    #[test]
    fn finite_equality_retains_signed_residual_and_weight() {
        let config = penalties(1.0, 1.0);
        assert!(config.penalty_ineq.is_empty());
        assert_eq!(config.penalty_eq.len(), 1);
        let (residual, weight) = &config.penalty_eq[0];
        assert_eq!(*weight, 11.0);
        assert_eq!(residual(&Array1::from_vec(vec![0.0])), -1.0);
        assert_eq!(residual(&Array1::from_vec(vec![2.0])), 1.0);
    }

    #[test]
    fn finite_interval_penalizes_both_violated_sides() {
        let config = penalties(1.0, 3.0);
        assert!(config.penalty_eq.is_empty());
        assert_eq!(config.penalty_ineq.len(), 2);
        for value in [0.0_f64, 2.0, 4.0] {
            let point = Array1::from_vec(vec![value]);
            let total: f64 = config
                .penalty_ineq
                .iter()
                .map(|(residual, weight)| weight * residual(&point).max(0.0).powi(2))
                .sum();
            assert_eq!(total, if value == 2.0 { 0.0 } else { 7.0 });
        }
    }
}
