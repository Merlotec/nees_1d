use crate::solver::Agent;

pub fn indifferent_price<F: num::Float, A>(
    agent: &A,
    quality: F,
    u_0: F,
    x_min: F,
    x_max: F,
    epsilon: F,
    max_iter: usize,
) -> Option<F>
where
    A: Agent<FloatType = F>,
{
    let mut lower = x_min;
    let mut upper = x_max;
    let mut iter = 0;

    while iter < max_iter {
        let mid = (lower + upper) / F::from(2.0).unwrap();
        let u_mid = agent.utility(mid, quality);
        let diff = u_mid - u_0;

        if diff.abs() < epsilon {
            return Some(mid);
        }

        // Because the utility function is increasing in quality, we swap this from the solver for quality.
        if diff > F::zero() {
            lower = mid;
        } else {
            upper = mid;
        }

        iter += 1;
    }

    // Return NaN if the solution was not found within tolerance
    None
}
