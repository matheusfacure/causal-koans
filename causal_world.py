from typing import Optional, Dict, Any, Callable, Tuple

import numpy as np
import pandas as pd
from numpy.random import normal
from scipy.linalg import orth
from toolz import partial, merge


def generate_cov_matrix(mu: np.ndarray) -> np.ndarray:
    w = np.random.uniform(-1, 1, (len(mu), len(mu)))
    o = orth(w)
    var = np.abs(mu / 2)
    return o.T.dot(np.diag(var)).dot(o)


def sample_parameters(features: int) -> Dict[str, Any]:
    mean = np.random.normal(loc=0, scale=100, size=features)
    covariance = generate_cov_matrix(mean)
    treatment_function_coef = np.random.normal(0, 10, (features + 1, 1))
    y_coef = np.random.normal(0, 10, (features + 1, 1))
    t_effect = np.random.normal(0, 10, (features + 1, 1))
    noise_tempering = np.random.uniform(1, 10, 1)
    daily_samples = int(np.random.lognormal(6))
    return dict(mean=mean,
                covariance=covariance,
                treatment_function_coef=treatment_function_coef,
                y_coef=y_coef,
                t_effect=t_effect,
                noise_tempering=noise_tempering,
                daily_samples=daily_samples)


def confounded_treatment(features_matrix: np.ndarray, treatment_function_coef: np.ndarray) -> np.ndarray:
    non_norm_p = features_matrix.dot(treatment_function_coef)
    p = (non_norm_p - non_norm_p.mean()) / non_norm_p.std()
    return (np.random.normal(p) > 0).astype(int)


def sample_features(n: int, mean: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    features = np.random.multivariate_normal(mean, covariance, size=n)
    # add constant
    return np.concatenate([np.ones((n, 1)), features], axis=1)


def define_treatment(feature: np.ndarray,
                     treatment_function_coef: np.ndarray,
                     treatment_fn: Optional[Callable[[np.ndarray], np.ndarray]]):

    if treatment_fn is not None:
        assert treatment_fn(feature).shape == (feature.shape[0], 1), "treatment_fn should return a column vector"

    return (treatment_fn(feature) if treatment_fn is not None
            else confounded_treatment(feature, treatment_function_coef))


def define_outcome(feature: np.ndarray,
                   treatment: np.ndarray,
                   y_coef: np.ndarray,
                   t_effect: np.ndarray):

    features_treatment = np.concatenate([feature, feature*treatment], axis=1)
    all_coefs = np.concatenate([y_coef, t_effect], axis=0)

    return features_treatment.dot(all_coefs)


def random_outcome(outcome: np.ndarray, noise_tempering: float) -> np.ndarray:
    return np.random.normal(outcome, scale=abs(outcome.mean()) / noise_tempering)


def sample(**parameters) -> Tuple[pd.DataFrame, dict]:

    mean = parameters.get("mean")
    time = parameters.get("time")
    np.random.seed(time)
    feature = sample_features(parameters.get("daily_samples"), mean, parameters.get("covariance"))

    treatment = define_treatment(feature,
                                 parameters.get("treatment_function_coef"),
                                 parameters.get("treatment_fn"))

    outcome = random_outcome(define_outcome(feature, treatment, parameters.get("y_coef"), parameters.get("t_effect")),
                             noise_tempering=parameters.get("noise_tempering"))

    data = pd.DataFrame(
        data=np.concatenate([feature, treatment, outcome], axis=1).round(2),
        columns=[f"x_{f}" for f in range(len(mean) + 1)] + ["T", "Y"]
    )

    return data, dict(sampler=partial(sample, **merge(parameters, dict(time=time+1))), parameters=parameters)


def world_generator(features: int, parameters_override: Any) -> Any:
    parameters = sample_parameters(features)
    return dict(sampler=partial(sample, time=0, **merge(parameters, parameters_override)),
                parameters=parameters)


def show_parameters(state, filter_parameter=None):
    assert filter_parameter is None or filter_parameter in state.get("parameters").keys(),\
        f"{filter_parameter} not a world parameter"
    for parameter in state.get("parameters"):
        if filter_parameter is None or filter_parameter == parameter:
            print(parameter, "\n",  state.get("parameters").get(parameter), "\n")
