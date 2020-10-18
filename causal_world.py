from typing import Dict, Any, Tuple, Callable

import numpy as np
import pandas as pd
from numpy.random import normal
from scipy.linalg import orth
from sklearn.preprocessing import MinMaxScaler
from toolz import partial, merge


def generate_cov_matrix(mu: np.ndarray) -> np.ndarray:
    w = np.random.uniform(-1, 1, (len(mu), len(mu)))
    o = orth(w)
    var = np.abs(mu / 2)
    return o.T.dot(np.diag(var)).dot(o)


def sample_parameters(features: int) -> Dict[str, Any]:
    mean = np.random.normal(loc=0, scale=100, size=features+1)
    covariance = generate_cov_matrix(mean)

    t_effect = np.random.normal(0, 10, (features + 1, 1))
    beta0 = np.random.normal(0, 10, 1)
    y_coef = np.random.normal(10/mean[1:], 0.5, (features, 1))

    noise_tempering = np.random.uniform(1, 10, 1)
    daily_samples = int(np.random.lognormal(6))
    return dict(mean=mean,
                covariance=covariance,
                y_coef=y_coef,
                t_effect=t_effect,
                beta0=beta0,
                binary_treatment=True,
                noise_tempering=noise_tempering,
                daily_samples=daily_samples)


def sample_features(n: int,
                    mean: np.ndarray,
                    covariance: np.ndarray,
                    binary_treatment: bool,
                    treatment_fn: Callable[[np.ndarray], np.ndarray] = None) -> np.ndarray:

    X = np.random.multivariate_normal(mean, covariance, size=n)
    features = X[:, 1:]
    treatment = X[:, 0].reshape(-1, 1) if treatment_fn is None else treatment_fn(features)

    if binary_treatment:
        treatment = MinMaxScaler((1e-3, .999)).fit_transform(treatment)
        return np.concatenate([np.random.binomial(1, p=treatment), features], axis=1)
    else:
        return np.concatenate([treatment, features], axis=1)


def define_outcome(feature: np.ndarray, y_coef: np.ndarray, t_effect: np.ndarray, beta0: np.ndarray):
    treatment = feature[:, 0].reshape(-1, 1)
    treatment_interaction = treatment * feature[:, 1:]  # treatment times features
    treatment_features = np.concatenate([treatment, treatment_interaction, feature[:, 1:]], axis=1)
    return treatment_features.dot(np.concatenate([t_effect, y_coef])) + beta0


def random_outcome(outcome: np.ndarray, noise_tempering: float) -> np.ndarray:
    return np.random.normal(outcome, scale=abs(outcome.mean()) / noise_tempering)


def sample(**parameters) -> Tuple[pd.DataFrame, dict]:

    mean = parameters.get("mean")
    time = parameters.get("time")
    y_coef = parameters.get("y_coef")
    treatment_fn = parameters.get("treatment_fn")
    np.random.seed(time)
    feature = sample_features(parameters.get("daily_samples"),
                              mean,
                              parameters.get("covariance"),
                              parameters.get("binary_treatment", True),
                              treatment_fn)

    outcome = define_outcome(feature, y_coef, parameters.get("t_effect"), parameters.get("beta0"))
    outcome = random_outcome(outcome, parameters.get("noise_tempering"))

    data = pd.DataFrame(
        data=np.concatenate([feature, outcome], axis=1).round(2),
        columns=["T"] + [f"x_{f}" for f in range(1, len(y_coef)+1)] + ["Y"]
    )

    return data, dict(sampler=partial(sample, **merge(parameters, dict(time=time+1))), parameters=parameters)


def world_generator(features: int, parameters_override=None) -> Any:
    if parameters_override is None:
        parameters_override = dict()
    parameters = sample_parameters(features)
    return dict(sampler=partial(sample, time=0, **merge(parameters, parameters_override)),
                parameters=parameters)


def show_parameters(state, filter_parameter=None):
    assert filter_parameter is None or filter_parameter in state.get("parameters").keys(),\
        f"{filter_parameter} not a world parameter"
    for parameter in state.get("parameters"):
        if filter_parameter is None or filter_parameter == parameter:
            print(parameter, "\n",  state.get("parameters").get(parameter), "\n")
