import numpy as np
import pandas as pd
from causal_world import generate_cov_matrix, sample_parameters, confounded_treatment, \
    sample_features, define_treatment, define_outcome, random_outcome, sample


def data_frame_test(data, columns):
    return pd.DataFrame(np.array(data), columns=columns)


def test_generate_cov_matrix():
    np.random.seed(99)
    result = generate_cov_matrix(3)

    expected = np.array(
        [[7.85880814, 0.81218219, -0.49900906],
         [0.81218219, 13.34825859, -1.47700435],
         [-0.49900906, -1.47700435, 7.31515279]]
    )

    assert np.allclose(result, expected)

    # check if positive definite
    assert np.all(np.linalg.eigvals(result) > 0)

    # check if symmetric
    assert np.allclose(result, result.T)


def test_sample_parameters():
    np.random.seed(99)
    result = sample_parameters(4)

    expected_mean = np.array([-14.23588427, 205.72217376, 28.32619411, 132.98119783])

    expected_cov = np.array(
        [[7.27985875, 1.4376169, 0.84678988, 0.2790446],
         [1.4376169, 6.50193927, 0.65921642, -0.33876261],
         [0.84678988, 0.65921642, 6.25753432, -0.02908338],
         [0.2790446, -0.33876261, -0.02908338, 5.51484175]]
    )
    expected_treatment_function_coef = np.array(
        [[-7.3161625],
         [2.61755461],
         [-8.55795579],
         [-1.87525912],
         [-3.73486289]]
    )
    expected_y_coef = np.array([[-4.61970967],
                                [-8.16466104],
                                [-0.45123303],
                                [1.21327776],
                                [9.2595278]]),

    expected_t_interaction_coef = np.array([[-5.73819696],
                                            [0.5270311],
                                            [22.07310587],
                                            [3.91821869],
                                            [4.82713428]])

    assert np.allclose(result.get("mean"), expected_mean)
    assert np.allclose(result.get("covariance"), expected_cov)
    assert np.allclose(result.get("treatment_function_coef"), expected_treatment_function_coef)
    assert np.allclose(result.get("y_coef"), expected_y_coef)
    assert np.allclose(result.get("t_interaction_coef"), expected_t_interaction_coef)


def test_confounded_treatment():
    np.random.seed(12)
    features = np.array(
        [[1, 1, 2, 0.0, 1],
         [1, 1, 2, 1.0, 5],
         [1, 2, 3, 3.0, 0],
         [1, 2, 3, 4.0, 2]]
    )
    treatment_function_coef = np.array(
        [[-10],
         [1],
         [2],
         [0],
         [4]]
    )

    result = confounded_treatment(features, treatment_function_coef)

    expected = np.array([
        [0],
        [1],
        [0],
        [0]
    ])

    assert np.allclose(expected, result)


def test_sample_features():
    np.random.seed(99)
    result = sample_features(5,
                             np.array([1, 2]),
                             np.array([[1, 0.5],
                                       [0.5, 1]]))

    expected = np.array(
        [[1., 0.09467551, 3.15189724],
         [1., 0.08978197, 2.41959395],
         [1., 1.16842188, 2.09939102],
         [1., -0.06682881, 1.75881784],
         [1., 2.28183961, 0.91400201]]
    )

    assert np.allclose(expected, result)


def test_define_treatment():
    features = np.array(
        [[1., 5.0, 1.0],
         [1., 8.0, 2.0],
         [1., 9.0, 2.0],
         [1., 5.0, 1.0],
         [1., 2.0, 2.0]]
    )

    treatment_fn_coef = np.array([4, 2, 1])

    expected = np.array([0, 1, 1, 1, 0])

    np.random.seed(99)
    result = define_treatment(features, treatment_fn_coef, None)

    assert np.allclose(expected, result), "should work with no treatment function provided"

    treatment_fn = lambda x: np.where(x[:, 1] > 5, 1, 0).reshape(-1, 1)

    expected = np.array([[0],
                         [1],
                         [1],
                         [0],
                         [0]])

    result_2 = define_treatment(features, treatment_fn_coef, treatment_fn)

    assert np.allclose(expected, result_2), "should work with a treatment function"


def test_define_outcome():

    features = np.array(
        [[1., 5.0, 1.0],
         [1., 8.0, 2.0],
         [1., 9.0, 2.0],
         [1., 5.0, 1.0],
         [1., 2.0, 2.0]]
    )

    treatment = np.array([[0],
                          [1],
                          [1],
                          [0],
                          [0]])

    y_coef = np.array([[5], [2], [1]])
    t_interaction_coef = np.array([[2], [0], [1]])

    expected = np.array([[1.*5 + 2*5.0 + 1*1.0 + 0*(1.*2 + 5.0*0 + 1.0*1)],
                         [1.*5 + 2*8.0 + 1*2.0 + 1*(1.*2 + 8.0*0 + 2.0*1)],
                         [1.*5 + 2*9.0 + 1*2.0 + 1*(1.*2 + 9.0*0 + 2.0*1)],
                         [1.*5 + 2*5.0 + 1*1.0 + 0*(1.*2 + 5.0*0 + 1.0*1)],
                         [1.*5 + 2*2.0 + 1*2.0 + 0*(1.*2 + 2.0*0 + 2.0*1)]])

    result = define_outcome(features, treatment, y_coef, t_interaction_coef)

    assert np.allclose(result, expected)

def test_random_outcome():
    outcome = np.array(
        [[5.],
         [10.],
         [20.],
         [15.],
         [30.]]
    )

    expected = np.array(
        [[17.99476291],
         [5.10594869],
         [15.77462598],
         [6.41625102],
         [36.92326103]]

    )

    np.random.seed(1)
    result = random_outcome(outcome)

    assert np.allclose(expected, result)


def test_sample():
    mean = np.array([-14.23588427, 205.72217376, 28.32619411])

    cov = np.array(
        [[7.27985875, 1.4376169, 0.84678988],
         [1.4376169, 6.50193927, 0.65921642],
         [0.84678988, 0.65921642, 6.25753432]]
    )

    treatment_function_coef = np.array(
        [[-7.3161625],
         [2.61755461],
         [-8.55795579],
         [-1.87525912]]
    )

    y_coef = np.array(
        [[-4.61970967],
         [-8.16466104],
         [-0.45123303],
         [9.2595278]]
    )

    t_interaction_coef = np.array(
        [[-5.73819696],
         [0.5270311],
         [22.07310587],
         [4.82713428]]
    )

    result = sample(4, 1, mean, cov, treatment_function_coef, y_coef, t_interaction_coef)

    expected = data_frame_test([
        [1.0, -18.98, 203.67, 27.80, 1.0, 4706.89],
        [1.0, -14.46, 212.24, 27.53, 0.0, -863.15],
        [1.0, -18.17, 201.82, 28.04, 1.0, 4941.64],
        [1.0, -15.46, 210.73, 25.27, 0.0, 1014.50]
    ], ["x_0", "x_1", "x_2", "x_3", "T", "Y"])

    pd.testing.assert_frame_equal(expected, result)


