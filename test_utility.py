import pytest
import utility
import numpy as np


@pytest.fixture(scope="module")
def brain_data_path():
    return "C:/xd/bt/data/ROIs"


@pytest.fixture(scope="module")
def sample_data():
    sample_data = [{"xd": np.array([1, 2, 3, 4]), "lmao": np.array([2, 3, 4, 5, 6])},
                   {"xd": np.array([3, 4, 5, 6]), "lmao": np.array([1, 2, 3, 4, 5])}]
    return sample_data


@pytest.fixture(scope="module")
def sample_solution():
    return {"xd": np.array([2, 3, 4, 5]), "lmao": np.array([1.5, 2.5, 3.5, 4.5, 5.5])}


def test_load_brain_data_run_through_on_tr_int(brain_data_path):
    data_ = utility.load_brain_data(brain_data_path, 1, 3)
    assert True


def test_load_brain_data_run_through_on_tr_list(brain_data_path):
    data_ = utility.load_brain_data(brain_data_path, 1, [3, 4])
    assert True


def test_load_brain_data_all_rois_are_keys(brain_data_path):
    data_ = utility.load_brain_data(brain_data_path, 1, 3)
    assert set(utility.ROI_LABELS) == set(data_.keys())


def test_aggregate_over_list_of_dict_with_npmean(sample_data, sample_solution):
    solution = utility.aggregate_over_list_of_dict(sample_data, lambda x: np.mean(x, axis=0))
    for key in solution.keys():
        assert (solution[key] == sample_solution[key]).all(), solution
