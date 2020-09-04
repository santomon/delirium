import pytest
import utility
import numpy as np


@pytest.fixture(scope="package")
def sample_data():
    sample_data = [{"xd": np.array([1, 2, 3, 4, 5]), "lmao": np.array([2, 3, 4, 5, 6])},
                   {"xd": np.array([3, 4, 5, 6, 7]), "lmao": np.array([1, 2, 3, 4, 5])}]
    return sample_data


@pytest.fixture(scope="module")
def sample_solution():
    return {"xd": np.array([2, 3, 4, 5, 6]), "lmao": np.array([1.5, 2.5, 3.5, 4.5, 5.5])}


def test_aggregate_over_list_of_dict_with_npmean(sample_data, sample_solution):
    solution = utility.aggregate_over_list_of_dict(sample_data, lambda x: np.mean(x, axis=0))
    for key in solution.keys():
        assert (solution[key] == sample_solution[key]).all(), solution


def test_inspect_with_sample_data(sample_data, capsys):
    capsys.readouterr()
    utility.inspect(sample_data)
    captured = capsys.readouterr()
    assert captured.out == "xd (5,)\nlmao (5,)\n\nxd (5,)\nlmao (5,)\n\n"


def test_eliminate_by_substr__xd_lmao_lul_lmao__lma():
    list_, indices = utility.eliminate_by_substr(["xd", "lmao", "lul", "lmao"], "lma")
    assert list_ == ["xd", "lul"]
    assert indices == [1, 3]


def test_eliminate_by_substr_with_int():
    with pytest.raises(NotImplementedError) as excinfo:
        result = utility.eliminate_by_substr(3, "xd")
    assert "only be used on a list that contains only strings" in str(excinfo.value)


def test_eliminate_by_indices_with_np_array3456_indices12():
    result = utility.eliminate_by_indices(np.array([3, 4, 5, 6]), [1, 2])
    assert (result == np.array([3, 6])).all()


def test_eliminate_by_indices_with_list3456_indices12():
    result = utility.eliminate_by_indices([3, 4, 5, 6], [1, 2])
    assert result == [3, 6]


def test_eliminate_by_indices_with_dict():
    with pytest.raises(NotImplementedError) as excinfo:
        result = utility.eliminate_by_indices(dict(), [3, 4])
    assert "Currently not supported for other types than np.ndarrays and lists" in str(excinfo.value)
