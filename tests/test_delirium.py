import pytest
import delirium
import delirium_config
import utility
import numpy as np
import os
import typing as t


@pytest.fixture(scope="package")
def sample_data():
    sample_data = [{"xd": np.array([1, 2, 3, 4, 5]), "lmao": np.array([2, 3, 4, 5, 6])},
                   {"xd": np.array([3, 4, 5, 6, 7]), "lmao": np.array([1, 2, 3, 4, 5])}]
    return sample_data


@pytest.fixture(scope="module")
def sample_stim_lists():
    return [["xd", "lmao", "lmaooo", "hehe", "rofl"],
            ["abc", "destiny", "maokai", "insanity", "endeavour"]]


@pytest.fixture(scope="module")
def sample_substr():
    return "mao"


@pytest.fixture(scope="module")
def brain_data_path():
    return delirium_config.BOLD5K_ROI_DATA_PATH


@pytest.fixture(scope="module")
def target_shapes_after_load():
    target_shapes = {
        "LHPPA": (5254, 131),
        "RHLOC": (5254, 190),
        "LHLOC": (5254, 152),
        "RHEarlyVis": (5254, 285),
        "RHRSC": (5254, 143),
        "RHOPA": (5254, 187),
        "RHPPA": (5254, 200),
        "LHEarlyVis": (5254, 210),
        "LHRSC": (5254, 86),
        "LHOPA": (5254, 101)
    }
    return target_shapes


@pytest.fixture(scope="module")
def nn_data_path():
    return delirium_config.NN_DATA_PATH


@pytest.fixture(scope="module")
def nn_data_suffix():
    return "_bb_f16"


@pytest.fixture(scope="module")
def nn_data_prefix():
    return ""


@pytest.fixture(scope="module")
def nn_data_file_ending():
    return "npy"


###########################################################################################################
#################################### test functions #######################################################
###########################################################################################################


@pytest.fixture(scope="module")
def brain_data_subj123_TR3():
    return [delirium.load_brain_data(subject=i, ) for i in range(1, 4)]


def test_load_brain_data_run_through_on_tr_int(brain_data_path, target_shapes_after_load):
    data_ = delirium.load_brain_data(brain_data_path, 1, 3)
    utility.inspect(data_)

    for roi in data_.keys():
        assert data_[roi].shape == target_shapes_after_load[roi]


def test_load_brain_data_all_rois_are_keys(brain_data_path):
    data_ = delirium.load_brain_data(brain_data_path, 1, 3)
    assert set(delirium.ROI_LABELS) == set(data_.keys())


def test_load_stim_lists_with_dummy2x3(tmp_path):
    content_stim_list_1 = "xd\nlmao\nhaha"
    content_stim_list_2 = "rofl\nhehe\nkiller"

    d = tmp_path / "stim_lists"
    d.mkdir()
    p1 = d / "CSI01_stim_lists.txt"
    p1.write_text(content_stim_list_1)

    p2 = d / "CSI02_stim_lists.txt"
    p2.write_text(content_stim_list_2)

    stim_lists = delirium.load_stim_lists(str(tmp_path), [1, 2])
    assert stim_lists == [["xd", "lmao", "haha"], ["rofl", "hehe", "killer"]]


def test_eliminate_data_by_substr_rep_assert_shape0_equals_4916(brain_data_subj123_TR3):
    data_ = brain_data_subj123_TR3
    stim_lists = delirium.load_stim_lists(subjects=[1, 2, 3])

    new_data, new_stim_lists = delirium.eliminate_from_data_by_substr(data_, stim_lists, "rep_")

    utility.inspect(new_data)

    for sl in new_stim_lists:
        assert len(sl) == 4916

    for i in range(3):
        for roi in new_data[i].keys():
            assert new_data[i][roi].shape[0] == 4916, new_data[i][roi].shape[0]


@pytest.mark.dependency()
def test_eliminate_data_by_substr_3entities_assert_shape0_equals_4913(brain_data_subj123_TR3):
    data_ = brain_data_subj123_TR3
    stim_lists = delirium.load_stim_lists(subjects=[1, 2, 3])

    new_data, new_stim_lists = delirium.eliminate_from_data_by_substr(data_, stim_lists, "rep_")

    for image_name in delirium.TO_ELIMINATE[:3]:
        new_data, new_stim_lists = delirium.eliminate_from_data_by_substr(new_data, new_stim_lists, image_name)

    for sl in new_stim_lists:
        assert len(sl) == 4913
    for i in range(3):
        for roi in new_data[i].keys():
            assert new_data[i][roi].shape[0] == 4913, new_data[i][roi].shape[0]

    pytest.Class.stim_lists = new_stim_lists


def test_load_brain_data_run_through_on_tr_list(brain_data_path, target_shapes_after_load):
    data_ = delirium.load_brain_data(brain_data_path, 1, [3, 4])

    for roi in data_.keys():
        assert data_[roi].shape == target_shapes_after_load[roi]


def test_eliminate_data_by_substr_with_sample_data(sample_data, sample_stim_lists, sample_substr):
    new_data, new_stim_lists = delirium.eliminate_from_data_by_substr(sample_data, sample_stim_lists, sample_substr)
    assert new_stim_lists == [["xd", "hehe", "rofl"], ["abc", "destiny", "insanity", "endeavour"]]
    solution = [{"xd": np.array([1, 4, 5]), "lmao": np.array([2, 5, 6])},
                {"xd": np.array([3, 4, 6, 7]), "lmao": np.array([1, 2, 4, 5])}]

    assert (solution[0]["xd"] == new_data[0]["xd"]).all()
    assert (solution[0]["lmao"] == new_data[0]["lmao"]).all()
    assert (solution[1]["xd"] == new_data[1]["xd"]).all()
    assert (solution[1]["lmao"] == new_data[1]["lmao"]).all()


@pytest.mark.slow
@pytest.mark.dependency(depends=["test_eliminate_data_by_substr_3entities_assert_shape0_equals_4913"])
def test_load_nn_data_with_legit_data_assert_shape(nn_data_path, nn_data_prefix, nn_data_suffix, nn_data_file_ending):
    if os.path.isdir(nn_data_path):
        data_ = delirium.load_nn_data(pytest.Class.stim_lists[0], nn_data_path, nn_data_prefix, nn_data_suffix,
                              nn_data_file_ending)

        assert len(data_.shape) == 2
        assert data_.shape[0] == len(pytest.Class.stim_lists[0])

    else:
        print("test_load_nn_data_with_legit_data_assert_shape() skipped because nn_data_path does not exist")

@pytest.mark.slow
@pytest.mark.dependency(depends=["test_eliminate_data_by_substr_3entities_assert_shape0_equals_4913"])
def test_load_nn_data_with_file_ending_xd(nn_data_path, nn_data_prefix, nn_data_suffix):
    if os.path.isdir(nn_data_path):
        with pytest.raises(NotImplementedError) as excinfo:
            data_ = delirium.load_nn_data(pytest.Class.stim_lists[0], nn_data_path, nn_data_prefix, nn_data_suffix, "xd")
        assert "operation is currently only supported for npy-files" in str(excinfo.value)
    else:
        print("test_load_nn_data_with_file_ending_xd() has been skipped because nn_data_path does not exist")