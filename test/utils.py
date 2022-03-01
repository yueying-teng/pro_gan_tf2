import numpy as np


def assert_almost_equal(x, y, error_margin=3.0):
    assert np.abs(x - y) <= error_margin
