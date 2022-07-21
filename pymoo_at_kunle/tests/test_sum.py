import pytest
import numpy as np 

from pymoo.algorithms.moo.nsga2 import calc_crowding_distance

def test_sum():
    np.testing.assert_almost_equal (np.sum([1, 2, 3]),  6 )
def test_product():
    np.testing.assert_almost_equal(np.product([1, 2, 3, 4]),  24.00) 
def test_sum_tuple():
    n = 5
    np.testing.assert_almost_equal(np.sum( (1, 2, 2) ) , n, decimal = 7, err_msg= "Not equal 5" )

bCond  = False
@pytest.mark.skipif(bCond, reason = "cant test when it's true")
def test_crowding_distance_two_duplicates():
    F = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [0.5, 1.5], [0.0, 2.0]])
    cd = calc_crowding_distance(F)
    print(cd)
    np.testing.assert_almost_equal(cd, np.array([np.inf, 0.0, 0.0, 1.0, np.inf]))