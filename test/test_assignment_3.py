# src/test/test_assignment_3.py

import unittest
import numpy as np
from src.main.assignment_3 import gaussian_elimination, lu_factorization, is_diagonally_dominant, is_positive_definite

class TestAssignment3(unittest.TestCase):

    def test_gaussian_elimination(self):
        A = np.array([
            [2, -1, 1],
            [1, 3, 1],
            [-1, 5, 4]
        ])
        b = np.array([6, 0, -3])
        expected_solution = np.array([2, -1, 1])

        result = gaussian_elimination(A, b)

        self.assertTrue(np.allclose(result, expected_solution, atol=1e-4))

    def test_lu_factorization(self):
        A = np.array([
            [1, 1, 0, 3],
            [2, 1, -1, 1],
            [3, -1, -1, 2],
            [-1, 2, 3, -1]
        ])
        L, U, expected_det = lu_factorization(A)

        expected_L = np.array([
            [ 1.,  0.,  0.,  0.],
            [ 2.,  1.,  0.,  0.],
            [ 3.,  1.,  1.,  0.],
            [-1.,  1.,  1.,  1.]
        ])
        expected_U = np.array([
            [ 1.,  1.,  0.,  3.],
            [ 0., -1., -1., -5.],
            [ 0.,  0.,  1.,  6.],
            [ 0.,  0.,  0., -12.]
        ])

        self.assertTrue(np.allclose(L, expected_L, atol=1e-4))
        self.assertTrue(np.allclose(U, expected_U, atol=1e-4))
        self.assertAlmostEqual(expected_det, -20.0, places=4)

    def test_is_diagonally_dominant(self):
        A = np.array([
            [9, 0, 5, 2, 1],
            [3, 9, 1, 2, 1],
            [0, 1, 7, 2, 3],
            [4, 2, 3, 12, 2],
            [3, 2, 4, 0, 8]
        ])
        self.assertTrue(is_diagonally_dominant(A))

    def test_is_positive_definite(self):
        A = np.array([
            [2, 2, 1],
            [2, 3, 0],
            [1, 0, 2]
        ])
        self.assertTrue(is_positive_definite(A))

if __name__ == '__main__':
    unittest.main()
