import numpy as np

from src.grid import Grid
from src.matrix import Matrix


def test_constructing_and_inspecting_4x4_matrix():
    M = Matrix.from_string(
        """
        | 1 | 2 | 3 | 4 |
        | 5.5 | 6.5 | 7.5 | 8.5 |
        | 9 | 10 | 11 | 12 |
        | 13.5 | 14.5 | 15.5 | 16.5 |
        """
    )

    assert M[0, 0] == 1
    assert M[0, 3] == 4
    assert M[1, 0] == 5.5
    assert M[1, 2] == 7.5
    assert M[2, 2] == 11
    assert M[3, 0] == 13.5
    assert M[3, 2] == 15.5


def test_2x2_matrix_ought_to_be_representable():
    M = Matrix.from_string(
        """
        | -3 | 5 |
        | 1 | -2 |
        """
    )

    assert M[0, 0] == -3
    assert M[0, 1] == 5
    assert M[1, 0] == 1
    assert M[1, 1] == -2


def test_3x3_matrix_ought_to_be_representable():
    M = Matrix.from_string(
        """
        | -3 | 5 | 0 |
        | 1 | -2 | -7 |
        | 0 | 1 | 1 |
        """
    )

    assert M[0, 0] == -3
    assert M[1, 1] == -2
    assert M[2, 2] == 1


def test_matrix_equality_with_identical_matrices():
    A = Matrix.from_string(
        """
        | 1 | 2 | 3 | 4 |
        | 5 | 6 | 7 | 8 |
        | 9 | 8 | 7 | 6 |
        | 5 | 4 | 3 | 2 |
        """
    )

    B = Matrix.from_string(
        """
        | 1 | 2 | 3 | 4 |
        | 5 | 6 | 7 | 8 |
        | 9 | 8 | 7 | 6 |
        | 5 | 4 | 3 | 2 |
        """
    )

    assert np.array_equal(A, B)


def test_matrix_equality_with_different_matrices():
    A = Matrix.from_string(
        """
        | 1 | 2 | 3 | 4 |
        | 5 | 6 | 7 | 8 |
        | 9 | 8 | 7 | 6 |
        | 5 | 4 | 3 | 2 |
        """
    )

    B = Matrix.from_string(
        """
        | 2 | 3 | 4 | 5 |
        | 6 | 7 | 8 | 9 |
        | 8 | 7 | 6 | 5 |
        | 4 | 3 | 2 | 1 |
        """
    )

    assert not np.array_equal(A, B)


def test_multiplying_two_matrices():
    A = Matrix.from_string(
        """
        | 1 | 2 | 3 | 4 |
        | 5 | 6 | 7 | 8 |
        | 9 | 8 | 7 | 6 |
        | 5 | 4 | 3 | 2 |
        """
    )

    B = Matrix.from_string(
        """
        | -2 | 1 | 2 | 3 |
        | 3 | 2 | 1 | -1 |
        | 4 | 3 | 6 | 5 |
        | 1 | 2 | 7 | 8 |
        """
    )

    expected = Matrix.from_string(
        """
        | 20| 22 | 50 | 48 |
        | 44| 54 | 114 | 108 |
        | 40| 58 | 110 | 102 |
        | 16| 26 | 46 | 42 |
        """
    )

    assert np.array_equal(A @ B, expected)


def test_matrix_multiplied_by_tuple():
    A = Matrix.from_string(
        """
        | 1 | 2 | 3 | 4 |
        | 2 | 4 | 4 | 2 |
        | 8 | 6 | 4 | 1 |
        | 0 | 0 | 0 | 1 |
        """
    )

    b = Grid(1, 2, 3, 1)

    assert A @ b == Grid(18, 24, 33, 1)


def test_multiplying_matrix_by_identity_matrix():
    A = Matrix.from_string(
        """
        | 0 | 1 | 2 | 4 |
        | 1 | 2 | 4 | 8 |
        | 2 | 4 | 8 | 16 |
        | 4 | 8 | 16 | 32 |
        """
    )

    assert np.array_equal(A @ np.eye(4), A)


def test_multiplying_identity_matrix_by_tuple():
    a = Grid(1, 2, 3, 4)
    m = Matrix(np.eye(4))
    assert m @ a == a


def test_transposing_matrix():
    A = Matrix.from_string(
        """
        | 0 | 9 | 3 | 0 |
        | 9 | 8 | 0 | 8 |
        | 1 | 8 | 5 | 3 |
        | 0 | 0 | 5 | 8 |
        """
    )

    expected = Matrix.from_string(
        """
        | 0 | 9 | 1 | 0 |
        | 9 | 8 | 8 | 0 |
        | 3 | 0 | 5 | 5 |
        | 0 | 8 | 3 | 8 |
        """
    )

    assert np.array_equal(A.T, expected)


def test_transposing_identity_matrix():
    A = np.eye(4).T
    assert np.array_equal(A, np.eye(4))


def test_calculating_inverse_of_another_matrix():
    A = Matrix.from_string(
        """
        | 8 | -5 | 9 | 2 |
        | 7 | 5 | 6 | 1 |
        | -6 | 0 | 9 | 6 |
        | -3 | 0 | -9 | -4 |
        """
    )

    expected = Matrix.from_string(
        """
        | -0.15385 | -0.15385 | -0.28205 | -0.53846 |
        | -0.07692 | 0.12308 | 0.02564 | 0.03077 |
        | 0.35897 | 0.35897 | 0.43590 | 0.92308 |
        | -0.69231 | -0.69231 | -0.76923 | -1.92308 |
        """
    )

    assert np.allclose(A.inv, expected, 1e-03, 1e-03)


def test_calculating_inverse_of_third_matrix():
    A = Matrix.from_string(
        """
        | 9 | 3 | 0 | 9 |
        | -5 | -2 | -6 | -3 |
        | -4 | 9 | 6 | 4 |
        | -7 | 6 | 6 | 2 |
        """
    )

    expected = Matrix.from_string(
        """
        | -0.04074 | -0.07778 | 0.14444 | -0.22222 |
        | -0.07778 | 0.03333 | 0.36667 | -0.33333 |
        | -0.02901 | -0.14630 | -0.10926 | 0.12963 |
        | 0.17778 | 0.06667 | -0.26667 | 0.33333 |
        """
    )

    assert np.allclose(A.inv, expected, 1e-03, 1e-03)


def test_multiplying_product_by_its_inverse():
    A = Matrix.from_string(
        """
        | 3 | -9 | 7 | 3 |
        | 3 | -8 | 2 | -9 |
        | -4 | 4 | 4 | 1 |
        | -6 | 5 | -1 | 1 |
        """
    )

    B = Matrix.from_string(
        """
        | 8 | 2 | 2 | 2 |
        | 3 | -1 | 7 | 0 |
        | 7 | 0 | 5 | 4 |
        | 6 | -2 | 0 | 5 |
        """
    )

    C = A @ B

    assert C @ B.inv == A
