import numpy as np

from src.grid import ColorGrid, Grid, Point, PointGrid, Vector, VectorGrid


def test_w_should_be_0_when_it_is_vector_grid():
    v = VectorGrid([11, 22], [33, 44], [55, 66])
    expected = [
        [11, 33, 55, 0],
        [11, 33, 66, 0],
        [11, 44, 55, 0],
        [11, 44, 66, 0],
        [22, 33, 55, 0],
        [22, 33, 66, 0],
        [22, 44, 55, 0],
        [22, 44, 66, 0],
    ]
    assert v == expected


def test_w_should_be_1_when_it_is_point_grid():
    p = PointGrid([11, 22], [33, 44], [55, 66])
    expected = [
        [11, 33, 55, 1],
        [11, 33, 66, 1],
        [11, 44, 55, 1],
        [11, 44, 66, 1],
        [22, 33, 55, 1],
        [22, 33, 66, 1],
        [22, 44, 55, 1],
        [22, 44, 66, 1],
    ]
    assert p == expected


def test_subtracting_two_points():
    p1 = PointGrid([0, 1], 0, 0)
    p2 = Point(10, 20, 30)
    expected = [[10, 20, 30, 0], [9, 20, 30, 0]]
    assert p2 - p1 == expected


def test_subtracting_vector_from_point():
    p = Point(3, 2, 1)
    v = Vector(5, 6, 7)
    assert p - v == Point(-2, -4, -6)


def test_subtracting_vector_from_zero_vector():
    zero = Vector(0, 0, 0)
    v = Vector(1, -2, 3)
    assert zero - v == Vector(-1, 2, -3)


def test_negating_grid():
    g = Grid(1, 2, [10, 20], -1)
    expected = [[-1, -2, -10, 1], [-1, -2, -20, 1]]
    assert -g == expected


def test_multiplying_grid_by_scalar():
    g = Grid(1, 2, [10, 20], [100, 200])
    expected = [
        [2.5, 5, 25, 250],
        [2.5, 5, 25, 500],
        [2.5, 5, 50, 250],
        [2.5, 5, 50, 500],
    ]
    assert g * 2.5 == expected


def test_multiplying_grid_by_fraction():
    g = Grid(1, 2, [10, 20], [100, 200])
    expected = [[0.5, 1, 5, 50], [0.5, 1, 5, 100], [0.5, 1, 10, 50], [0.5, 1, 10, 100]]
    assert g * 0.5 == expected


def test_dividing_by_scalar():
    g = Grid(1, 2, [10, 20], [100, 200])
    expected = [[0.5, 1, 5, 50], [0.5, 1, 5, 100], [0.5, 1, 10, 50], [0.5, 1, 10, 100]]
    assert g / 2 == expected


def test_computing_magnitude_of_vector():
    v = VectorGrid([1, 0, 0, 1, -1], [0, 1, 0, 2, -2], [0, 0, 1, 3, -3], False)
    expected = [1, 1, 1, 14 ** 0.5, 14 ** 0.5]
    assert np.allclose(v.magnitude.squeeze(), expected)


def test_normalizing_vector():
    v = VectorGrid([4, 1], [0, 2], [0, 3], False)
    expected = np.array(
        [[1, 0, 0, 0], [1 / 14 ** 0.5, 2 / 14 ** 0.5, 3 / 14 ** 0.5, 0]]
    )
    assert v.normalize() == expected


def test_magnitude_of_normalized_vector():
    v = VectorGrid(1, 2, [10, 20])
    norm = v.normalize()
    assert np.allclose(norm.magnitude, [[1], [1]])


def test_dot_product_of_two_grids():
    v1 = Vector(1, 2, 3)
    v2 = Vector(2, 3, 4)
    assert v1 @ v2 == 20
    assert v2 @ v1 == 20


def test_cross_product_of_two_vectors():
    v1 = Vector(1, 2, 3)
    v2 = Vector(2, 3, 4)
    assert np.allclose(v1.cross(v2), [-1, 2, -1])


def test_adding_colors():
    c1 = ColorGrid(0.9, 0.6, 0.75)
    c2 = ColorGrid(0.7, 0.1, 0.25)
    assert c1 + c2 == ColorGrid(1.6, 0.7, 1.0)


def test_subtracting_colors():
    c1 = ColorGrid(0.9, 0.6, 0.75)
    c2 = ColorGrid(0.7, 0.1, 0.25)
    assert c1 - c2 == ColorGrid(0.2, 0.5, 0.5)


def test_multiplying_color_by_scalar():
    c = ColorGrid(0.2, 0.3, 0.4)
    assert c * 2 == ColorGrid(0.4, 0.6, 0.8)


def test_multiplying_colors():
    c1 = ColorGrid(1, 0.2, 0.4)
    c2 = ColorGrid(0.9, 1, 0.1)
    assert c1 * c2 == ColorGrid(0.9, 0.2, 0.04)


def test_reflecting_vector():
    v = VectorGrid([1, 0], [-1, -1], [0, 0], False)
    n = VectorGrid([0, 2 ** 0.5 / 2], [-1, 2 ** 0.5 / 2], [0, 0], False)
    r = v.reflect(n)
    expected = [[1, 1, 0, 0], [1, 0, 0, 0]]
    assert r == expected
