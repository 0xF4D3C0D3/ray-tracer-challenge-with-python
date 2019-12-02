import numpy as np

from src.grid import Point, PointGrid, Vector
from src.matrix import Rotation, Scaling, Shearing, Translation


def test_multiplying_by_translation_matrix():
    transform = Translation(5, -3, 2)
    p = Point(-3, 4, 5)
    assert transform @ p == Point(2, 1, 7)


def test_multiplying_by_inverse_of_translation_matrix():
    transform = Translation(5, -3, 2)
    inv = transform.inv
    p = Point(-3, 4, 5)
    assert inv @ p == Point(-8, 7, 3)


def test_translation_dose_not_affect_vectors():
    transform = Translation(5, -3, 2)
    v = Vector(-3, 4, 5)
    assert transform @ v == v


def test_scaling_matrix_applied_to_point():
    transform = Scaling(2, 3, 4)
    p = Point(-4, 6, 8)
    assert transform @ p == Point(-8, 18, 32)


def test_scaling_matrix_applied_to_vector():
    transform = Scaling(2, 3, 4)
    v = Vector(-4, 6, 8)
    assert transform @ v == Vector(-8, 18, 32)


def multiplying_by_inverse_of_scaling_matrix():
    transform = Scaling(2, 3, 4)
    inv = transform.inv
    v = Vector(-4, 6, 8)
    assert inv @ v == Vector(-2, 2, 2)


def test_reflection_is_scaling_by_negative_value():
    transform = Scaling(-1, 1, 1)
    p = Point(2, 3, 4)
    assert transform @ p == Point(-2, 3, 4)


def test_rotating_point_around_x_axis():
    p = Point(0, 1, 0)
    half_quarter = Rotation(np.pi / 4, 0, 0)
    full_quarter = Rotation(np.pi / 2, 0, 0)
    assert half_quarter @ p == Point(0, 2 ** 0.5 / 2, 2 ** 0.5 / 2)
    assert full_quarter @ p == Point(0, 0, 1)


def test_invesre_of_x_rotation_rotates_in_opposite_direction():
    p = Point(0, 1, 0)
    half_quarter = Rotation(np.pi / 4, 0, 0)
    inv = half_quarter.inv
    assert inv @ p == Point(0, 2 ** 0.5 / 2, -(2 ** 0.5) / 2)


def test_rotating_point_around_y_axis():
    p = Point(0, 0, 1)
    half_quarter = Rotation(0, np.pi / 4, 0)
    full_quarter = Rotation(0, np.pi / 2, 0)
    assert half_quarter @ p == Point(2 ** 0.5 / 2, 0, 2 ** 0.5 / 2)
    assert full_quarter @ p == Point(1, 0, 0)


def test_rotating_point_around_z_axis():
    p = Point(0, 1, 0)
    half_quarter = Rotation(0, 0, np.pi / 4)
    full_quarter = Rotation(0, 0, np.pi / 2)
    assert half_quarter @ p == Point(-(2 ** 0.5) / 2, 2 ** 0.5 / 2, 0)
    assert full_quarter @ p == Point(-1, 0, 0)


def test_shearing_transformation_moves_x_in_proportion_to_y():
    transform = Shearing(1, 0, 0, 0, 0, 0)
    p = Point(2, 3, 4)
    assert transform @ p == Point(5, 3, 4)


def test_shearing_transformation_moves_x_in_proportion_to_z():
    transform = Shearing(0, 1, 0, 0, 0, 0)
    p = Point(2, 3, 4)
    assert transform @ p == Point(6, 3, 4)


def test_shearing_transformation_moves_y_in_proportion_to_x():
    transform = Shearing(0, 0, 1, 0, 0, 0)
    p = Point(2, 3, 4)
    assert transform @ p == Point(2, 5, 4)


def test_shearing_transformation_moves_y_in_proportion_to_z():
    transform = Shearing(0, 0, 0, 1, 0, 0)
    p = Point(2, 3, 4)
    assert transform @ p == Point(2, 7, 4)


def test_shearing_transformation_moves_z_in_proportion_to_x():
    transform = Shearing(0, 0, 0, 0, 1, 0)
    p = Point(2, 3, 4)
    assert transform @ p == Point(2, 3, 6)


def test_shearing_transformation_moves_z_in_proportion_to_y():
    transform = Shearing(0, 0, 0, 0, 0, 1)
    p = Point(2, 3, 4)
    assert transform @ p == Point(2, 3, 7)


def test_individual_transformations_are_applied_in_sequence():
    p = Point(1, 0, 1)
    A = Rotation(np.pi / 2, 0, 0)
    B = Scaling(5, 5, 5)
    C = Translation(10, 5, 7)

    p2 = A @ p
    assert p2 == Point(1, -1, 0)

    p3 = B @ p2
    assert p3 == Point(5, -5, 0)

    p4 = C @ p3
    assert p4 == Point(15, 0, 7)


def test_chained_transformations_must_be_applied_in_reverse_order():
    p = Point(1, 0, 1)
    A = Rotation(np.pi / 2, 0, 0)
    B = Scaling(5, 5, 5)
    C = Translation(10, 5, 7)

    T = C @ B @ A
    assert T @ p == Point(15, 0, 7)

    pg = PointGrid([1, 10], [2, 3], 4)
    assert T @ pg == PointGrid(
        [15, 15, 60, 60], [-15, -15, -15, -15], [17, 22, 17, 22], False
    )
