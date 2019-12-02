from src.grid import Point, VectorGrid
from src.light import Ray
from src.matrix import Scaling, Translation


def test_creating_and_querying_ray():
    origin = Point(0, 0, -5)
    direction = VectorGrid([-1, 0, 1], [-1, 0, 1], 3)
    r = Ray(origin, direction)
    assert r.origin == origin
    assert r.direction == direction


def test_computing_point_from_distance():
    origin = Point(0, 0, -1)
    direction = VectorGrid(5, [10, 20], 1)
    r = Ray(origin, direction)

    assert r.after(0) == [[0, 0, -1, 1], [0, 0, -1, 1]]
    assert r.after(1) == [[5, 10, 0, 1], [5, 20, 0, 1]]
    assert r.after(-1) == [[-5, -10, -2, 1], [-5, -20, -2, 1]]
    assert r.after(2.5) == [[12.5, 25, 1.5, 1], [12.5, 50, 1.5, 1]]


def test_translating_ray():
    origin = Point(0, 0, -1)
    direction = VectorGrid(5, [10, 20], 1)
    r = Ray(origin, direction)
    m = Translation(3, 4, 5)
    r2 = r.transform(m)
    assert r2.origin == [[3, 4, 4, 1]]
    assert r2.direction == [[5, 10, 1, 0], [5, 20, 1, 0]]


def test_scaling_ray():
    origin = Point(0, 0, -1)
    direction = VectorGrid(5, [10, 20], 1)
    r = Ray(origin, direction)
    m = Scaling(2, 3, 4)
    r2 = r.transform(m)
    assert r2.origin == [[0, 0, -4, 1]]
    assert r2.direction == [[10, 30, 4, 0], [10, 60, 4, 0]]
