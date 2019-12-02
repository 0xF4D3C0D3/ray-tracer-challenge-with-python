import numpy as np

from src.grid import Point, Vector
from src.light import Ray
from src.material import Material
from src.matrix import Rotation, Scaling, Translation
from src.shape.sphere import Sphere


def test_ray_intersects_sphere_at_two_points():
    r = Ray(Point(0, 0, -5), Vector(0, 0, 1))
    s = Sphere()
    xs = s.intersect(r)
    assert xs.count == 1
    assert xs == [4, 6]


def test_ray_intersects_sphere_at_tangent():
    r = Ray(Point(0, 1, -5), Vector(0, 0, 1))
    s = Sphere()
    xs = s.intersect(r)
    assert xs.count == 1
    assert xs == [5, 5]


def test_ray_misses_sphere():
    r = Ray(Point(0, 2, -5), Vector(0, 0, 1))
    s = Sphere()
    xs = s.intersect(r)
    assert xs.count == 0


def test_ray_originates_inside_sphere():
    r = Ray(Point(0, 0, 0), Vector(0, 0, 1))
    s = Sphere()
    xs = s.intersect(r)
    assert xs.count == 1
    assert xs == [-1, 1]


def test_sphere_is_behind_ray():
    r = Ray(Point(0, 0, 5), Vector(0, 0, 1))
    s = Sphere()
    xs = s.intersect(r)
    assert xs.count == 1
    assert xs == [-6, -4]


def test_intersect_sets_the_object_on_intersection():
    r = Ray(Point(0, 0, -5), Vector(0, 0, 1))
    s = Sphere()
    xs = s.intersect(r)
    assert xs.count == 1
    assert xs.obj is s


def test_sphere_default_transformation():
    s = Sphere()
    assert np.allclose(s.transform, np.eye(4))


def test_changing_sphere_transformation():
    s = Sphere()
    t = Translation(2, 3, 4)
    s = s.set_transform(t)
    assert np.allclose(s.transform, t)


def test_intersecting_scaled_sphere_with_ray():
    r = Ray(Point(0, 0, -5), Vector(0, 0, 1))
    s = Sphere()
    s = s.set_transform(Scaling(2, 2, 2))
    xs = s.intersect(r)
    assert xs.count == 1
    assert xs == [3, 7]


def test_intersecting_translated_sphere_with_ray():
    r = Ray(Point(0, 0, -5), Vector(0, 0, 1))
    s = Sphere()
    s = s.set_transform(Translation(5, 0, 0))
    xs = s.intersect(r)
    assert xs.count == 0


def test_normal_on_sphere_at_point_xaxis():
    s = Sphere()
    n = s.normal_at(Point(1, 0, 0))
    assert n == Vector(1, 0, 0)


def test_normal_on_sphere_at_point_yaxis():
    s = Sphere()
    n = s.normal_at(Point(0, 1, 0))
    assert n == Vector(0, 1, 0)


def test_normal_on_sphere_at_point_zaxis():
    s = Sphere()
    n = s.normal_at(Point(0, 0, 1))
    assert n == Vector(0, 0, 1)


def test_normal_on_sphere_at_nonaxial_point():
    s = Sphere()
    a = 3 ** 0.5 / 3
    n = s.normal_at(Point(a, a, a))
    assert n == Vector(a, a, a)


def test_normal_is_normalized_vector():
    s = Sphere()
    a = 3 ** 0.5 / 3

    n = s.normal_at(Point(a, a, a))
    assert n == n.normalize()


def test_computing_normal_on_translated_sphere():
    s = Sphere()
    s = s.set_transform(Translation(0, 1, 0))
    n = s.normal_at(Point(0, 1.70711, -0.70711))
    assert n == Vector(0, 0.70711, -0.70711)


def test_computing_normal_on_transformed_sphere():
    s = Sphere()
    m = Scaling(1, 0.5, 1) @ Rotation(0, 0, np.pi / 5)
    s = s.set_transform(m)
    n = s.normal_at(Point(0, 2 ** 0.5 / 2, -(2 ** 0.5) / 2))
    assert np.allclose(n, Vector(0, 0.97014, -0.24254), 1e-03, 1e-03)


def test_sphere_has_default_material():
    s = Sphere()
    m = s.material
    assert m == Material()


def test_sphere_may_be_assigned_material():
    s = Sphere()
    m = Material(ambient=1)
    s = s.set_material(m)
    assert s.material == m
