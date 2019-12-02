import numpy as np

from src.grid import Point, VectorGrid
from src.intersection import Intersection
from src.material import Material
from src.matrix import Matrix


class Sphere:
    def __init__(self, transform=None, material=None):
        self.transform = Matrix() if transform is None else transform
        self.material = Material() if material is None else material

    def __repr__(self):
        return (
            f"Sphere(transform={repr(self.transform)}, material={repr(self.material)})"
        )

    def set_transform(self, transform):
        return Sphere(transform, self.material)

    def set_material(self, material):
        return Sphere(self.transform, material)

    def normal_at(self, point):
        obj_point = self.transform.inv @ point
        obj_normal = obj_point - Point(0, 0, 0)

        world_normal = VectorGrid(*(self.transform.inv.T @ obj_normal).T[:-1], False)
        return world_normal.normalize()

    def intersect(self, ray):
        ray = ray.transform(self.transform.inv)
        sphere_to_ray = ray.origin - Point(0, 0, 0)
        a = ray.direction @ ray.direction

        b = 2 * (ray.direction @ sphere_to_ray)
        c = sphere_to_ray @ sphere_to_ray - 1
        discriminant = b ** 2 - 4 * a * c

        mask = discriminant >= 0
        masked_a = a[mask]
        masked_b = b[mask]
        masked_discriminant = discriminant[mask]

        t1 = (-masked_b - masked_discriminant ** 0.5) / (2 * masked_a)
        t2 = (-masked_b + masked_discriminant ** 0.5) / (2 * masked_a)
        res = Intersection(np.vstack([t1, t2]), mask, self)

        return res
