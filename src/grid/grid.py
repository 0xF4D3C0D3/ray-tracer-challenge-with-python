from collections.abc import Iterable
from itertools import product

import numpy as np


class Grid(np.ndarray):
    """
    Grid holds the list of vectors.
    When to_mesh is True, the list will be the cartesian product.
    For the convenience, you can pass the scalar.
    """

    def __new__(cls, xs, ys, zs, ws, to_mesh=True):
        def _l(x):
            return x if isinstance(x, Iterable) else [x]

        if to_mesh:
            obj = np.array(list(product(_l(xs), _l(ys), _l(zs), _l(ws)))).view(cls)
        else:
            obj = np.vstack([xs, ys, zs, ws * len(xs)]).T.view(cls)
        return obj

    def __eq__(self, other):
        return np.allclose(self, other)

    def __matmul__(self, other):
        # __matmul__ of grid return the row-wise dot product
        # this is equivalent to `(self.T * other.T).sum(0)[:,np.newaxis]` but this is three times faster
        return np.einsum("ij,ij->i", self, other)[:, np.newaxis]

    def __repr__(self):
        return "Grid({xs}, {ys}, {zs}, {ws}, False)".format(
            xs=repr(self.x.tolist()),
            ys=repr(self.y.tolist()),
            zs=repr(self.z.tolist()),
            ws=repr([self.w[0]]),
        )

    @property
    def x(self):
        return self[:, 0]

    @property
    def y(self):
        return self[:, 1]

    @property
    def z(self):
        return self[:, 2]

    @property
    def w(self):
        return self[:, 3]


class VectorGrid(Grid):
    def __new__(cls, xs, ys, zs, to_mesh=True):
        return super().__new__(cls, xs, ys, zs, [0], to_mesh=to_mesh)

    def __repr__(self):
        return f"VectorGrid({', '.join(map(repr, self.T[:-1].tolist()))}, False)"

    @property
    def magnitude(self):
        return np.asarray(np.sqrt(np.sum(self ** 2, -1)))[:, np.newaxis]

    def cross(self, other):
        return np.cross(self[:, :-1], other[:, :-1])

    def normalize(self):
        return self / self.magnitude

    def reflect(self, normal):
        return self - normal * 2 * (self @ normal)


class Vector(VectorGrid):
    def __new__(cls, x, y, z):
        return super().__new__(cls, [x], [y], [z], to_mesh=False)

    def __repr__(self):
        return f"Vector({self.x.item()}, {self.y.item()}, {self.z.item()})"


class PointGrid(Grid):
    def __new__(cls, xs, ys, zs, to_mesh=True):
        return super().__new__(cls, xs, ys, zs, [1], to_mesh=to_mesh)

    def __sub__(self, other):
        # Point - Point should be Vector
        res = super().__sub__(other)
        type_self = type(self)
        type_other = type(other)

        if (type_self == PointGrid) or (type_other == PointGrid):
            return res.view(VectorGrid)
        elif type_self == Point:
            return res.view(Vector)
        else:
            return res

    def __repr__(self):
        return f"PointGrid({', '.join(map(repr, self.T[:-1].tolist()))}, False)"


class Point(PointGrid):
    def __new__(cls, x, y, z):
        return super().__new__(cls, [x], [y], [z], to_mesh=False)

    def __repr__(self):
        return f"Point({self.x.item()}, {self.y.item()}, {self.z.item()})"


class ColorGrid(np.ndarray):
    def __new__(cls, reds=0, greens=0, blues=0):
        return np.vstack([reds, greens, blues]).T.view(cls)

    def __repr__(self):
        return f"ColorGrid({', '.join(map(repr, self.T.tolist()))})"

    def __array_wrap__(self, out_arr, context=None):
        res = super().__array_wrap__(out_arr, context)
        if out_arr.ndim > 0:
            if out_arr.shape[-1] == 3:
                if out_arr.shape[0] == 1:
                    return res.view(Color)
                else:
                    return res.view(ColorGrid)
            else:
                return res.view(np.ndarray)
        else:
            return res.item()

    def __eq__(self, other):
        return np.allclose(self, other)

    @property
    def red(self):
        return self.T[0]

    @property
    def green(self):
        return self.T[1]

    @property
    def blue(self):
        return self.T[2]


class Color(ColorGrid):
    def __new__(cls, red=0, green=0, blue=0):
        return super().__new__(cls, [red], [green], [blue])

    def __repr__(self):
        return f"Color({self.red.item()}, {self.green.item()}, {self.blue.item()})"
