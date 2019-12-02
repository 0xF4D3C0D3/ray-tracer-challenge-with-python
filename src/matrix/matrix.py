import numpy as np


class Matrix(np.ndarray):
    def __new__(cls, ndarray=None):
        if ndarray is None:
            ndarray = np.eye(4)
        obj = np.asanyarray(ndarray).view(cls)
        return obj

    def __repr__(self):
        return f"Matrix({repr(self.tolist())})"

    def __eq__(self, other):
        return np.allclose(self, other)

    def __matmul__(self, other):
        if isinstance(other, Matrix):
            return super().__matmul__(other).view(other.__class__)
        else:
            return super().__matmul__(other.T).T.view(other.__class__)

    @property
    def inv(self):
        return np.linalg.inv(self)

    @staticmethod
    def from_string(string):
        ndarray = np.array(
            [
                [float(xx) for xx in x.split("|")[1:-1]]
                for x in string.strip().splitlines()
            ]
        )
        return Matrix(ndarray)


class Translation(Matrix):
    def __new__(cls, x, y, z):
        obj = super().__new__(cls)
        obj[:3, 3] = [x, y, z]
        return obj


class Scaling(Matrix):
    def __new__(cls, x, y, z):
        obj = super().__new__(cls, np.diag([x, y, z, 1]))
        return obj


class Rotation(Matrix):
    def __new__(cls, x, y, z):
        matrix = np.eye(4)

        if x:
            matrix_x = np.eye(4)
            matrix_x[1:3, 1:3] = [[np.cos(x), -np.sin(x)], [np.sin(x), np.cos(x)]]
            matrix = matrix_x @ matrix

        if y:
            matrix_y = np.eye(4)
            matrix_y[::2, ::2] = [[np.cos(y), np.sin(y)], [-np.sin(y), np.cos(y)]]
            matrix = matrix_y @ matrix

        if z:
            matrix_z = np.eye(4)
            matrix_z[:2, :2] = [[np.cos(z), -np.sin(z)], [np.sin(z), np.cos(z)]]
            matrix = matrix_z @ matrix

        obj = super().__new__(cls, matrix)
        return obj


class Shearing(Matrix):
    def __new__(cls, xy, xz, yx, yz, zx, zy):
        matrix = np.eye(4)
        matrix[:3, :3] = [[1, xy, xz], [yx, 1, yz], [zx, zy, 1]]
        obj = super().__new__(cls, matrix)
        return obj
