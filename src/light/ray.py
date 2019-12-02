from numbers import Number

import numpy as np


class Ray:
    def __init__(self, origin, direction):
        self.origin, self.direction = origin, direction

    def __repr__(self):
        return f"Ray(origin={repr(self.origin)}, direction={repr(self.direction)})"

    def __getitem__(self, item):
        return Ray(self.origin, self.direction[item])

    def after(self, t, mask=None):
        if isinstance(t, Number):
            t = t
        else:
            t = t[:, np.newaxis]

        if mask is not None:
            direction = self.direction[mask]
        else:
            direction = self.direction

        return self.origin + direction * t

    def transform(self, transformation):
        return Ray(
            (transformation @ self.origin).view(self.origin.__class__),
            (transformation @ self.direction).view(self.direction.__class__),
        )

    def project(self, intersection, pixel_size, canvas_size, magnitude, anchor=None):
        if anchor is None:
            anchor = (canvas_size[0] / 2, canvas_size[1] / 2)

        canvas_width, canvas_height = canvas_size

        cols, rows = (
            (
                (self.direction * magnitude)[intersection.mask] / pixel_size
                + [*anchor, 0, 0]
            )
            .astype(int)[:, :2]
            .T
        )

        mask = (
            (0 <= rows) & (0 <= cols) & (rows < canvas_width) & (cols < canvas_height)
        )
        return rows[mask], cols[mask]
