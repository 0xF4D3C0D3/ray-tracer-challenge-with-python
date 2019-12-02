import textwrap

import matplotlib.pyplot as plt
import numpy as np


class Canvas(np.ndarray):
    """
    Canvas holds (cols, rows, 3) ndarray for drawing various formats.
    """

    def __new__(cls, rows, cols):
        obj = np.zeros((cols, rows, 3)).view(Canvas)
        obj.rows = rows
        obj.cols = cols
        return obj

    def to_matplotlib(self, figsize=(10, 10)):
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(np.flipud(self.clip(0, 1)))
        return fig, ax

    def to_ppm(self):
        header = "P3\n" f"{self.rows} {self.cols}\n" "255\n"

        raw_body = (
            np.ceil((np.flipud(self) * 255).clip(0, 255))
            .astype(int)
            .astype(str)
            .reshape(self.cols, -1)
        )
        raw_lines = [" ".join(["".join(cell) for cell in row]) for row in raw_body]
        body = "\n".join(["\n".join(textwrap.wrap(line)) for line in raw_lines])

        ppm = header + body + "\n"

        return ppm
