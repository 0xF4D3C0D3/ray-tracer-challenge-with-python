import numpy as np

from src.canvas import Canvas
from src.grid import Color


def test_create_canvas():
    c = Canvas(10, 20)
    assert c.rows == 10
    assert c.cols == 20
    assert np.all(c == Color(0, 0, 0))


def test_writing_pixel_to_canvas():
    c = Canvas(10, 20)
    red = Color(1, 0, 0)
    c[2, 3] = red
    assert red == c[2, 3]


def test_constructing_ppm_header():
    c = Canvas(5, 3)
    ppm = c.to_ppm()
    line_1_3 = "\n".join(ppm.splitlines()[0:3])

    assert line_1_3 == ("P3\n" "5 3\n" "255")


def test_constructing_ppm_pixel_data():
    c = Canvas(5, 3)
    cs = Color([0, 0, 1], [0, 0.5, 0], [1.5, 0, -0.5])

    c[(0, 1, 2), (4, 2, 0)] = cs

    ppm = c.to_ppm()
    lines_4_6 = "\n".join(ppm.splitlines()[3:6])

    assert lines_4_6 == (
        "255 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n"
        "0 0 0 0 0 0 0 128 0 0 0 0 0 0 0\n"
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 255"
    )


def test_splitting_long_lines_in_ppm_files():
    c = Canvas(10, 2)
    c[:2, :10] = Color(1, 0.8, 0.6)
    ppm = c.to_ppm()
    lines_4_7 = "\n".join(ppm.splitlines()[3:7])
    assert lines_4_7 == (
        "255 204 153 255 204 153 255 204 153 255 204 153 255 204 153 255 204\n"
        "153 255 204 153 255 204 153 255 204 153 255 204 153\n"
        "255 204 153 255 204 153 255 204 153 255 204 153 255 204 153 255 204\n"
        "153 255 204 153 255 204 153 255 204 153 255 204 153"
    )


def test_ppm_files_are_terminated_by_a_newline_character():
    c = Canvas(5, 3)
    ppm = c.to_ppm()
    assert ppm[-1] == "\n"
