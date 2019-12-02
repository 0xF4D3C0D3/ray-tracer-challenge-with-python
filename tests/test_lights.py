import numpy as np

from src.grid import Color, Point
from src.light import Light


def test_point_light_has_position_and_intensity():
    intensity = Color(1, 1, 1)
    position = Point(0, 0, 0)
    light = Light(position, intensity)
    assert light.position == position
    assert light.intensity == intensity
