from src.grid import Color, Point, Vector
from src.light import Light
from src.material import Material

m = Material()
position = Point(0, 0, 0)


def test_default_material():
    assert m.color == Color(1, 1, 1)
    assert m.ambient == 0.1
    assert m.diffuse == 0.9
    assert m.specular == 0.9
    assert m.shininess == 200.0


def test_lighting_with_eye_between_light_and_surface():
    eyev = Vector(0, 0, -1)
    normalv = Vector(0, 0, -1)
    light = Light(Point(0, 0, -10), Color(1, 1, 1))
    result = light.get_color(m, position, eyev, normalv)
    assert result == Color(1.9, 1.9, 1.9)


def test_lighting_with_eye_between_light_and_surface_when_eye_offset_45_degree():
    eyev = Vector(0, 2 ** 0.5 / 2, -(2 ** 0.5) / 2)
    normalv = Vector(0, 0, -1)
    light = Light(Point(0, 0, -10), Color(1, 1, 1))
    result = light.get_color(m, position, eyev, normalv)
    assert result == Color(1, 1, 1)


def test_lighting_with_eye_opposite_surface_when_light_offset_45_degree():
    eyev = Vector(0, 0, -1)
    normalv = Vector(0, 0, -1)
    light = Light(Point(0, 10, -10), Color(1, 1, 1))
    result = light.get_color(m, position, eyev, normalv)
    assert result == Color(0.7364, 0.7364, 0.7364)


def test_lighting_with_eye_in_path_of_reflection_vector():
    eyev = Vector(0, -(2 ** 0.5) / 2, -(2 ** 0.5) / 2)
    normalv = Vector(0, 0, -1)
    light = Light(Point(0, 10, -10), Color(1, 1, 1))
    result = light.get_color(m, position, eyev, normalv)
    assert result == Color(1.6364, 1.6364, 1.6364)


def test_ligthing_with_light_behind_surface():
    eyev = Vector(0, 0, -1)
    normalv = Vector(0, 0, -1)
    light = Light(Point(0, 0, 10), Color(1, 1, 1))
    result = light.get_color(m, position, eyev, normalv)
    assert result == Color(0.1, 0.1, 0.1)
