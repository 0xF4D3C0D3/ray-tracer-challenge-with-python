from src.grid import Color


class Material:
    def __init__(
        self, color=None, ambient=0.1, diffuse=0.9, specular=0.9, shininess=200.0
    ):
        self.color = Color(1, 1, 1) if color is None else color
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess

    def __repr__(self):
        return (
            f"Material({repr(self.color)}, {repr(self.ambient)}, "
            f"{repr(self.diffuse)}, {repr(self.specular)}, {repr(self.shininess)})"
        )

    def __eq__(self, other):
        return (
            self.color == other.color
            and self.ambient == other.ambient
            and self.diffuse == other.diffuse
            and self.specular == other.specular
            and self.shininess == other.shininess
        )
