import numpy as np

from src.grid import ColorGrid


class Light:
    def __init__(self, position, intensity):
        self._position = position
        self._intensity = intensity

    @property
    def position(self):
        return self._position

    @property
    def intensity(self):
        return self._intensity

    def get_color(self, material, position, eyev, normalv):
        effective_color = material.color * self.intensity
        lightv = (self.position - position).normalize()
        ambient = effective_color * material.ambient

        light_dot_normal = lightv @ normalv
        light_mask = (light_dot_normal >= 0).flatten()

        reflectv = (-lightv).reflect(normalv)
        reflect_dot_eye = reflectv @ eyev
        reflect_mask = (reflect_dot_eye > 0).flatten() & light_mask

        diffuse = ColorGrid(*np.zeros((3, len(light_dot_normal))))
        specular = ColorGrid(*np.zeros((3, len(reflect_dot_eye))))

        masked_light_dot_normal = light_dot_normal[light_mask]
        masked_reflect_dot_eye = reflect_dot_eye[reflect_mask]

        diffuse[light_mask] = (
            effective_color * material.diffuse * masked_light_dot_normal
        )
        factor = masked_reflect_dot_eye ** material.shininess
        specular[reflect_mask] = self.intensity * material.specular * factor

        return ambient + diffuse + specular
