import numpy as np

import rt

canvas_pixels = 1000
canvas = rt.Canvas(canvas_pixels, canvas_pixels)

wall_size = 7
wall_z = 10
pixel_size = wall_size / canvas_pixels

transformation = rt.Rotation(0.5, 0.5, np.pi/6) @ rt.Scaling(1.5, 0.8, 0.3) @ rt.Translation(0.1, 0, 0)
sphere = rt.Sphere(transformation, rt.Material(rt.Color(1, 0.5, 0.2)))

ray_origin = rt.Point(0, 0, -5)
ray_direction = rt.VectorGrid(np.arange(-wall_size, wall_size, pixel_size),
                              np.arange(-wall_size, wall_size, pixel_size),
                              wall_z)
ray = rt.Ray(ray_origin, ray_direction.normalize())
light = rt.Light(rt.Point(-10, 10, 0), rt.Color(1, 1, 1))

intersection = sphere.intersect(ray)

points = ray.after(intersection.hit, intersection.mask)
normals = intersection.obj.normal_at(points)# xs.hit.obj.normal_at(point)
eyes = -ray.direction[intersection.mask]
colors = light.get_color(sphere.material, points, eyes, normals)

cols, rows = ray.project(intersection, pixel_size, (canvas_pixels, canvas_pixels), ray_direction.magnitude)
canvas[cols, rows] = colors

fig, ax = canvas.to_matplotlib((20, 20))
fig.savefig('3d_sphere.png')
