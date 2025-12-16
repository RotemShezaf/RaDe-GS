import open3d as o3d
import numpy as np
import cv2

# -----------------------
# 1️⃣ Create mesh from polynomial z = x^2 + y^2
# -----------------------
n = 200
x = np.linspace(-1, 1, n)
y = np.linspace(-1, 1, n)
xx, yy = np.meshgrid(x, y)
zz = xx**2 + yy**2

vertices = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)

triangles = []
for i in range(n - 1):
    for j in range(n - 1):
        v0 = i * n + j
        v1 = v0 + 1
        v2 = v0 + n
        v3 = v2 + 1
        triangles += [[v0, v2, v1], [v1, v2, v3]]

mesh = o3d.geometry.TriangleMesh(
    o3d.utility.Vector3dVector(vertices),
    o3d.utility.Vector3iVector(triangles)
)

mesh.compute_vertex_normals()

# -----------------------
# 2️⃣ Generate UVs
# -----------------------
u = (xx.flatten() + 1) / 2
v = (yy.flatten() + 1) / 2
uv = np.stack([u, v], axis=-1)

triangle_uvs = []
for tri in triangles:
    triangle_uvs.append(uv[tri[0]])
    triangle_uvs.append(uv[tri[1]])
    triangle_uvs.append(uv[tri[2]])

mesh.triangle_uvs = o3d.utility.Vector2dVector(np.array(triangle_uvs))

# -----------------------
# 3️⃣ Generate procedural albedo texture (checker)
# -----------------------
tex_size = 512
texture = np.zeros((tex_size, tex_size, 3), dtype=np.uint8)
for i in range(tex_size):
    for j in range(tex_size):
        if ((i//32 + j//32) % 2) == 0:
            texture[i, j] = [200, 150, 100]  # albedo color 1
        else:
            texture[i, j] = [100, 200, 150]  # albedo color 2

cv2.imwrite("texture.png", texture)
#mesh.textures = [o3d.geometry.Image(texture)]

# -----------------------
# 4️⃣ Offscreen rendering with random camera
# -----------------------
import open3d.visualization.rendering as rendering

renderer = rendering.OffscreenRenderer(1024, 1024)
material = rendering.MaterialRecord()
material.shader = "defaultLit"

renderer.scene.add_geometry("mesh", mesh, material)

# Random camera position around the mesh
center = mesh.get_center()
radius = 3.0
theta = np.random.uniform(0, 2*np.pi)
phi = np.random.uniform(np.pi/6, np.pi/3)  # above xy-plane

cam_pos = center + radius * np.array([
    np.cos(theta)*np.cos(phi),
    np.sin(theta)*np.cos(phi),
    np.sin(phi)
])

renderer.scene.camera.look_at(center, cam_pos, [0,0,1])

# Lighting
renderer.scene.scene.set_sun_light([1,-1,-1], [1,1,1], 75000)
renderer.scene.scene.enable_sun_light(True)

# Render image
img = renderer.render_to_image()
breakpoint
o3d.io.write_image("polynomial_render.png", img)
print("Rendered image saved: polynomial_render.png")