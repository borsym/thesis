from scipy.spatial import procrustes
import numpy as np
import open3d as o3d
import cv2

human_3d = [
    [0.75806,    -0.29339,     0.80002],
    [0.74676, -0.34286,     0.82705],
    [0.75324, -0.31257,     0.77311],
    [0.8689, -0.41052,     0.97738],
    [0.72693, -0.3229,     0.74793],
    [0.8882, -0.27985,      1.1541],
    [0.57936, -0.13603,     0.64144],
    [0.9411, -0.025911,      1.2241],
    [0.73512, 0.069412,     0.73675],
    [1.1602, 0.15974,      1.2515],
    [1.0148, 0.20998,     0.91816],
    [1.0003, 0.18057,      1.3188],
    [1.0742, 0.22673,       1.159],
    [1.0496, 0.52266,      1.5001],
    [1.191, 0.56364,      1.3383],
    [1.028, 0.86823,      1.6038],
    [1.231, 0.88971,      1.4713]
]

connections = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
               (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]
human_3d = np.array(human_3d)
person_skeleton_cloud = o3d.geometry.PointCloud(
    o3d.utility.Vector3dVector(human_3d))

person_lines = o3d.geometry.LineSet()
person_lines.points = person_skeleton_cloud.points
person_lines.lines = o3d.utility.Vector2iVector(connections)
person_lines.paint_uniform_color([0, 1, 0])


mesh = o3d.io.read_triangle_mesh(
    "D:/thesis/realtime_update/meshes/beanbag.ply")

if not mesh.has_vertex_colors():
    mesh.vertex_colors = o3d.utility.Vector3dVector(
        [[0.5, 0.5, 0.5] for _ in range(len(mesh.vertices))])
if not mesh.has_vertex_normals():
    mesh.compute_vertex_normals()

def extract_corresponding_mesh_points(mesh, human_3d):
    mesh_vertices = np.asarray(mesh.vertices)
    extracted_points = np.zeros_like(human_3d)

    for i, point in enumerate(human_3d):
        # Compute distances from this human point to all mesh vertices
        distances = np.linalg.norm(mesh_vertices - point, axis=1)
        # Find the index of the closest mesh vertex
        closest_point_idx = np.argmin(distances)
        # Extract the closest vertex
        extracted_points[i] = mesh_vertices[closest_point_idx]

    return extracted_points


mesh_vertices_subset = extract_corresponding_mesh_points(mesh, human_3d)
# Perform Procrustes analysis with the subset
mtx1, mtx2, disparity = procrustes(mesh_vertices_subset, human_3d)

# mtx1 is the aligned mesh subset, mtx2 is the aligned human skeleton points
# For visualization, let's update the human skeleton points
person_skeleton_cloud.points = o3d.utility.Vector3dVector(mtx2)

# Update the lines to match the new points
person_lines.points = person_skeleton_cloud.points

# Visualize the aligned points along with the original mesh
o3d.visualization.draw_geometries([mesh, person_skeleton_cloud, person_lines])
