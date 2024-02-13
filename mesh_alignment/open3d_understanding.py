import open3d as o3d
import copy
import numpy as np
from scipy.spatial import procrustes


human_3d = [
    [0.26626, -0.77746,    2.8566],
    [0.27551,  -0.79932,      2.819],
    [0.26605,  -0.80444,     2.8521],
    [0.3398,  -0.75837,     2.6685],
    [0.40176,  -0.79614,     2.8263],
    [0.35792,  -0.52819,      2.574],
    [0.52638,  -0.58967,     2.9048],
    [0.34076,  -0.22582,     2.5475],
    [0.55342,  -0.31466,     3.0782],
    [0.27764, 0.0043032,     2.6044],
    [0.38004,  -0.10257,     3.2292],
    [0.40329, 0.0046143,     2.7028],
    [0.54785, -0.028368,     2.8938],
    [0.29032,    0.3715,     2.8131],
    [0.60189,   0.33737,     2.9271],
    [0.13915,   0.74992,     2.9061],
    [0.67337,   0.72656,     2.9738]
]
R_ext = np.array([
    [0.07541070458878951, -0.16218477631596587, 0.9838746485019128],
    [0.12432429135836522, 0.980514408874011, 0.1521018230288162],
    [-0.9893718695271574, 0.11084941281661857, 0.09410478981724704]
])
T_ext = np.array([
    [-3221.352499290285],
    [-239.17093718886326],
    [5148.128536539155]
])

camera_positions = np.array([
    [-5.61087799, -0.10080624, 0.24461792],
    [-1.26616669, 0.10356972, 5.83888578]
])


mesh = o3d.io.read_triangle_mesh(
    "D:/thesis/realtime_update/meshes/beanbag.ply")

skeleton = o3d.geometry.PointCloud()
skeleton.points = o3d.utility.Vector3dVector(human_3d)

transformation_matrix = np.eye(4)
transformation_matrix[:3, :3] = R_ext
transformation_matrix[:3, 3] = T_ext.flatten()


def normal_look():
    # visualize the mesh and the skeleton
    o3d.visualization.draw_geometries([mesh, skeleton])


def procrustes_2():
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
    skeleton.points = o3d.utility.Vector3dVector(mtx2)
    o3d.visualization.draw_geometries([mesh, skeleton])


def proc_3():
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

    human_3d.transform(transformation_matrix)
    mesh_vertices_subset = extract_corresponding_mesh_points(mesh, human_3d)
# Perform Procrustes analysis with the subset
    mtx1, mtx2, disparity = procrustes(mesh_vertices_subset, human_3d)

    # mtx1 is the aligned mesh subset, mtx2 is the aligned human skeleton points
    # For visualization, let's update the human skeleton points
    skeleton.points = o3d.utility.Vector3dVector(mtx2)
    o3d.visualization.draw_geometries([mesh, skeleton])


def third():
    # Transform keypoints to world coordinate system
    keypoints_world = np.dot(human_3d, R_ext.T) + T_ext.reshape(-1)
    skeleton.points = o3d.utility.Vector3dVector(keypoints_world)
    o3d.visualization.draw_geometries([mesh, skeleton])


def icp():

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = mesh.vertices

    initial_position = np.mean(camera_positions, axis=0)
    init_transformation = np.eye(4)
    init_transformation[:3, 3] = initial_position

    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R_ext
    transformation_matrix[:3, 3] = T_ext.flatten()
    skeleton.transform(transformation_matrix)

    icp_result = o3d.pipelines.registration.registration_icp(
        point_cloud, skeleton, max_correspondence_distance=0.2,
        init=init_transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=50)
    )
    skeleton.transform(icp_result.transformation)
    o3d.visualization.draw_geometries(
        [mesh, skeleton], window_name="ICP Alignment Result")


if __name__ == "__main__":
    # normal_look()
    # procrustes_2()
    # proc_3()
    # third()
    # icp()
    print(np.asarray(mesh.vertices).shape)
