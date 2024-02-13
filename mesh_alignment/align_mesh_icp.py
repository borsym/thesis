from scipy.spatial.transform import Rotation as R
import copy
import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize

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


# Given data
camera_positions = np.array([
    [-5.61087799, -0.10080624, 0.24461792],
    [-1.26616669, 0.10356972, 5.83888578]
])
human_skeleton = np.array([
    [0.75806, -0.29339, 0.80002],
    [0.74676, -0.34286, 0.82705],
    [0.75324, -0.31257, 0.77311],
    [0.8689, -0.41052, 0.97738],
    [0.72693, -0.3229, 0.74793],
    [0.8882, -0.27985, 1.1541],
    [0.57936, -0.13603, 0.64144],
    [0.9411, -0.025911, 1.2241],
    [0.73512, 0.069412, 0.73675],
    [1.1602, 0.15974, 1.2515],
    [1.0148, 0.20998, 0.91816],
    [1.0003, 0.18057, 1.3188],
    [1.0742, 0.22673, 1.159],
    [1.0496, 0.52266, 1.5001],
    [1.191, 0.56364, 1.3383],
    [1.028, 0.86823, 1.6038],
    [1.231, 0.88971, 1.4713]
])

# Intristic and extristic parameters
cam0_K = np.array([
    [1143.5629058671955, 0.0, 897.4701399737154],
    [0.0, 1146.6229630039772, 571.4789156478826],
    [0.0, 0.0, 1.0]
])
cam0_dist = np.array([-0.20212810339004855, 0.016856910836021612, -
                     0.00038445457626206384, 0.0029733591935219177, 0.028772663897866188])

cam1_K = np.array([
    [1390.5311354774792, 0.0, 941.6657612837951],
    [0.0, 1396.1775075864095, 564.5650596969682],
    [0.0, 0.0, 1.0]
])
cam1_dist = np.array([-0.37112152866840364, 0.3342635457860893,
                     0.00089837167379825, -0.00047573092618939504, -0.2159938879592358])

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

source_mesh = o3d.io.read_triangle_mesh(
    "D:/thesis/realtime_update/meshes/beanbag.ply")
if not source_mesh.has_vertex_colors():
    source_mesh.vertex_colors = o3d.utility.Vector3dVector(
        [[0.5, 0.5, 0.5]] * len(source_mesh.vertices))
if not source_mesh.has_vertex_normals():
    source_mesh.compute_vertex_normals()


skeleton = o3d.geometry.PointCloud()
skeleton.points = o3d.utility.Vector3dVector(human_3d)


source_mesh_vertices = source_mesh.vertices

# visualize skeleton and source mesh_vertices
o3d.visualization.draw_geometries([skeleton, source_mesh])


# def second_try():
#     """
#     Aligns two point clouds using the Iterative Closest Point (ICP) algorithm.
#     This is a simplified version and may require further optimization for robustness and accuracy.

#     :param source: Nx3 numpy array of points to align to the target.
#     :param target: Mx3 numpy array of points considered as the fixed target.
#     :return: The aligned source point cloud.
#     """
#     rot = R.from_matrix(R_ext)

#     target = rot.apply(human_skeleton) + T_ext.T
#     # The skeleton is now in the same coordinate system as defined by the extrinsic parameters.
#     # This code block does not align the point clouds using ICP or Procrustes as the alignment here depends on camera extrinsics and the provided 3D points.
#     # For further alignment with the room's mesh, additional steps involving ICP or similar algorithms might be required based on the room's 3D data.

#     # Initial guess for the transformation
#     init_translation = np.mean(source_mesh, axis=0) - np.mean(target, axis=0)
#     init_rotation = R.from_euler('xyz', [0, 0, 0], degrees=True)

#     source_centered = target - np.mean(target, axis=0)
#     target_centered = source_mesh - np.mean(source_mesh, axis=0)

#     def objective_function(params):
#         angle_x, angle_y, angle_z, tx, ty, tz = params
#         rotation = R.from_euler(
#             'xyz', [angle_x, angle_y, angle_z], degrees=True)
#         transformed_source = rotation.apply(source_centered) + [tx, ty, tz]
#         nbrs = NearestNeighbors(
#             n_neighbors=1, algorithm='auto').fit(target_centered)
#         distances, _ = nbrs.kneighbors(transformed_source)
#         return np.sum(distances)

#     # Optimization to minimize the distance between source and target point clouds
#     result = minimize(
#         objective_function,
#         np.hstack((init_rotation.as_euler(
#             'xyz', degrees=True), init_translation)),
#         method='Powell'
#     )

#     # Applying the optimized transformation
#     optimized_angles = result.x[:3]
#     optimized_translation = result.x[3:]
#     optimized_rotation = R.from_euler('xyz', optimized_angles, degrees=True)
#     aligned_source = optimized_rotation.apply(
#         source_centered) + optimized_translation

#     result = aligned_source + np.mean(target, axis=0)  # ez az eredmeny

# # Align human skeleton to the room's coordinate system

#     print(result)


# second_try()


# def first_try():
#     # We will not directly use the intrinsic and distortion parameters for aligning point clouds.
#     # Instead, we transform the skeleton points using the extrinsic parameters.

#     # Convert rotation matrix to a rotation object

#     # human_3d = np.array(human_3d)
#     # # Load the target point cloud (the one that's upside down)
#     # target = o3d.geometry.PointCloud()
#     # target.points = o3d.utility.Vector3dVector(human_3d)

#     # source = o3d.io.read_triangle_mesh(
#     #     "D:/thesis/realtime_update/meshes/beanbag.ply")

#     # if not source.has_vertex_colors():
#     #     source.vertex_colors = o3d.utility.Vector3dVector(
#     #         [[0.5, 0.5, 0.5] for _ in range(len(source.vertices))])
#     # if not source.has_vertex_normals():
#     #     source.compute_vertex_normals()

#     # point_cloud = o3d.geometry.PointCloud()
#     # point_cloud.points = source.vertices
#     # # ###Example: Rotate the target model 180 degrees around the X-axis to flip it upright
#     # # ###R = target.get_rotation_matrix_from_xyz((np.pi, 0, 0))  # Rotate 180 degrees around X-axis
#     # # ###target.rotate(R, center=(0,0,0))

#     # # Create a 4x4 transformation matrix from R and T
#     # transformation_matrix = np.eye(4)
#     # transformation_matrix[:3, :3] = R
#     # transformation_matrix[:3, 3] = T.flatten()

#     # target.transform(transformation_matrix)
#     # # R = o3d.geometry.get_rotation_matrix_from_xyz((0, np.pi, 0))
#     # # initial_position = np.mean(camera_positions, axis=0)
#     # # init_transformation = np.eye(4)
#     # # init_transformation[:3, 3] = initial_position
#     # # # Run ICP
#     # # icp_result = o3d.pipelines.registration.registration_icp(
#     # #     point_cloud, target, max_correspondence_distance=1.2,
#     # #     init=init_transformation,
#     # #     estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
#     # #     criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
#     # #         max_iteration=50)
#     # # )

#     # # # Apply the resulting transformation to the target model
#     # # target.transform(icp_result.transformation)

#     # # Visualize the result
#     # o3d.visualization.draw_geometries([source, target])

#     # # Save the adjusted model
#     # # o3d.io.write_point_cloud("path/to/adjusted_model.ply", target)

#     # Camera positions and skeleton data as provided
#     # camera_positions = np.array([...])  # Same as your data
#     # R = np.array([...])  # Extrinsic rotation
#     # T = np.array([...])  # Extrinsic translation
#     # human_3d = np.array([...])  # Your skeleton points

#     # Load and prepare the mesh (source) and convert it to a point cloud
#     source_mesh = o3d.io.read_triangle_mesh(
#         "D:/thesis/realtime_update/meshes/beanbag.ply")
#     if not source_mesh.has_vertex_colors():
#         source_mesh.vertex_colors = o3d.utility.Vector3dVector(
#             [[0.5, 0.5, 0.5]] * len(source_mesh.vertices))
#     if not source_mesh.has_vertex_normals():
#         source_mesh.compute_vertex_normals()
#     point_cloud = source_mesh.sample_points_poisson_disk(
#         number_of_points=1000)  # Optional: Adjust sample size

#     # Prepare the target point cloud (human skeleton)
#     target = o3d.geometry.PointCloud()
#     target.points = o3d.utility.Vector3dVector(human_3d)

#     # Create a 4x4 transformation matrix from R and T for initial alignment (if needed)
#     # Note: This step assumes the transformation is relevant to your alignment strategy
#     transformation_matrix = np.eye(4)
#     transformation_matrix[:3, :3] = R
#     transformation_matrix[:3, 3] = T.flatten()
#     # Consider applying this to the target if it helps with initial alignment:
#     # target.transform(transformation_matrix)

#     # Determine the initial position based on camera positions (for ICP initialization)
#     initial_position = np.mean(camera_positions, axis=0)
#     init_transformation = np.eye(4)
#     # Adjust based on scene specifics
#     init_transformation[:3, 3] = initial_position

#     # Correct upside-down orientation by rotating 180 degrees around the X-axis
#     rotation_matrix_x = o3d.geometry.get_rotation_matrix_from_xyz(
#         (np.pi, 0, 0))
#     target.rotate(rotation_matrix_x, center=target.get_center())

#     # Adjust target position to stand on the ground
#     # Here we assume the ground level is at y=0, adjust this as necessary for your scene
#     min_y = np.min(np.asarray(target.points)[:, 1])
#     # Adjust Y-axis to move to ground level, assuming Y is vertical
#     translation_to_ground = [0, -min_y, 0]
#     target.translate(translation_to_ground)

#     # Run ICP with adjusted parameters
#     icp_result = o3d.pipelines.registration.registration_icp(
#         # Adjust based on your scene scale
#         point_cloud, human_skeleton_transformed, max_correspondence_distance=0.1,
#         init=init_transformation,
#         estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
#         criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
#             max_iteration=50)
#     )

#     # Apply the resulting transformation to the target model
#     target.transform(icp_result.transformation)

#     # Visualize the result
#     o3d.visualization.draw_geometries(
#         [source_mesh, human_skeleton_transformed], window_name="ICP Alignment Result")
