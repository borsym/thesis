import copy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
from scipy.spatial import procrustes


robot_keypoints = np.array([
    [-0.4429,    0.60455,     4.1511],
    [-0.22797,  0.5718,  4.4382],
    [0.025888, 0.59645,  3.7247],
    [0.27237, 0.55627,  4.0103],
    [-0.41515, 0.29699,  4.3268],
    [-0.3097, 0.34575,   4.341],
    [-0.39385, 0.35938,   4.226],
    [-0.35421, 0.022742,  4.1289],
    [-0.35107, -0.13321,  4.1824],
    [-0.27508, 0.012975,  4.2428],
    [-0.034737, 0.41228,  4.1487],
    [-0.15411, 0.42936,  3.9748],
    [0.26888, 0.36082,  3.8191],
    [0.18203, 0.37811,  3.7176],
    [0.16138, 0.053772,  3.6926],
    [0.22111, 0.042928,  3.7952],
    [0.28559, 0.068742,  3.6676]
]
)

nerf_robot_keypoints = np.array([
    [-0.88645411, -0.44562197, 0.93118215],
    [-0.63839042, -0.37820864, -1.22604775],
    [0.83720541, -0.33930898, 1.09157586],
    [1.00591755, -0.24008346, -1.06751966],
    [-1.39356089, 0.32691932, -0.14874268],
    [-1.04946733, 0.22727919, -0.5598197],
    [-1.1274302, 0.16455293, 0.26233435],
    [-1.02111721, 1.14981174, 0.29777205],
    [-1.28275752, 1.59558785, -0.07786727],
    [-0.99985462, 1.190431, -0.446419],
    [0.09162557, 0.18090272, -0.83623338],
    [-0.03595006, 0.10247564, 0.74428666],
    [1.17964864, 0.3623569, -0.29758072],
    [1.11931801, 0.30521822, 0.43243515],
    [0.86416674, 1.27875948, 0.51748562],
    [0.89251685, 1.31221807, -0.29049325],
    [1.25816941, 1.35461175, 0.17728388]
])

connections = [[
    1,
    3
],
    [
    1,
    2
],
    [
    2,
    4
],
    [
    4,
    3
],
    [
    6,
    7
],
    [
    7,
    5
],
    [
    6,
    5
],
    [
    2,
    6
],
    [
    7,
    1
],
    [
    8,
    7
],
    [
    8,
    5
],
    [
    5,
    9
],
    [
    9,
    8
],
    [
    8,
    10
],
    [
    10,
    6
],
    [
    10,
    5
],
    [
    10,
    9
],
    [
    4,
    11
],
    [
    2,
    11
],
    [
    1,
    12
],
    [
    3,
    12
],
    [
    13,
    6
],
    [
    13,
    14
],
    [
    14,
    7
],
    [
    14,
    15
],
    [
    15,
    16
],
    [
    16,
    13
],
    [
    13,
    17
],
    [
    14,
    17
],
    [
    16,
    17
],
    [
    15,
    17
],
    [
    13,
    11
],
    [
    6,
    11
],
    [
    7,
    12
],
    [
    14,
    12
],
    [
    16,
    10
],
    [
    8,
    15
],
    [
    3,
    14
],
    [
    4,
    13
]
]
# visualize the nerf_robot_keypoints with matplotlib and use the connections
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# x = nerf_robot_keypoints[:, 0]
# y = nerf_robot_keypoints[:, 1]
# z = nerf_robot_keypoints[:, 2]

# # Plot keypoints
# ax.scatter(x, y, z)

# # Draw lines for each connection

# for i, pair in enumerate(connections):
#     start_point = nerf_robot_keypoints[pair[0]-1]
#     end_point = nerf_robot_keypoints[pair[1]-1]
#     ax.plot([start_point[0], end_point[0]], [start_point[1],
#             end_point[1]], [start_point[2], end_point[2]], 'r-')

# # Setting labels
# ax.set_xlabel('X axis')
# ax.set_ylabel('Y axis')
# ax.set_zlabel('Z axis')

# plt.title('Nerf Robot Keypoints with Connections Visualization')
# plt.show()

# exit(0)


keypoints_with_name = [
    "rightFrontWheel",
    "leftFrontWheel",
    "rightBackWheel",
    "leftBackWheel",
    "lidar",
    "leftFrontBottom",
    "rightFrontBottom",
    "rightFrontTop",
    "Hand",
    "leftFrontTop",
    "leftEmergencyStop",
    "rightEmergencyStop",
    "leftBackBottom",
    "rightBackBottom",
    "rightBackTop",
    "leftBackTop",
    "backScreen"
],

# Load the robot mesh
robot_mesh = o3d.io.read_triangle_mesh(
    "D:/thesis/realtime_update/meshes/save4.ply")

mesh_vertex_cloud = np.array(robot_mesh.vertices)


skeleton_robot = o3d.geometry.PointCloud()
skeleton_robot.points = o3d.utility.Vector3dVector(robot_keypoints)


# SZAR PONTOK
robot_ABCD = np.array([
    [-0.83684134, -0.46688437,  0.92713898],
    [-0.68091559, -0.31804633, -1.2244041],
    [0.77202892, -0.30531812, 1.09157586],
    [1.00591755, -0.18338323, -1.06859374]])


skeleton_ABCD = np.array([[-0.4429,    0.60455,     4.1511],
                          [-0.22797,  0.5718,  4.4382],
                          [0.025888, 0.59645,  3.7247],
                          [0.27237, 0.55627,  4.0103],])


def visualize_robot():
    # Adjusting connection indices to match Python's 0-based indexing

    # Plotting the keypoints with connections
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    x = robot_keypoints[:, 0]
    y = robot_keypoints[:, 1]
    z = robot_keypoints[:, 2]

    # Plot keypoints
    ax.scatter(x, y, z)
    connections_zero_based = [[p-1 for p in pair] for pair in connections]
    # Draw lines for each connection
    for i, pair in enumerate(connections_zero_based):
        start_point = robot_keypoints[pair[0]]
        end_point = robot_keypoints[pair[1]]
        ax.plot([start_point[0], end_point[0]], [start_point[1],
                end_point[1]], [start_point[2], end_point[2]], 'r-')

    # Setting labels
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    plt.title('Robot Keypoints with Connections Visualization')
    plt.show()


def visualize_mesh():
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    total_vertices = len(mesh_vertex_cloud)

    # Desired number of points
    sampled_points = sample_points(total_vertices)

    # Plot the point cloud
    ax.scatter(sampled_points[:, 0], sampled_points[:, 1],
               sampled_points[:, 2], c='b', marker='o')

    # Setting labels
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('Point Cloud Visualization')

    plt.show()


def sample_points(total_vertices, num_points=1500):
    # Calculate step size for uniform sampling
    step_size = total_vertices // num_points

    # Sample points
    sampled_points = mesh_vertex_cloud[::step_size]

    # If the sampled points are less than 1000, adjust by including more points from the end
    if len(sampled_points) < num_points:
        additional_points_needed = num_points - len(sampled_points)
        sampled_points = np.vstack(
            [sampled_points, mesh_vertex_cloud[-additional_points_needed:]])

    return sampled_points


def icp_calc():
    pcd_17 = o3d.geometry.PointCloud()
    pcd_17.points = o3d.utility.Vector3dVector(robot_keypoints)

    pcd_1500 = o3d.geometry.PointCloud()
    pcd_1500.points = o3d.utility.Vector3dVector(
        sample_points(len(mesh_vertex_cloud)))

    # Calculate centroids of both point clouds
    centroid_1500 = np.mean(np.asarray(pcd_1500.points), axis=0)
    centroid_17 = np.mean(np.asarray(pcd_17.points), axis=0)

    # Estimate scale factor (This is a simplified approach; you might need a more sophisticated method)
    # For a more accurate estimation, compare distances between points and centroids or between corresponding points
    scale_factor = np.linalg.norm(centroid_17) / np.linalg.norm(centroid_1500)
    print("Scale factor:", scale_factor)
    # Scale pcd_1500
    scaled_points_1500 = (np.asarray(pcd_1500.points) -
                          centroid_1500) / scale_factor + centroid_1500
    pcd_1500.points = o3d.utility.Vector3dVector(scaled_points_1500)

    # Apply ICP to align the mesh's point cloud to the keypoint cloud
    # Note: You might need to adjust the parameters based on your specific requirements
    icp_result = o3d.pipelines.registration.registration_icp(
        pcd_1500, pcd_17, 7, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))

    # Apply the resulting transformation to the original mesh
    pcd_1500.transform(icp_result.transformation)

    # Visualize the aligned mesh and keypoints
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    # Set color of pcd_1500 points to blue
    pcd_1500.paint_uniform_color([0, 0, 1])
    pcd_17.paint_uniform_color([1, 0, 0])  # Set color of pcd_17 points to red
    visualizer.add_geometry(pcd_1500)
    visualizer.add_geometry(pcd_17)
    visualizer.run()
    visualizer.destroy_window()


def icp_calc2():
    source = copy.deepcopy(mesh_vertex_cloud)
    target = copy.deepcopy(skeleton_robot)
    # keep only 17 points from the mesh use linalg to get the 17 points
    source = sample_points(len(source), 1500)

    # Convert source to PointCloud object
    source_cloud = o3d.geometry.PointCloud()
    source_cloud.points = o3d.utility.Vector3dVector(source)
    source_cloud.paint_uniform_color([0, 0, 1])

    # Compute normal vectors for target PointCloud
    target.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    print("Apply point-to-point ICP")
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_cloud, target, 4, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())

    # apply the transformation to the source
    source_cloud.transform(reg_p2p.transformation)
    # visualize
    o3d.visualization.draw_geometries([source_cloud, target])


def calculate_plane_robot():

    robot_mesh.compute_vertex_normals()
    # vector AB
    AB = skeleton_ABCD[1] - skeleton_ABCD[0]
    print("AB", AB)
    # vector CD
    CD = skeleton_ABCD[3] - skeleton_ABCD[2]
    print("CD", CD)
    # cross product
    normal_vector = np.cross(AB, CD)

    # plane equation
    print("The plane equation is: ",
          normal_vector[0], "x +", normal_vector[1], "y +", normal_vector[2], "z = d")

    robot_AB = robot_ABCD[1] - robot_ABCD[0]

    print("AB", robot_AB)
    # vector CD
    robot_CD = robot_ABCD[3] - robot_ABCD[2]
    print("CD", robot_CD)
    # cross product, for the robot
    robot_normal_vector = np.cross(robot_AB, robot_CD)

    print("The plane equation is: ",
          robot_normal_vector[0], "x +", robot_normal_vector[1], "y +", robot_normal_vector[2], "z = d")

    # magnitude of the normal vector, linalg.norm => sqrt(x^2 + y^2 + z^2)
    magnitude_normal = np.linalg.norm(normal_vector)
    print("magnitude of the normal vector", magnitude_normal)
    # magnitude of the robot_normal_vector
    magnitude_robot_normal = np.linalg.norm(robot_normal_vector)
    print("magnitude of the robot_normal_vector", magnitude_robot_normal)

    scale_factor = magnitude_normal / magnitude_robot_normal
    print("Scale factor:", scale_factor)

    # scale the robot_normal_vector ###########
    # robot_normal_vector = [i * scale_factor for i in robot_normal_vector]

    def calculate_rotation_matrix(N1, N2):
        # Normalize the vectors
        N1_norm = N1 / np.linalg.norm(N1)
        N2_norm = N2 / np.linalg.norm(N2)

        # Calculate the cross product and angle
        axis = np.cross(N1_norm, N2_norm)
        axis_norm = axis / np.linalg.norm(axis)
        angle = np.arccos(
            np.clip(np.dot(N1_norm, N2_norm), -1.0, 1.0))  # clip?

        # Skew-symmetric matrix for the axis
        K = np.array([[0, -axis_norm[2], axis_norm[1]],
                      [axis_norm[2], 0, -axis_norm[0]],
                      [-axis_norm[1], axis_norm[0], 0]])
        # Rodrigues' rotation formula
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)

        return R
    # Calculate the rotation matrix
    rotation_matrix = calculate_rotation_matrix(
        normal_vector, robot_normal_vector)
    # get the centroid of the robot_ABCD
    centroid_robot = np.mean(robot_ABCD, axis=0)
    # get the centroid of the A,B,C,D
    centroid_skeleton = np.mean(skeleton_ABCD, axis=0)
    # LEHET MEG KELL CSERELNI
    translation_vector = centroid_robot - centroid_skeleton

    print("Scale factor:", scale_factor)
    print("Rotation matrix:\n", rotation_matrix)
    print("Translation vector:", translation_vector)

    translated_vertices = np.asarray(robot_mesh.vertices) + translation_vector
    robot_mesh.vertices = o3d.utility.Vector3dVector(translated_vertices)

    # translate the skeleton
    skeleton_robot.translate(translation_vector)

    rotation_mat_4x4 = np.eye(4)
    rotation_mat_4x4[:3, :3] = rotation_matrix
    robot_mesh.transform(rotation_mat_4x4)

    scaled_vertices = np.asarray(robot_mesh.vertices) * scale_factor
    robot_mesh.vertices = o3d.utility.Vector3dVector(scaled_vertices)

    # Visualize the transformed mesh
    o3d.visualization.draw_geometries([robot_mesh, skeleton_robot])


# calculate_plane_robot()


def procrustrace():

    mtx1, mtx2, disparity = procrustes(skeleton_ABCD, robot_ABCD)

    print("Transformed skeleton points:\n", mtx1)
    print("Transformed robot_ABCD points to match skeleton:\n", mtx2)
    print("Disparity:", disparity)

    total_vertices = len(robot_mesh.vertices)
    sampled_points = sample_points(total_vertices, 800)

    centroid_robot = np.mean(robot_mesh.vertices, axis=0)

    # Calculate the new centroid based on the transformed keypoints (mtx2)
    centroid_transformed = np.mean(mtx2, axis=0)

    # Calculate translation vector
    translation_vector = centroid_transformed - centroid_robot

    # Translate the entire mesh
    transformed_vertices = sampled_points + translation_vector

    def visualize_m():
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(mtx1[:, 0], mtx1[:, 1], mtx1[:, 2], c='r', marker='o')
        ax.scatter(mtx2[:, 0], mtx2[:, 1], mtx2[:, 2], c='b', marker='o')

        # ax.scatter(robot_ABCD[:, 0], robot_ABCD[:, 1],
        #            robot_ABCD[:, 2], c='g', marker='o')

        # ax.scatter(skeleton_ABCD[:, 0], skeleton_ABCD[:, 1],
        #            skeleton_ABCD[:, 2], c='black', marker='o')
        # Plot keypoints

        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')

        plt.title('Transformed Robot Mesh')
        plt.show()
    # visualize_m()
    # exit(0)
    # visualize the transformed_vertices
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    x = transformed_vertices[:, 0]
    y = transformed_vertices[:, 1]
    z = transformed_vertices[:, 2]

    # Plot keypoints
    ax.scatter(x, y, z)
    # with mtx1 normalize the robot_keypoints
    # mtx1 = mtx1 / np.linalg.norm(mtx1)

    # robot_keypoints2 = robot_keypoints / np.linalg.norm(robot_keypoints)

    # visualize the robot_keypoints
    k = robot_keypoints[:, 0]
    m = robot_keypoints[:, 1]
    l = robot_keypoints[:, 2]

    # connections_zero_based = [[p-1 for p in pair] for pair in connections[:4]]
    # # Draw lines for each connection
    # for i, pair in enumerate(connections_zero_based):
    #     start_point = robot_keypoints[pair[0]]
    #     end_point = robot_keypoints[pair[1]]
    #     ax.plot([start_point[0], end_point[0]], [start_point[1],
    #             end_point[1]], [start_point[2], end_point[2]], 'r-')

    # Plot keypoints
    ax.scatter(k, m, l)

    # Setting labels
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    plt.title('Transformed Robot Mesh')
    plt.show()


def proc2():
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

    mesh_vertices_subset = extract_corresponding_mesh_points(
        robot_mesh, robot_keypoints)
    # Perform Procrustes analysis with the subset
    print("mesh_vertices_subset", len(mesh_vertices_subset))
    mtx1, mtx2, disparity = procrustes(robot_keypoints, mesh_vertices_subset)
    # print(mtx2)
    # transformed_vertices = np.dot(robot_mesh.vertices, mtx2.T)
    kecske = robot_mesh.vertices
    # select 1500 points from the transformed vertices
    total_vertices = len(kecske)
    transformed_vertices = sample_points(total_vertices, 1500)

    print("kecskeee", transformed_vertices[300])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    x = transformed_vertices[:, 0]
    y = transformed_vertices[:, 1]
    z = transformed_vertices[:, 2]

    # Plot keypoints
    ax.scatter(x, y, z)
    # visualize the robot_keypoints
    k = robot_keypoints[:, 0]
    m = robot_keypoints[:, 1]
    l = robot_keypoints[:, 2]

    connections_zero_based = [[p-1 for p in pair] for pair in connections]
    # Draw lines for each connection
    for i, pair in enumerate(connections_zero_based):
        start_point = robot_keypoints[pair[0]]
        end_point = robot_keypoints[pair[1]]
        ax.plot([start_point[0], end_point[0]], [start_point[1],
                end_point[1]], [start_point[2], end_point[2]], 'r-')

    # Plot keypoints
    ax.scatter(k, m, l)

    # Setting labels
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    plt.title('Transformed Robot Mesh')
    plt.show()

    # mtx1 is the aligned mesh subset, mtx2 is the aligned human skeleton points
    # For visualization, let's update the human skeleton points

    # Visualize the aligned points along with the original mesh
    # o3d.visualization.draw_geometries(
    #     [transformed_vertices, skeleton_robot])


def visualize(points):
    # visualize in matplotlib the points which are 3d coordinates
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    RightFrontWheel = points[0]
    LeftFrontWheel = points[1]
    RightBackWheel = points[2]
    LeftBackWheel = points[3]

    # Plot keypoints
    ax.scatter(RightFrontWheel[0], RightFrontWheel[1], RightFrontWheel[2])
    ax.scatter(LeftFrontWheel[0], LeftFrontWheel[1], LeftFrontWheel[2])
    ax.scatter(RightBackWheel[0], RightBackWheel[1], RightBackWheel[2])
    ax.scatter(LeftBackWheel[0], LeftBackWheel[1], LeftBackWheel[2])

    # put text on the points
    ax.text(RightFrontWheel[0], RightFrontWheel[1], RightFrontWheel[2],
            'wheelRight', color='black', fontsize=8)
    ax.text(LeftFrontWheel[0], LeftFrontWheel[1], LeftFrontWheel[2],
            'wheelLeft', color='black', fontsize=8)
    ax.text(RightBackWheel[0], RightBackWheel[1], RightBackWheel[2],
            'backRightWheel', color='black', fontsize=8)
    ax.text(LeftBackWheel[0], LeftBackWheel[1], LeftBackWheel[2],
            'backLeftWheel', color='black', fontsize=8)
    # connect the rightfrontwheel to leftfrontwheel and connect rightfrontwheel to rightbackwheel and connect leftfrontwheel to leftbackwheel and connect rightbackwheel to leftbackwheel
    ax.plot([RightFrontWheel[0], LeftFrontWheel[0]], [RightFrontWheel[1],
            LeftFrontWheel[1]], [RightFrontWheel[2], LeftFrontWheel[2]], 'r-')
    ax.plot([RightFrontWheel[0], RightBackWheel[0]], [RightFrontWheel[1],
            RightBackWheel[1]], [RightFrontWheel[2], RightBackWheel[2]], 'r-')
    ax.plot([LeftFrontWheel[0], LeftBackWheel[0]], [LeftFrontWheel[1],
            LeftBackWheel[1]], [LeftFrontWheel[2], LeftBackWheel[2]], 'r-')
    ax.plot([RightBackWheel[0], LeftBackWheel[0]], [RightBackWheel[1],
            LeftBackWheel[1]], [RightBackWheel[2], LeftBackWheel[2]], 'r-')

    # Setting labels
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    plt.title('Transformed Robot Mesh')
    plt.show()


def my_last_hope():
    mesh_vertices = np.asarray(robot_mesh.vertices)

    # Calculate centroids
    centroid_robot_keypoints = np.mean(robot_keypoints, axis=0)
    centroid_robot_mesh = np.mean(nerf_robot_keypoints, axis=0)

    centered_keypoints = robot_keypoints - centroid_robot_keypoints
    centered_mesh_vertices = nerf_robot_keypoints - centroid_robot_mesh

    mtx1, mtx2, disparity = procrustes(
        centered_keypoints, centered_mesh_vertices)

    # Calculate the pairwise distances in the original datasets

    scale_factor = 1.8

    transformed_mesh_vertices = mtx2 * scale_factor

    # Translate the transformed mesh vertices to match the keypoints centroid
    final_transformed_vertices = transformed_mesh_vertices + centroid_robot_keypoints

    # visualize in matplotlib the transofmerd_mesh and the robot_keypoints
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # keep every 1000th points from the tnrasformed_mesh
    # transformed_vertices = final_transformed_vertices[::3000, :]

    x = final_transformed_vertices[:, 0]
    y = final_transformed_vertices[:, 1]
    z = final_transformed_vertices[:, 2]

    # Plot keypoints
    ax.scatter(x, y, z)

    # visualize the robot_keypoints
    k = robot_keypoints[:, 0]
    m = robot_keypoints[:, 1]
    l = robot_keypoints[:, 2]

    connections_zero_based = [[p-1 for p in pair] for pair in connections]
    # Draw lines for each connection
    for i, pair in enumerate(connections_zero_based):
        start_point = robot_keypoints[pair[0]]
        end_point = robot_keypoints[pair[1]]
        ax.plot([start_point[0], end_point[0]], [start_point[1],
                end_point[1]], [start_point[2], end_point[2]], 'r-')

    # Plot keypoints

    ax.scatter(k, m, l)

    # Setting labels

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    plt.title('Transformed Robot Mesh')
    plt.show()


my_last_hope()
# visualize(robot_ABCD)
# visualize(skeleton_ABCD)
# calculate_plane_robot()
# procrustrace()
# proc2()


def visualize_robot_keypoints_mathplotlib():
    # visualize the robot_keyponits with matplotlib
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    x = robot_keypoints[:, 0]
    y = robot_keypoints[:, 1]
    z = robot_keypoints[:, 2]

    # Plot keypoints
    ax.scatter(x, y, z)
    connections_zero_based = [[p-1 for p in pair] for pair in connections[:4]]
    # Draw lines for each connection
    for i, pair in enumerate(connections_zero_based):
        start_point = robot_keypoints[pair[0]]
        end_point = robot_keypoints[pair[1]]
        ax.plot([start_point[0], end_point[0]], [start_point[1],
                end_point[1]], [start_point[2], end_point[2]], 'r-')

    # Setting labels
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    plt.title('Robot Keypoints with Connections Visualization')
    plt.show()


# visualize_robot_keypoints_mathplotlib()


def calibrate_by_procrustes(points3d, camera, gt):
    """Calibrates the predictied 3d points by Procrustes algorithm.

    This function estimate an orthonormal matrix for aligning the predicted 3d
    points to the ground truth. This orhtonormal matrix is computed by
    Procrustes algorithm, which ensures the global optimality of the solution.
    """
    # Shift the center of points3d to the origin
    if camera is not None:
        singular_value = np.linalg.norm(camera, 2)
        camera = camera / singular_value
        points3d = points3d * singular_value
    scale = np.linalg.norm(gt) / np.linalg.norm(points3d)
    points3d = points3d * scale

    U, s, Vh = np.linalg.svd(points3d.T.dot(gt))
    rot = U.dot(Vh)
    if camera is not None:
        return points3d.dot(rot), rot.T.dot(camera)
    else:
        return points3d.dot(rot), None
