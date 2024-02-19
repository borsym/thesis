import open3d as o3d
import numpy as np
import json


class Visualizer:
    """
    A class for visualizing 3D objects and skeletons.

    Attributes:
    - mesh: The triangle mesh object representing the environment.
    - robot_mesh: The triangle mesh object representing the robot.
    - nerf_robot_keypoints: The keypoints of the robot in the NeRF coordinate system.
    - first_robot_keypoints: The initial keypoints of the robot.
    - visualizer: The Open3D visualizer object.
    - connections: The connections between keypoints in the skeleton.
    - person_skeleton_cloud: The point cloud object representing the person's skeleton.
    - person_lines: The line set object representing the connections between keypoints in the person's skeleton.
    - robot_conections: The connections between keypoints in the robot's skeleton.
    - robot_skeleton_cloud: The point cloud object representing the robot's skeleton.
    - robot_lines: The line set object representing the connections between keypoints in the robot's skeleton.
    - previous_transfomation: The previous transformation matrix applied to the robot mesh.
    - direction_human_arrow_shaft: The line set object representing the shaft of the arrow indicating the direction of the person.
    - direction_human_arrow_head: The triangle mesh object representing the head of the arrow indicating the direction of the person.
    - direction_robot_arrow_shaft: The line set object representing the shaft of the arrow indicating the direction of the robot.
    - direction_robot_arrow_head: The triangle mesh object representing the head of the arrow indicating the direction of the robot.
    """

    def __init__(self) -> None:
        mesh = o3d.io.read_triangle_mesh(
            "D:/thesis/realtime_update/meshes/rotated_beanbag.ply")

        if not mesh.has_vertex_colors():
            mesh.vertex_colors = o3d.utility.Vector3dVector(
                [[0.5, 0.5, 0.5] for _ in range(len(mesh.vertices))])
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()

        self.robot_mesh = o3d.io.read_triangle_mesh(
            "D:/thesis/realtime_update/meshes/save4.ply")

        if not self.robot_mesh.has_vertex_colors():
            self.robot_mesh.vertex_colors = o3d.utility.Vector3dVector(
                [[0.5, 0.5, 0.5] for _ in range(len(self.robot_mesh.vertices))])
        if not self.robot_mesh.has_vertex_normals():
            self.robot_mesh.compute_vertex_normals()

        self.nerf_robot_keypoints = np.array([
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

        self.first_robot_keypoints = np.array([
            [-1.5052,  0.3022,  5.6128],
            [-1.3801,  0.33318,  5.8066],
            [-1.1815,  0.3669,  5.1104],
            [-1.0278,  0.39118,  5.3233],
            [-1.502,  0.037726,  5.7502],
            [-1.4306,  0.10683,  5.7399],
            [-1.489,  0.10121,  5.6533],
            [-1.4394, -0.22139,  5.5396],
            [-1.4233, -0.36347,  5.5755],
            [-1.3666, -0.21828,  5.6215],
            [-1.2277,  0.20028,  5.4801],
            [-1.3271,  0.19053,  5.3658],
            [-1.0123,  0.18271,  5.1322],
            [-1.0651,  0.17497,  5.0526],
            [-1.0838, -0.14808,  5.0171],
            [-1.0229, -0.14653,  5.1043],
            [-0.99055, -0.12007,  4.969]])

        self.visualizer = o3d.visualization.Visualizer()
        self.visualizer.create_window()

        self.connections = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
                            (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]

        self.person_skeleton_cloud = o3d.geometry.PointCloud()
        self.person_skeleton_cloud.paint_uniform_color([0, 1, 0])

        self.person_lines = o3d.geometry.LineSet()
        self.person_lines.points = self.person_skeleton_cloud.points
        self.person_lines.lines = o3d.utility.Vector2iVector(self.connections)
        self.person_lines.paint_uniform_color([0, 1, 0])

        self.robot_conections = self.get_skeleton_connection_robot()
        self.robot_skeleton_cloud = o3d.geometry.PointCloud()
        self.robot_skeleton_cloud.paint_uniform_color([0, 0, 1])

        self.robot_lines = o3d.geometry.LineSet()
        self.robot_lines.points = self.robot_skeleton_cloud.points
        self.robot_lines.lines = o3d.utility.Vector2iVector(
            self.robot_conections)
        self.robot_lines.paint_uniform_color([0, 0, 1])

        _, _, transformation = self.procrustes_mine(self.first_robot_keypoints, self.nerf_robot_keypoints,
                                                    scaling=True, reflection='best')
        initial_transformation_matrix = np.eye(4)
        initial_transformation_matrix[:3,
                                      :3] = transformation['rotation'] * transformation['scale']
        initial_transformation_matrix[:3, 3] = transformation['translation']

        # Apply the initial transformation to the mesh
        self.robot_mesh.transform(initial_transformation_matrix)
        self.previous_transfomation = np.linalg.inv(
            initial_transformation_matrix)
        self.visualizer.add_geometry(mesh)
        self.visualizer.add_geometry(self.robot_mesh)
        self.visualizer.add_geometry(self.person_lines)
        self.visualizer.add_geometry(self.person_skeleton_cloud)
        self.visualizer.add_geometry(self.robot_lines)
        self.visualizer.add_geometry(self.robot_skeleton_cloud)

        render_option = self.visualizer.get_render_option()
        render_option.mesh_show_back_face = True
        render_option.light_on = False

        self.direction_human_arrow_shaft = o3d.geometry.LineSet()
        self.direction_human_arrow_head = o3d.geometry.TriangleMesh.create_cone(
            radius=0.04, height=0.1)
        self.direction_human_arrow_head.paint_uniform_color(
            [1, 0, 0])  # Red color for the arrowhead
        self.direction_human_arrow_shaft.paint_uniform_color(
            [1, 0, 0])  # Red color for the arrow shaft
        self.visualizer.add_geometry(self.direction_human_arrow_shaft)
        # self.visualizer.add_geometry(self.direction_human_arrow_head)

        self.direction_robot_arrow_shaft = o3d.geometry.LineSet()
        self.direction_robot_arrow_head = o3d.geometry.TriangleMesh.create_cone(
            radius=0.04, height=0.1)
        self.direction_robot_arrow_head.paint_uniform_color(
            [1, 0, 0])  # Red color for the arrowhead
        self.direction_robot_arrow_shaft.paint_uniform_color(
            [1, 0, 0])  # Red color for the arrow shaft
        self.visualizer.add_geometry(self.direction_robot_arrow_shaft)
        # self.visualizer.add_geometry(self.direction_robot_arrow_head)

        view_control = self.visualizer.get_view_control()

        # Set the front, lookat, up, and zoom parameters
        camera_params = {
            # Assuming the camera is looking along the negative Z-axis
            "front": [4, 0, -1],
            "lookat": [3, 0, 3],  # The point at which the camera is looking
            # The "up" direction for the camera (here set to the negative Y-axis)
            "up": [0, -1, 0],
            "zoom": 0.02           # Zoom level
        }
        # Apply the camera parameters
        view_control.set_front(camera_params["front"])
        view_control.set_lookat(camera_params["lookat"])
        view_control.set_up(camera_params["up"])
        view_control.set_zoom(camera_params["zoom"])

    def update_arrow(self, start_point, direction_vector, object_type):
        """
        Update the arrow's position and orientation based on the given start point, direction vector, and object type.

        Parameters:
        start_point (numpy.ndarray): The starting point of the arrow.
        direction_vector (numpy.ndarray): The direction vector of the arrow.
        object_type (str): The type of the object associated with the arrow.

        Returns:
        None
        """
        # Update the arrow's shaft
        scaling_factor = 5  # Adjust the scaling factor as needed
        end_point = start_point + scaling_factor * direction_vector
        points = [start_point, end_point]
        lines = [[0, 1]]  # LineSet uses indices into the points list
        if object_type == "human":
            self.direction_human_arrow_shaft.points = o3d.utility.Vector3dVector(
                points)
            self.direction_human_arrow_shaft.lines = o3d.utility.Vector2iVector(
                lines)
        elif object_type == "robot":
            self.direction_robot_arrow_shaft.points = o3d.utility.Vector3dVector(
                points)
            self.direction_robot_arrow_shaft.lines = o3d.utility.Vector2iVector(
                lines)

        # Update the arrow's head
        # Place the cone at the end of the shaft and rotate it to point in the direction of the vector
        transformation_matrix = self.get_arrow_transformation_matrix(
            end_point, direction_vector)
        self.direction_human_arrow_head.transform(
            transformation_matrix)

        if object_type == "human":
            # Update the geometries
            self.visualizer.update_geometry(self.direction_human_arrow_shaft)
            self.visualizer.update_geometry(self.direction_human_arrow_head)
        elif object_type == "robot":
            # Update the geometries
            self.visualizer.update_geometry(self.direction_robot_arrow_shaft)
            self.visualizer.update_geometry(self.direction_robot_arrow_head)

    def get_arrow_transformation_matrix(self, end_point, direction_vector):
        """
        Calculates the transformation matrix for an arrow given the end point and direction vector.

        Parameters:
            end_point (numpy.ndarray): The coordinates of the arrow's end point.
            direction_vector (numpy.ndarray): The direction vector of the arrow.

        Returns:
            numpy.ndarray: The transformation matrix representing the arrow's rotation and translation.
        """
        # Normalize the direction vector
        direction = direction_vector / np.linalg.norm(direction_vector)

        # Create a rotation matrix that aligns the z-axis to the direction vector
        z_axis = np.array([0, 0, 1])
        rotation_axis = np.cross(z_axis, direction)
        rotation_angle = np.arccos(
            np.clip(np.dot(z_axis, direction), -1.0, 1.0))
        rotation_axis_angle = rotation_axis * rotation_angle
        rotation_axis_angle = np.expand_dims(rotation_axis_angle, axis=1)

        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(
            rotation_axis_angle)

        # Create the transformation matrix with rotation and translation
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = end_point
        return transformation_matrix

    def get_skeleton_connection_robot(self):
        """
        Retrieves the skeleton connections for the robot from a JSON file.

        Returns:
            list: A list of skeleton connections, where each connection is represented as a pair of indices.
        """
        with open("D:/thesis/realtime_update/robot-keypoints-connection.json", 'r') as file:
            annotations_data = json.load(file)
        skeleton_connections = annotations_data['skeleton']
        # shift the indices by 1 to match the 0-based indexing in open3d
        skeleton_connections = [[x - 1, y - 1]
                                for x, y in skeleton_connections]
        return skeleton_connections

    def update_open3d(self):
        """
        Updates the Open3D visualizer by polling events and updating the renderer.
        """
        self.visualizer.poll_events()
        self.visualizer.update_renderer()

    def update_sphere_position(self, sphere, new_center):
        # Calculate the current center of the sphere
        current_center = np.mean(np.asarray(sphere.vertices), axis=0)
        # Calculate the shift required to move to the new center
        shift = new_center - current_center
        # Update the vertices
        np_vertices = np.asarray(sphere.vertices)
        sphere.vertices = o3d.utility.Vector3dVector(np_vertices + shift)
        # Update the mesh (important for visualization)
        sphere.compute_vertex_normals()
        return sphere

    def procrustes_mine(self, X, Y, scaling=True, reflection='best'):
        """
        A port of MATLAB's `procrustes` function to Numpy.

        Procrustes analysis determines a linear transformation (translation,
        reflection, orthogonal rotation and scaling) of the points in Y to best
        conform them to the points in matrix X, using the sum of squared errors
        as the goodness of fit criterion.

            d, Z, [tform] = procrustes(X, Y)

        Inputs:
        ------------
        X, Y    
            matrices of target and input coordinates. they must have equal
            numbers of  points (rows), but Y may have fewer dimensions
            (columns) than X.

        scaling 
            if False, the scaling component of the transformation is forced
            to 1

        reflection
            if 'best' (default), the transformation solution may or may not
            include a reflection component, depending on which fits the data
            best. setting reflection to True or False forces a solution with
            reflection or no reflection respectively.

        Outputs
        ------------
        d       
            the residual sum of squared errors, normalized according to a
            measure of the scale of X, ((X - X.mean(0))**2).sum()

        Z
            the matrix of transformed Y-values

        tform   
            a dict specifying the rotation, translation and scaling that
            maps X --> Y

        """

        n, m = X.shape
        ny, my = Y.shape

        muX = X.mean(0)
        muY = Y.mean(0)

        X0 = X - muX
        Y0 = Y - muY

        ssX = (X0**2.).sum()
        ssY = (Y0**2.).sum()

        # centred Frobenius norm
        normX = np.sqrt(ssX)
        normY = np.sqrt(ssY)

        # scale to equal (unit) norm
        X0 /= normX
        Y0 /= normY

        if my < m:
            Y0 = np.concatenate((Y0, np.zeros(n, m-my)), 0)

        # optimum rotation matrix of Y
        A = np.dot(X0.T, Y0)
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        V = Vt.T
        T = np.dot(V, U.T)

        if reflection != 'best':

            # does the current solution use a reflection?
            have_reflection = np.linalg.det(T) < 0

            # if that's not what was specified, force another reflection
            if reflection != have_reflection:
                V[:, -1] *= -1
                s[-1] *= -1
                T = np.dot(V, U.T)

        traceTA = s.sum()

        if scaling:

            # optimum scaling of Y
            b = traceTA * normX / normY

            # standarised distance between X and b*Y*T + c
            d = 1 - traceTA**2

            # transformed coords
            Z = normX*traceTA*np.dot(Y0, T) + muX

        else:
            b = 1
            d = 1 + ssY/ssX - 2 * traceTA * normY / normX
            Z = normY*np.dot(Y0, T) + muX

        # transformation matrix
        if my < m:
            T = T[:my, :]
        c = muX - b*np.dot(muY, T)

        # transformation values
        tform = {'rotation': T, 'scale': b, 'translation': c}

        return d, Z, tform

    def run_human(self, points_3d):
        """
        Updates the visualization of the human skeleton in 3D.

        Args:
            points_3d (list): List of 3D points representing the human skeleton.

        Returns:
            None
        """
        self.person_skeleton_cloud.points = o3d.utility.Vector3dVector(
            points_3d)
        self.person_lines.points = o3d.utility.Vector3dVector(points_3d)

        self.visualizer.update_geometry(self.person_skeleton_cloud)
        self.visualizer.update_geometry(self.person_lines)

    def run_robot(self, points_3d):
        """
        Update the robot's skeleton cloud and lines with the given 3D points.

        Args:
            points_3d (List[List[float]]): The 3D points representing the robot's skeleton.

        Returns:
            None
        """
        self.robot_skeleton_cloud.points = o3d.utility.Vector3dVector(
            points_3d)
        self.robot_lines.points = o3d.utility.Vector3dVector(points_3d)

        self.visualizer.update_geometry(self.robot_skeleton_cloud)
        self.visualizer.update_geometry(self.robot_lines)
