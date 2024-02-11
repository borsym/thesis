import open3d as o3d
import numpy as np

def get_pose_xyz(frame):
    pose_set = []
    for x,y,z in zip(frame[0],frame[1],frame[2]):
        pose_set.append(np.asarray([x,y,z]))
    return np.asarray(pose_set)

def create_cylinder_between_points(p1, p2, radius=0.01, resolution=20, color=[0, 0, 1]):
    # Compute the length and direction of the cylinder
    direction = p2 - p1
    length = np.linalg.norm(direction)
    direction = direction / length

    # Create a mesh cylinder
    mesh_frame = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length, resolution=resolution)
    mesh_frame.paint_uniform_color(color)

    # Move it to the right place
    transform = np.eye(4)
    transform[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz([0, np.arccos(direction[2]), np.arctan2(direction[1], direction[0])])
    transform[:3, 3] = p1
    mesh_frame.transform(transform)

    return mesh_frame

# Camera positions in the mesh
cam_stand = np.asarray([-1.01197171,  0.1033909,  -5.64958239]) #allvany (-1.0, 0.1, -5.6)
cam_corner = np.asarray([-5.25061131, -0.15126795,  0.53622693]) #sarok (-5.3, -0.15, 0.54)
cam_door = np.asarray([1.1673286,  -0.07970255,  5.54580784]) # ajto (1.2, -0.08, 5.5)
box_edge = np.asarray([0.75468755, -1.41470146, -0.69726562])

# Load your .PLY file
mesh = o3d.io.read_triangle_mesh("meshes/dfki-2_full_mesh_aligned.ply")

# wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
# video_poses = np.load("points_3d_list.npy")
video_poses = np.load("demo_points3d_list_dfki.npy")
if not mesh.has_vertex_colors():
    mesh.vertex_colors = o3d.utility.Vector3dVector([[0.5, 0.5, 0.5] for _ in range(len(mesh.vertices))])
if not mesh.has_vertex_normals():
    mesh.compute_vertex_normals()

# Calculate the center of mass of the point cloud
center_of_mass = np.mean(mesh.vertices, axis=0)

# Create a spinning cube at the center of mass
cube = o3d.geometry.TriangleMesh.create_box(width=0.5, height=0.5, depth=0.5)
cube.paint_uniform_color([0, 0, 1])  # Red color for the cube

# Create a rotation transformation for spinning
rot_degrees_per_sec = 30  # Rotation speed in degrees per second
rot_degrees_per_frame = rot_degrees_per_sec / 30  # Assuming 30 frames per second
rot_matrix = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, np.radians(rot_degrees_per_frame)))

skeleton_cloud = o3d.geometry.PointCloud()
# skeleton_cloud.points = o3d.utility.Vector3dVector(get_pose_xyz(video_poses[0]))

# Optional: Customize the appearance of the keypoints
keypoint_color = [0, 1, 0]  # Red color for keypoints
skeleton_cloud.paint_uniform_color(keypoint_color)

# Create a visualization window
# vis = o3d.visualization.Visualizer()
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window()

# Add the mesh, point cloud, and spinning cube to the visualizer
vis.add_geometry(mesh)
# vis.add_geometry(wireframe)
# vis.add_geometry(cube)
vis.add_geometry(skeleton_cloud)


base_trans_vector = np.asarray([0., 0., 0.])
base_R = np.eye(3)
R_matrices = []
angle = 1  # Rotation angle in degrees
angle_x = np.asarray([0])
angle_y = np.asarray([0])
angle_z = np.asarray([0])
frame_id = np.asarray([0])
scale_size = np.asarray([1.0])

pose = get_pose_xyz(video_poses[frame_id[0]])
skeleton_cloud.points = o3d.utility.Vector3dVector(pose)

connections = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
               [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]]

lines = o3d.geometry.LineSet()
lines.points = skeleton_cloud.points

sphere_list = []
for point_idx, point in enumerate(range(17)):
    print(point_idx)
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)  # Adjust radius for size
    sphere.translate([0,0,0])
    sphere.paint_uniform_color([1,0,0])
    sphere_list.append(sphere)
    vis.add_geometry(sphere)

lines.lines = o3d.utility.Vector2iVector(connections)
lines.paint_uniform_color([0,1,0])
vis.add_geometry(lines)

cylinder_list = np.asarray([])

#mesh init

# scale_size: [1.]
# frame_id: [183]
# angle_x: [-188]
# angle_y: [183]
# angle_z: [-10]
# translation: [-1.5 -0.6  7.7]
# R: [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]



def translate_point_cloud(pcd, translation_vector):
    """ Translate the point cloud by a given vector. """
    pcd.vertices = o3d.utility.Vector3dVector(np.asarray(pcd.vertices) + translation_vector)
    #base_vector += translation_vector
    return pcd

def get_center_of_mass(pcd):
    """ Calculate the center of mass of the point cloud. """
    return np.mean(np.asarray(pcd.vertices), axis=0)

def rotate_point_cloud(pcd, axis, angle):
    """ Rotate the point cloud around a given axis by an angle. """
    cos_angle = np.cos(np.radians(angle))
    sin_angle = np.sin(np.radians(angle))
    R = np.eye(3)
    if axis == 'x':
        R[1, 1], R[1, 2], R[2, 1], R[2, 2] = cos_angle, -sin_angle, sin_angle, cos_angle
    elif axis == 'y':
        # angle_y += angle
        R[0, 0], R[0, 2], R[2, 0], R[2, 2] = cos_angle, sin_angle, -sin_angle, cos_angle
    elif axis == 'z':
        # angle_z += angle
        R[0, 0], R[0, 1], R[1, 0], R[1, 1] = cos_angle, -sin_angle, sin_angle, cos_angle

    com = get_center_of_mass(pcd)
    # print(com)
    R_matrices.append(R)
    # Translate to origin, apply rotation, and translate back
    translated_points = np.asarray(pcd.vertices) - com
    rotated_points = np.dot(translated_points, R)
    pcd.vertices = o3d.utility.Vector3dVector(rotated_points + com)

def update_sphere_position(sphere, new_center):
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

def keyboard_callback2(vis, pcd, action, mods, angle_x, angle_y, angle_z):
        if action == 326:  # Numpad 6: Rotate around Y-axis (positive direction)
            rotate_point_cloud(pcd, 'y', angle)
            angle_y += angle
        elif action == 321:  # Numpad 4: Rotate around Y-axis (negative direction)
            rotate_point_cloud(pcd, 'y', -angle)
            angle_y -= angle
        elif action == 322:  # Numpad 2: Rotate around X-axis (positive direction)
            rotate_point_cloud(pcd, 'x', angle)
            angle_x += angle
        elif action == 323:  # Numpad 8: Rotate around X-axis (negative direction)
            rotate_point_cloud(pcd, 'x', -angle)
            angle_x -= angle
        elif action == 324:  # Numpad 7: Rotate around Z-axis (positive direction)
            rotate_point_cloud(pcd, 'z', angle)
            angle_z += angle
        elif action == 325:  # Numpad 9: Rotate around Z-axis (negative direction)
            rotate_point_cloud(pcd, 'z', -angle)
            angle_z -= angle

        vis.update_geometry(pcd)

def keyboard_callback(vis, action, mods, pcd, base_trans_vector, id, scale_size, cylinder_list):
    """ Keyboard callback function for Open3D visualizer. """
    step_size = 0.1  # Translation step size
    if action == ord('W'):
        translate_point_cloud(pcd, [0, step_size, 0])  # Move up
        base_trans_vector[1] += step_size
    elif action == ord('S'):
        translate_point_cloud(pcd, [0, -step_size, 0]) # Move down
        base_trans_vector[1] -= step_size
    elif action == ord('A'):
        translate_point_cloud(pcd, [-step_size, 0, 0]) # Move left
        base_trans_vector[0] -= step_size
    elif action == ord('D'):
        translate_point_cloud(pcd, [step_size, 0, 0])  # Move right
        base_trans_vector[0] += step_size
    elif action == ord('R'):
        translate_point_cloud(pcd, [0, 0, step_size]) # Move Forward
        base_trans_vector[2] += step_size
    elif action == ord('T'):
        translate_point_cloud(pcd, [0, 0, -step_size])  # Move Backward
        base_trans_vector[2] -= step_size
    elif action == ord('B'):
        print(f'scale_size: {scale_size}')
        print(f'frame_id: {id}')
        print(f'angle_x: {angle_x}')
        print(f'angle_y: {angle_y}')
        print(f'angle_z: {angle_z}')
        print(f'translation: {base_trans_vector}')
        R_matrices2 = np.asarray(R_matrices)
        reversed_arr = R_matrices2[::-1]
        R_base = np.eye(3)
        for R_step in reversed_arr:
            R_base = R_base @ R_step
        print(f'R: {R_base}')
        o3d.io.write_triangle_mesh("meshes/dfki-2_full_mesh_aligned.ply", pcd)
    elif action == ord('O'):
        id += 1
        print(id)
        pose = video_poses[id[0]]
        pcd.points = o3d.utility.Vector3dVector(pose)
        lines.points = pcd.points

        for point_idx, point in enumerate(range(17)):
            if point_idx > len(pcd.points):
                # temporarly remove sphere visibility
                sphere = sphere_list[point_idx]
                update_sphere_position(sphere, np.array([0,0,0]))
            else:
                sphere = sphere_list[point_idx]
                # sphere.translate(pcd.points[point_idx])
                update_sphere_position(sphere, np.array(pcd.points[point_idx]))
            vis.update_geometry(sphere)
        # print(f'cylinder list: {cylinder_list}')
        # for cyl_idx, cylinder in enumerate(cylinder_list):
        #     print(f'removing cylinder {cyl_idx} at frame {frame_id}')
        #     vis.remove_geometry(cylinder, reset_bounding_box=False)
        # for connection in connections:
        #     try:
        #         cylinder = create_cylinder_between_points(pcd.points[connection[0]],
        #                                                   pcd.points[connection[1]])
        #         cylinder_list.append(cylinder)
        #         vis.add_geometry(cylinder)
        #     except:
        #         s = 'a'
        # rotate_point_cloud(pcd, 'x', angle_x)
        # rotate_point_cloud(pcd, 'y', angle_y)
        # rotate_point_cloud(pcd, 'z', angle_z)
        # translate_point_cloud(pcd, base_trans_vector)

    elif action == ord('L'):
        id -= 1
        pose = get_pose_xyz(video_poses[id[0]])
        pose = video_poses[id[0]]
        pcd.points = o3d.utility.Vector3dVector(pose)
        lines.points = pcd.points

        for point_idx, point in enumerate(range(17)):
            if point_idx > len(pcd.points):
                # temporarly remove sphere visibility
                sphere = sphere_list[point_idx]
                update_sphere_position(sphere, np.array([0,0,0]))
            else:
                sphere = sphere_list[point_idx]
                # sphere.translate(pcd.points[point_idx])
                update_sphere_position(sphere, np.array(pcd.points[point_idx]))
            vis.update_geometry(sphere)

        # for cylinder in cylinder_list:
        #     vis.remove_geometry(cylinder)
        # cylinder_list = []
        # for connection in connections:
        #     try:
        #         cylinder = create_cylinder_between_points(pcd.points[connection[0]],
        #                                                   pcd.points[connection[1]])
        #         vis.add_geometry(cylinder)
        #     except:
        #         pass
        # rotate_point_cloud(pcd, 'x', angle_x)
        # rotate_point_cloud(pcd, 'y', angle_y)
        # rotate_point_cloud(pcd, 'z', angle_z)
        # translate_point_cloud(pcd, base_trans_vector)

    elif action == ord('M'):
        scaling_factor = 1.01
        scaling_matrix = np.diag([scaling_factor, scaling_factor, scaling_factor, 1])
        pcd.scale(scale=1.0, center=mesh.get_center())
        pcd.transform(scaling_matrix)

        scale_size *= 0.01
        # pose = get_pose_xyz(video_poses[id[0]])
        # pose *= scale_size
        # pcd.points = o3d.utility.Vector3dVector(pose)
        # translate_point_cloud(pcd, base_trans_vector)
        # translate_point_cloud(pcd, [0, 0, step_size]) # Move Forward
        # base_trans_vector[2] += step_size
    elif action == ord('N'):
        scaling_factor = 0.99
        scaling_matrix = np.diag([scaling_factor, scaling_factor, scaling_factor, 1])
        pcd.scale(scale=1.0, center=mesh.get_center())
        pcd.transform(scaling_matrix)
        scale_size *= 0.99
        # pose = get_pose_xyz(video_poses[id[0]])
        # pose *= scale_size
        # pcd.points = o3d.utility.Vector3dVector(pose)
        # translate_point_cloud(pcd, base_trans_vector)
        # translate_point_cloud(pcd, [0, 0, -step_size])  # Move Backward
        # base_trans_vector[2] -= step_size
    vis.update_geometry(pcd)
    vis.update_geometry(lines)
    return False


vis.register_key_callback(87, lambda vis: keyboard_callback(vis, ord('W'), 0, mesh, base_trans_vector, frame_id, scale_size, cylinder_list))  # Key 'W'
vis.register_key_callback(65, lambda vis: keyboard_callback(vis, ord('A'), 0, mesh, base_trans_vector, frame_id, scale_size, cylinder_list))  # Key 'A'
vis.register_key_callback(68, lambda vis: keyboard_callback(vis, ord('D'), 0, mesh, base_trans_vector, frame_id, scale_size, cylinder_list))  # Key 'D'
vis.register_key_callback(83, lambda vis: keyboard_callback(vis, ord('S'), 0, mesh, base_trans_vector, frame_id, scale_size, cylinder_list))  # Key 'S'
vis.register_key_callback(82, lambda vis: keyboard_callback(vis, ord('R'), 0, mesh, base_trans_vector, frame_id, scale_size, cylinder_list))  # Key 'R'
vis.register_key_callback(84, lambda vis: keyboard_callback(vis, ord('T'), 0, mesh, base_trans_vector, frame_id, scale_size, cylinder_list))  # Key 'T'
vis.register_key_callback(79, lambda vis: keyboard_callback(vis, ord('O'), 0, skeleton_cloud, base_trans_vector, frame_id, scale_size, cylinder_list))  # Key 'O' frame +
vis.register_key_callback(76, lambda vis: keyboard_callback(vis, ord('L'), 0, skeleton_cloud, base_trans_vector, frame_id, scale_size, cylinder_list))  # Key 'L' frame -
vis.register_key_callback(77, lambda vis: keyboard_callback(vis, ord('M'), 0, mesh, base_trans_vector, frame_id, scale_size, cylinder_list))  # Key 'M' scale +
vis.register_key_callback(78, lambda vis: keyboard_callback(vis, ord('N'), 0, mesh, base_trans_vector, frame_id, scale_size, cylinder_list))  # Key 'N' scale -
vis.register_key_callback(66, lambda vis: keyboard_callback(vis, ord('B'), 0, mesh, base_trans_vector, frame_id, scale_size, cylinder_list))  # Key 'B'

vis.register_key_callback(321, lambda vis: keyboard_callback2(vis, mesh, 321, 0, angle_x, angle_y, angle_z))  # Numpad 4
vis.register_key_callback(322, lambda vis: keyboard_callback2(vis, mesh,322, 0, angle_x, angle_y, angle_z))  # Numpad 2
vis.register_key_callback(323, lambda vis: keyboard_callback2(vis, mesh,323, 0, angle_x, angle_y, angle_z))  # Numpad 8
vis.register_key_callback(324, lambda vis: keyboard_callback2(vis, mesh,324, 0, angle_x, angle_y, angle_z))  # Numpad 7
vis.register_key_callback(325, lambda vis: keyboard_callback2(vis, mesh,325, 0, angle_x, angle_y, angle_z))  # Numpad 9
vis.register_key_callback(326, lambda vis: keyboard_callback2(vis, mesh,326, 0, angle_x, angle_y, angle_z))  # Numpad 6

# Set the initial position of the cube at the center of mass
# cube.vertices = cube.vertices + center_of_mass



# Get the rendering options and disable shading
render_option = vis.get_render_option()
render_option.mesh_show_back_face = True
render_option.light_on = False
# Run the visualizer

target_R = np.asarray([[-0.13840089,  0.28613705, -0.9481407 ],
 [ 0.04809612, -0.95428253, -0.2950112 ],
 [-0.98920774, -0.0864317,   0.11831147]]
)
target_scale = 0.41708936158237075
target_translation = np.asarray([0.91201383, -1.74475789, -0.90505653])

base_rot_mx = np.eye(3)
# translation: [-1.5 -0.4  7.7]
# base_trans_vector = np.asarray([-0.1, -0.8,  9.1 ])
# base_rot_mx = np.asarray([[-0.13840089,  0.28613705, -0.9481407 ],
#  [ 0.04809612, -0.95428253, -0.2950112 ],
#  [-0.98920774, -0.0864317,   0.11831147]])
#
# new_rot_mx = np.asarray([[-0.91617806,  0.1599969,  -0.36744899],
#  [ 0.28389457,  0.90624591, -0.31324466],
#  [ 0.28288097, -0.39130466, -0.87570487]])

translate_point_cloud(mesh, base_trans_vector)
com = get_center_of_mass(mesh)
translated_points = np.asarray(mesh.vertices) - com
_rotated_points = np.dot(translated_points, base_rot_mx)
mesh.vertices = o3d.utility.Vector3dVector(_rotated_points + com)
vis.update_geometry(mesh)

vis.run()

# i=0
# while True:
#     # cube.vertices = np.dot(cube.vertices,
#     #                        rot_matrix.T)  # Update cube's transformation
#     # cube.rotate(rot_matrix.T, center=(0, 0, 0))
#     pose = get_pose_xyz(video_poses[i])
#     pose = pose @ target_R
#     pose *= target_scale
#     pose += target_translation
#     # print(pose)
#     # skeleton_cloud.points = o3d.utility.Vector3dVector(pose)
#     # vis.update_geometry(skeleton_cloud)  # Update skeleton's transformation
#     # vis.update_geometry(cube)  # Update cube's transformation
#     vis.poll_events()
#     vis.update_renderer()
#     i+=1
#     i = i % 1000
