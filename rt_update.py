import open3d as o3d
import numpy as np

def get_pose_xyz(frame):
    pose_set = []
    for x,y,z in zip(frame[0],frame[1],frame[2]):
        pose_set.append(np.asarray([x,y,z])/1000)
    return np.asarray(pose_set)

# Camera positions in the mesh
cam_stand = np.asarray([-1.01197171,  0.1033909,  -5.64958239]) #allvany (-1.0, 0.1, -5.6)
cam_corner = np.asarray([-5.25061131, -0.15126795,  0.53622693]) #sarok (-5.3, -0.15, 0.54)
cam_door = np.asarray([1.1673286,  -0.07970255,  5.54580784]) # ajto (1.2, -0.08, 5.5)
box_edge = np.asarray([0.75468755, -1.41470146, -0.69726562])

# Load your .PLY file
mesh = o3d.io.read_triangle_mesh("beanbag_full_mesh_updated.ply")

print(mesh.vertices[360379])
# wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
video_poses = np.load("points_3d_list.npy")
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
vis = o3d.visualization.Visualizer()
vis.create_window()

# Add the mesh, point cloud, and spinning cube to the visualizer
vis.add_geometry(mesh)
# vis.add_geometry(wireframe)
# vis.add_geometry(cube)
vis.add_geometry(skeleton_cloud)

# Set the initial position of the cube at the center of mass
# cube.vertices = cube.vertices + center_of_mass



# Get the rendering options and disable shading
render_option = vis.get_render_option()
render_option.mesh_show_back_face = True
render_option.light_on = False
# Run the visualizer

target_R = np.asarray([[-0.93535974,  0.00259579, -0.35368832],
 [0.02792994, -0.99630843, -0.0811753 ],
 [-0.35259337, -0.0858066,   0.93183429]])
target_scale = 0.41708936158237075
target_translation = np.asarray([0.91201383, -1.74475789, -0.90505653])

i=0
while True:
    # cube.vertices = np.dot(cube.vertices,
    #                        rot_matrix.T)  # Update cube's transformation
    cube.rotate(rot_matrix.T, center=(0, 0, 0))
    pose = get_pose_xyz(video_poses[i])
    # pose = pose @ target_R
    # pose *= target_scale
    # pose += target_translation
    # print(pose)
    skeleton_cloud.points = o3d.utility.Vector3dVector(pose)
    vis.update_geometry(skeleton_cloud)  # Update skeleton's transformation
    vis.update_geometry(cube)  # Update cube's transformation
    vis.poll_events()
    vis.update_renderer()
    i+=1

