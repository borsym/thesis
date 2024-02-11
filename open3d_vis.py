import open3d as o3d
import numpy as np

# pcd = o3d.io.read_point_cloud("beanbag_full_mesh_updated.ply")

# def pick_points(pcd):
#    vis = o3d.visualization.VisualizerWithEditing()
#    vis.create_window()
#    vis.add_geometry(pcd)
#    vis.run() # user picks points
#    vis.destroy_window()
#    return vis.get_picked_points()
# # shift + click kell
# # Get the indices of the picked points
# picked_points = pick_points(pcd)

# # Print the coordinates of the picked points
# for point_index in picked_points:
#    point_coordinates = np.asarray(pcd.points)[point_index]
#    print(point_coordinates)

# visualizer = o3d.visualization.Visualizer()
# visualizer.create_window()
