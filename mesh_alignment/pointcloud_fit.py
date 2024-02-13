import cv2
import numpy as np
import open3d as o3d

# Intrinsic parameters for camera 0
cam0_K = np.array([
    [1143.5629058671955, 0.0, 897.4701399737154],
    [0.0, 1146.6229630039772, 571.4789156478826],
    [0.0, 0.0, 1.0]
])
cam0_dist = np.array([-0.20212810339004855, 0.016856910836021612, -
                     0.00038445457626206384, 0.0029733591935219177, 0.028772663897866188])

# Intrinsic parameters for camera 1
cam1_K = np.array([
    [1390.5311354774792, 0.0, 941.6657612837951],
    [0.0, 1396.1775075864095, 564.5650596969682],
    [0.0, 0.0, 1.0]
])
cam1_dist = np.array([-0.37112152866840364, 0.3342635457860893,
                     0.00089837167379825, -0.00047573092618939504, -0.2159938879592358])

# Extrinsic parameters (assuming these are for camera 0 or the transformation from world to camera 0)
R = np.array([
    [0.07541070458878951, -0.16218477631596587, 0.9838746485019128],
    [0.12432429135836522, 0.980514408874011, 0.1521018230288162],
    [-0.9893718695271574, 0.11084941281661857, 0.09410478981724704]
])
T = np.array([
    [-3221.352499290285],
    [-239.17093718886326],
    [5148.128536539155]
])


def transform_points(points, R, T):
    # Apply rotation and translation
    transformed_points = np.dot(points, R.T) + T.T
    return transformed_points


def project_points_to_image(points, camera_matrix, dist_coeffs):
    # Assuming points are in the shape (N, 3)
    # Convert points from 3D to 2D
    points_2d, _ = cv2.projectPoints(
        points.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs)
    return points_2d.reshape(-1, 2)


# Example: Load a pointcloud and transform it
# Update the path to your pointcloud file
pcd = o3d.io.read_point_cloud("D:/thesis/realtime_update/meshes/save4.ply")
points = np.asarray(pcd.points)

# Transform the pointcloud points to the camera's coordinate system
transformed_points = transform_points(points, R, T)

# Project the transformed points onto the image plane of camera 0
points_2d = project_points_to_image(transformed_points, cam0_K, cam0_dist)

# Load the RGB image captured by camera 0
# Update the path to your image file
image = cv2.imread(
    "D:/thesis/realtime_update/recordings/Scenario1/Cam2/image1796.bmp")

# Overlay points on the image
for x, y in np.int32(points_2d):
    cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

# Display the result
cv2.imshow("Overlayed Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
