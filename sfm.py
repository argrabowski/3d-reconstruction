import numpy as np
import cv2 as cv
import open3d as o3d
import glob

def read_images():
    # Read first image and scale down to 20%
    img1 = cv.imread('scene1.jpg')
    img1 = cv.resize(img1, (0, 0), fx=0.2, fy=0.2)

    # Read second image and scale down to 20%
    img2 = cv.imread('scene2.jpg')
    img2 = cv.resize(img2, (0, 0), fx=0.2, fy=0.2)

    return img1, img2

def chessboard_corners(board_size):
    # Get list of file paths for images
    images = glob.glob('boards/*.jpg')

    # Create grid of 3D points for chessboard corners
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)

    # Lists to store 3D object points and 2D image points
    obj_pnts = []
    img_pnts = []

    # Criteria for subpixel refinement
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Iterate through images to find chessboard corners
    for fname in images:
        # Read and resize board image
        img = cv.imread(fname)
        img = cv.resize(img, (0, 0), fx=0.2, fy=0.2)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find chessboard corners in resized image
        ret, corners = cv.findChessboardCorners(gray, board_size, None)

        # If corners found
        if ret:
            # Refine and store corners in lists
            obj_pnts.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            img_pnts.append(corners2)

            # Draw chessboard corners on image and display
            cv.drawChessboardCorners(img, board_size, corners2, ret)
            cv.imshow('boards', img)
            cv.waitKey(100)

    return obj_pnts, img_pnts, gray.shape[::-1]

def calibrate_camera(obj_pnts, img_pnts, img_shape, img1, img2):
    # Calibrate camera using object and image points
    _, mtx, dist, _, _ = cv.calibrateCamera(obj_pnts, img_pnts, img_shape, None, None)

    # Save calibration matrix to numpy file
    np.save('calib_mtx.npy', mtx)

    # Undistort first and second images
    img1_ud = undistort_image(img1, mtx, dist)
    img2_ud = undistort_image(img2, mtx, dist)

    return mtx, img1_ud, img2_ud

def undistort_image(img, mtx, dist):
    # Get optimal new camera matrix and region of interest
    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # Initialize undistortion mapping
    mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)

    # Remap distorted image to undistorted image
    dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

    # Crop image based on ROI
    x, y, w, h = roi
    img_ud = dst[y:y+h, x:x+w]

    return img_ud

def find_matches(img1, img2):
    # Create SIFT detector
    sift = cv.SIFT_create()

    # Create FLANN-based matcher
    params = dict(algorithm=1, trees=5)
    flann = cv.FlannBasedMatcher(params, {})

    # Detect and compute keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Use FLANN matcher to find k-nearest matches
    matches = flann.knnMatch(des1, des2, k=2)

    # Select good matches based on ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    # Check enough good matches
    if len(good) < 8:
        raise Exception('Not enough good matches!')

    # Get corresponding points from good matches
    src1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst1 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Use RANSAC to find inliers and compute homography
    mask = cv.findHomography(src1, dst1, cv.RANSAC, 100.0)[1]

    # Filter inliers
    inliers = [good[i] for i in range(len(mask)) if mask[i] == 1]

    # Get corresponding points from inliers
    src2 = np.float32([kp1[m.queryIdx].pt for m in inliers]).reshape(-1, 1, 2)
    dst2 = np.float32([kp2[m.trainIdx].pt for m in inliers]).reshape(-1, 1, 2)

    # Draw keypoints and matches for visualization
    img1_sift = cv.drawKeypoints(img1, kp1, None, (255, 0, 0), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img2_sift = cv.drawKeypoints(img2, kp2, None, (0, 0, 255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    matched = cv.drawMatches(img1, kp1, img2, kp2, inliers, None, flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    # Display images and matches
    cv.imshow('first', img1_sift)
    cv.imshow('second', img2_sift)
    cv.imshow('matches', matched)

    # Wait for key event and close windows
    cv.waitKey(0)
    cv.destroyAllWindows()

    return src2, dst2, inliers

def fundamental_mat(src, dst, matches):
    # Compute fundamental matrix using 8-point algorithm
    F, _ = cv.findFundamentalMat(src, dst, cv.FM_8POINT)
    print(f'Fundamental Matrix (F):\n{F}')

    # Print epipolar line equations for each match
    for i in range(len(matches)):
        # Augmenting with homogenous coordinates
        pt1 = np.append(src[i], 1)
        pt2 = np.append(dst[i], 1)

        # Compute epipolar line equation for current match
        equation_result = np.matmul(np.matmul(pt2.T, F), pt1)
        print(f'{pt2}^T * F * {pt1} = {equation_result}')

    return F

def essential_mat(K, F):
    # Compute essential matrix using relation
    E = np.matmul(np.transpose(K), np.matmul(F, K))
    print(f'Essential Matrix (E):\n{E}')

    # Calculate determinant of essential matrix
    det_E = np.linalg.det(E)
    print(f'Determinant (E): {det_E}')

    return E

def rotation_translation_mats(E):
    # Singular value decomposition on essential matrix
    U, S, Vt = np.linalg.svd(E)

    # Ensure determinant of U and Vt positive
    if np.linalg.det(U) < 0:
        U[:, 2] *= -1
    if np.linalg.det(Vt) < 0:
        Vt[2, :] *= -1

    # Construct skew-symmetric matrix
    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])

    # Compute rotation matrix
    R = np.dot(U, np.dot(W, Vt))

    # Extract and normalize translation vector
    T = U[:, 2]
    T /= np.linalg.norm(T)

    # Print computed rotation matrix and translation vector
    print(f'Rotation Matrix (R):\n{R}')
    print(f'Translation Vector (T):\n{T}')

    return R, T

def projection_mats(K, R, T):
    # Create projection matrix for first camera
    P0 = np.hstack((K, np.zeros((3, 1))))

    # Initialize list to store projection matrices
    P1_list = []

    # Iterate through four possible orientations
    for i in range(4):
        # Create extrinsic matrix
        M = np.hstack((R, T.reshape((3, 1))))

        # Apply flips for different orientations
        if i > 0:
            M = np.dot(np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]), M)

        # Compute projection matrix for second camera
        P1 = np.dot(K, M)

        # Append computed projection matrix to list
        P1_list.append(P1)

    # Print computed projection matrices for both cameras
    print(f'Projection Matrix (P0):\n{P0}')
    for i in range(4):
        print(f'Projection Matrix (P1{i+1}):\n{P1_list[i]}')

    return P0, P1_list

def linear_ls_triangulation(u0, P0, u1, P1):
    # Homogeneous coordinates for image points
    u0_h = np.append(u0, 1)
    u1_h = np.append(u1, 1)

    # Construct linear system for triangulation
    A = np.vstack((u0_h[0] * P0[2, :] - P0[0, :],
                   u0_h[1] * P0[2, :] - P0[1, :],
                   u1_h[0] * P1[2, :] - P1[0, :],
                   u1_h[1] * P1[2, :] - P1[1, :]))

    # Perform SVD on linear system
    _, _, Vt = np.linalg.svd(A)

    # Extract homogeneous 3D point coordinates
    X_h = Vt[-1, :]

    # Normalize homogeneous coordinates
    X_h /= X_h[-1]

    # Return 3D point coordinates in Euclidean space
    X = X_h[:-1]

    return X

def triangulate_points(src, dst, P0, P1, print_pnts=True):
    # List to store triangulated 3D points
    tri_pnts = []

    # Iterate over image points
    for i in range(len(src)):
        # Extract 2D image points from source and destination
        u0 = src[i][0]
        u1 = dst[i][0]

        # Triangulate 3D point using linear least-squares triangulation
        X_tri = linear_ls_triangulation(u0, P0, u1, P1)
        tri_pnts.append(X_tri)

        # Print 3D points if requested
        if print_pnts:
            print(f'3D Point {i+1}: {X_tri}')

    return tri_pnts

def choose_projection_mat(src, dst, P0, P1_list):
    # Initialize selected matrix index
    selected_matrix_index = None

    # Iterate over each projection matrix
    for i in range(len(P1_list)):
        # Triangulate 3D points using current matrix
        tri_pnts = triangulate_points(src, dst, P0, P1_list[i], False)

        # Count number of points in front of camera
        front_cnt = sum(1 for point in tri_pnts if point[2] < 0)
        print(f'Front Points (P1{i+1}): {front_cnt}/{len(tri_pnts)}')

        # Choose if all points in front and no matrix selected
        if front_cnt == len(tri_pnts) and selected_matrix_index is None:
            selected_matrix_index = i

    return P1_list[i]

def reprojection_error(src, tri_pnts, K, P0, P1):
    # Initialize total errors for cameras
    tot_err_c0 = 0
    tot_err_c1 = 0
    count = 0

    # Iterate over each correspondence of 2D and 3D points
    for i in range(len(src)):
        # Extract 2D coordinates from source and 3D coordinates
        u0 = src[i][0]
        u1 = tri_pnts[i][:2]

        # Project 3D point to 2D using both cameras
        reproj_c0, _ = cv.projectPoints(np.array([tri_pnts[i]]), P0[:, :3], P0[:, 3:], K, None)
        reproj_c1, _ = cv.projectPoints(np.array([tri_pnts[i]]), P1[:, :3], P1[:, 3:], K, None)

        # Calculate reprojection error for both cameras
        err_c0 = np.linalg.norm(u0 - reproj_c0[0][0])
        err_c1 = np.linalg.norm(u1 - reproj_c1[0][0])

        # Accumulate errors and increment count
        tot_err_c0 += err_c0
        tot_err_c1 += err_c1
        count += 1

    # Calculate average errors
    avg_err_c0 = tot_err_c0 / count
    avg_err_c1 = tot_err_c1 / count

    # Print average reprojection errors for both cameras
    print(f'Reprojection Error (C0): {avg_err_c0}')
    print(f'Reprojection Error (C1): {avg_err_c1}')

def save_pcd_file(src, dst, img1, img2, tri_pnts):
    # Create PointCloud object
    pnt_cloud = o3d.geometry.PointCloud()

    # Initialize list to store point colors
    colors = []

    # Iterate over each correspondence
    for i in range(len(src)):
        # Convert 2D coordinates to integer indices
        u0 = src[i][0].astype(int)
        u1 = dst[i][0].astype(int)

        # Extract color information from images around corresponding points
        color_src = np.mean(img1[u0[1] - 5:u0[1] + 5, u0[0] - 5:u0[0] + 5], axis=(0, 1))
        color_dst = np.mean(img2[u1[1] - 5:u1[1] + 5, u1[0] - 5:u1[0] + 5], axis=(0, 1))

        # Calculate average color
        avg_color = (color_src + color_dst) / 2
        colors.append(avg_color)

    # Set 3D points and colors in PointCloud object
    pnt_cloud.points = o3d.utility.Vector3dVector(np.array(tri_pnts))
    pnt_cloud.colors = o3d.utility.Vector3dVector(np.array(colors) / 255.0)

    # Write PointCloud to PCD file
    o3d.io.write_point_cloud('tri_pnts.pcd', pnt_cloud)

def visualize_point_cloud(pcd_path):
    # Read PointCloud from specified PCD file
    pnt_cloud = o3d.io.read_point_cloud(pcd_path)

    # Create visualizer for front view
    vis_front = o3d.visualization.Visualizer()
    vis_front.create_window(window_name='front view')

    # Add PointCloud geometry to visualizer
    vis_front.add_geometry(pnt_cloud)

    # Set view to show front perspective
    vis_front.get_view_control().set_front([1, 0, 0])

    # Run visualizer for front view
    vis_front.run()
    vis_front.destroy_window()

    # Create visualizer for top view
    vis_top = o3d.visualization.Visualizer()
    vis_top.create_window(window_name='top view')

    # Add PointCloud geometry to visualizer
    vis_top.add_geometry(pnt_cloud)

    # Set view to show top perspective
    vis_top.get_view_control().set_up([0, 1, 0])

    # Run visualizer for top view
    vis_top.run()
    vis_top.destroy_window()

def main():
    # Read input scenes
    img1, img2 = read_images()

    # Find chessboard corners in scenes
    obj_pnts, img_pnts, img_shape = chessboard_corners((8, 6))

    # Calibrate camera using chessboard corners
    K, img1_ud, img2_ud = calibrate_camera(obj_pnts, img_pnts, img_shape, img1, img2)

    # Find feature matches between undistorted scenes
    src, dst, matches = find_matches(img1_ud, img2_ud)

    # Calculate fundamental matrix
    F = fundamental_mat(src, dst, matches)

    # Calculate essential matrix
    E = essential_mat(K, F)

    # Decompose essential matrix into rotation and translation
    R, T = rotation_translation_mats(E)

    # Calculate projection matrices for both cameras
    P0, P1_list = projection_mats(K, R, T)

    # Choose best projection matrix
    P1_best = choose_projection_mat(src, dst, P0, P1_list)

    # Triangulate 3D points using linear least-squares triangulation
    tri_pnts = triangulate_points(src, dst, P0, P1_best)

    # Compute and print reprojection errors
    reprojection_error(src, tri_pnts, K, P0, P1_best)

    # Save triangulated 3D points to PCD file
    save_pcd_file(src, dst, img1_ud, img2_ud, tri_pnts)

    # Visualize point cloud from PCD file
    visualize_point_cloud('tri_pnts.pcd')

if __name__ == '__main__':
    main()
