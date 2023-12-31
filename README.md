# Camera Calibration and 3D Reconstruction

This repository contains a Python script for camera calibration and 3D reconstruction using OpenCV and Open3D libraries. The script takes a pair of images, performs camera calibration using chessboard corners, matches features between the images, and reconstructs the 3D scene.

https://github.com/argrabowski/3d-reconstruction/assets/64287065/54e03f55-40de-4f7d-87a1-cdf5c9f5d09b

## Requirements

- Python 3.10
- OpenCV
- Open3D

Install the required libraries:

```bash
pip install opencv-python
pip install open3d
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/your-username/3d-reconstruction.git
cd 3d-reconstruction
```

2. Run the Python script:

```bash
python sfm.py
```

Make sure to customize the input images and directories as needed within the script.

## Script Overview

1. **Read Images:**
   - Reads two input images (`scene1.jpg` and `scene2.jpg`).
   - Resizes the images to 20% of their original size.

2. **Chessboard Corners:**
   - Reads a set of chessboard images from the `boards` directory.
   - Finds chessboard corners in each image, refines them using subpixel refinement, and displays the corners.

3. **Camera Calibration:**
   - Calibrates the camera using the object points (chessboard corners) and corresponding image points.
   - Saves the calibration matrix to a NumPy file (`calib_mtx.npy`).
   - Undistorts the original images using the calibrated camera.

4. **Feature Matching:**
   - Uses the SIFT detector to find keypoints and descriptors in the undistorted images.
   - Applies the FLANN-based matcher to find good matches between keypoints.

5. **Fundamental Matrix Calculation:**
   - Calculates the fundamental matrix using the RANSAC algorithm.
   - Displays the keypoints and matches between the two images.

6. **Essential Matrix Calculation:**
   - Computes the essential matrix using the calibration matrix and the fundamental matrix.

7. **Decomposition of Essential Matrix:**
   - Uses singular value decomposition (SVD) to decompose the essential matrix into rotation and translation matrices.

8. **Projection Matrix Calculation:**
   - Calculates the projection matrices for both cameras based on the decomposed rotation and translation matrices.
   - Considers four possible orientations for the second camera.

9. **Triangulation:**
   - Triangulates 3D points using linear least-squares triangulation.
   - Chooses the best projection matrix based on points in front of the cameras.

10. **Reprojection Error Calculation:**
    - Computes the reprojection error for both cameras.

11. **Save Point Cloud to PCD File:**
    - Saves the triangulated 3D points and their colors to a Point Cloud Data (PCD) file (`tri_pnts.pcd`).

12. **Visualize Point Cloud:**
    - Reads the PCD file and visualizes the 3D point cloud from front and top views.
