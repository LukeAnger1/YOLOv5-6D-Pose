import sys, os
import cv2
import numpy as np
from data_curation import pose_utils
import json
import matplotlib.pyplot as plt
from PIL import Image
import math

def binary_threshold(frame, threshold=220, remove_ground=True, remove_circles=True):
    """
    Convert image to binary and remove ground and circular objects.
    
    Args:
        frame: Input image
        threshold: Brightness threshold for binarization
        remove_ground: Whether to remove the ground plane
        remove_circles: Whether to remove circular objects
    
    Returns:
        Processed binary image
    """
    # Convert to grayscale if needed
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    
    # Apply threshold to get initial binary image
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # Create a copy of the binary image for processing
    result = binary.copy()
    
    # Find connected components (blobs)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    # Process each connected component
    for i in range(1, num_labels):  # Skip label 0 (background)
        x, y, w, h, area = stats[i]
        
        # Remove ground-like components (typically large and at the bottom of image)
        if remove_ground and h < w * 0.5 and y > binary.shape[0] * 0.6:
            # Likely ground if it's wide and in the bottom part of the image
            result[labels == i] = 0
        
        # Remove circular objects
        if remove_circles:
            # Create a mask of just this component
            component_mask = (labels == i).astype(np.uint8) * 255
            
            # Find contours of the component
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:  # If contours were found
                contour = contours[0]
                
                # Calculate circularity: 4π × area / perimeter²
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    # If shape is circular (circularity close to 1)
                    if circularity > 0.7:  # Threshold for "circular enough"
                        result[labels == i] = 0
    
    return result

# path to 3D model
object_path = r"/home/rick/Desktop/nasa/yolo container/data/LINEMOD_updated_multi/LINEMOD_updated/LINEMOD/lander/lander.ply"
# path to camera intrinsics
camera_path = r"/home/rick/Desktop/nasa/yolo container/YOLOv5-6D-Pose/configs/linemod/linemod_camera.json"
# Path to folder containing your images and transforms
data_path = r"/media/rick/New Volume/nasa/training" 
# Where to store all the data
output_path = r"/home/rick/Desktop/nasa/yolo container/data/output"

"""
This is the original code snippet which will be adapted for our dataset

# 1. Run through video
cap = cv2.VideoCapture(data_path)
frame_count = 0

while(cap.isOpened()):

    ret, frame = cap.read()
    if not ret:
        break

    print(f"Frame {frame_count}")
    # 2. detect charuco pose
    frame_remapped_gray = frame[:, :, 0]
    im_height, im_width = frame_remapped_gray.shape
    # print(frame.shape)
    im_with_charuco_board, pose = pose_utils.detect_Charuco_pose_board(frame, mtx, dtx) # update based on your own charuco board layout
    if pose != None:
        
        # 3. determine offset from charuco pose to object pose (in charuco frame coordinates)
        rvec = pose['rvec']
        tvec = pose['tvec']
        rotation = cv2.Rodrigues(rvec)[0]
        transform_mat = np.vstack((np.hstack((rotation, tvec)), np.array([0, 0, 0, 1])))
        transform_mat = np.matmul(transform_mat, offset_mat)
        # 4. project object onto image
        projected_corners = pose_utils.compute_projection(corners3D, transform_mat[:3, :], mtx)
        projected_vertices = pose_utils.compute_projection(vertices, transform_mat[:3, :], mtx)
        # 5. Draw projected object
        im_with_charuco_board = pose_utils.draw_BBox(im_with_charuco_board, projected_corners.T, projected_vertices.T)
        # Create mask
        mask_arr = pose_utils.create_simple_mask(projected_vertices.T, im_width, im_height)
        # Create label
        label = pose_utils.create_label(0, projected_vertices, mtx[0,0],  mtx[1,1] , im_width, im_height, mtx[0,2], mtx[1,2], im_width, im_height, transform_mat)
        imageName = f"frame_{frame_count}.png"
        # 6. store all information
        pose_utils.save_data(frame, mask_arr, label, imageName, output_path, im_with_charuco_board )

    frame_count += 1

"""

# load 3D model
mesh = pose_utils.MeshPly(object_path)
vertices_og = np.c_[np.array(mesh.vertices)/1000, np.ones((len(mesh.vertices), 1))].transpose() # vertices in object coordinate in meters
corners3D = pose_utils.get_3D_corners(vertices_og)
vertices = np.hstack((np.array([0,0,0,1]).reshape(4,1), corners3D)) # add center coordinate

# load camera params
with open(camera_path, 'r') as f:
    camera_data = json.load(f)

dtx = np.array(camera_data["distortion"])
mtx = np.array(camera_data["intrinsic"])

# Predefined offset from charuco frame to object frame - might need adjustment or removal depending on your data
rotation_offset = [0.0, 0.0, 0.0]  # No rotation offset since we're using direct measurements
translation_offset = [0.0, 0.0, 0.0]  # No translation offset since we're using direct measurements

# offset_mat = pose_utils.construct_transform(translation_offset, rotation_offset)

# Function to convert quaternion to rotation matrix
def quaternion_to_rotation_matrix(quat):
    """
    Convert quaternion to rotation matrix
    Input: quat as [x, y, z, w] or [w, x, y, z] depending on your format
    Output: 3x3 rotation matrix
    """
    # Normalize quaternion
    quat = quat / np.linalg.norm(quat)
    
    # Assuming quat is in format [x, y, z, w] - adjust if yours is different
    x, y, z, w = quat
    
    # Conversion to rotation matrix
    rotation_matrix = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])
    
    return rotation_matrix

# Process each image and its corresponding transform file
frame_count = 0
for file in os.listdir(data_path):
    if file.startswith("right-missiontime_") and file.endswith(".png"):
        image_path = os.path.join(data_path, file)
        transform_file = f"transforms-{file[:-4]}.npy"
        transform_path = os.path.join(data_path, transform_file)
        
        # Check if transform file exists
        if not os.path.exists(transform_path):
            print(f"Transform file not found for {file}, skipping...")
            continue
            
        print(f"Processing {file} with transfrom {transform_file}")
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Failed to load image {file}, skipping...")
            continue
            
        # Get image dimensions
        im_height, im_width = frame.shape[:2]

        # Load transformation data
        transforms = np.load(transform_path)
        # Assuming the format is [x, y, z, qw, qx, qy, qz] or similar
        # Get the last/only row if multiple
        print(f'the transforms are {transforms}')
        transform = transforms[-1]  
        
        # Extract translation and quaternion
        translation = transform[:3]  # x, y, z
        quaternion = transform[3:]  # quaternion components
        
        # Convert quaternion to rotation matrix
        rotation = quaternion_to_rotation_matrix(quaternion)


        for pose in transforms:
            translation = pose[:3]
            quaternion = pose[3:]
            print("Translation:", translation)
            print("Quaternion:", quaternion)


        print(f"Translation: {translation}, Quaternion: {quaternion} and image height and width: {im_height}, {im_width}")

        # Create transformation matrix
        transform_mat = np.vstack((np.hstack((rotation, translation.reshape(3, 1))), np.array([0, 0, 0, 1])))
        
        # Project object onto image
        projected_corners = pose_utils.compute_projection(corners3D, transform_mat[:3, :], mtx)
        projected_vertices = pose_utils.compute_projection(vertices, transform_mat[:3, :], mtx)
        
        # Draw projected object
        im_with_bbox = frame.copy()
        im_with_bbox = pose_utils.draw_BBox(im_with_bbox, projected_corners.T, projected_vertices.T)
        
        # Create mask
        mask_arr = pose_utils.create_simple_mask(projected_vertices.T, im_width, im_height)
        print(f'the type is {type(mask_arr)}')
        mask_arr = binary_threshold(frame, threshold=100)
        print(f'the type is {type(mask_arr)} after binary thresholding')
        
        # Create label
        label = pose_utils.create_label(0, projected_vertices, mtx[0,0], mtx[1,1], 
                                       im_width, im_height, mtx[0,2], mtx[1,2], 
                                       im_width, im_height, transform_mat)
        
        # Create output filename
        imageName = f"{frame_count:06d}.png"
        
        # Store all information
        pose_utils.save_data(frame, mask_arr, label, imageName, output_path, im_with_bbox)
        
        frame_count += 1

print(f"Processed {frame_count} images")