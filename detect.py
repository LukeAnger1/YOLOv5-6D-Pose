import argparse
from re import I
import time
from pathlib import Path
import yaml
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
from models.experimental import attempt_load
from utils.datasets import LoadMultiViewImages
from utils.general import check_img_size, \
    scale_coords, set_logging, increment_path, retrieve_image
from utils.torch_utils import select_device, time_synchronized
from utils.pose_utils import box_filter, get_3D_corners, pnp, get_camera_intrinsic, MeshPly

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import os

class MultiViewPoseNet(torch.nn.Module):
    def __init__(self, base_model, num_views=6):
        super(MultiViewPoseNet, self).__init__()
        self.num_views = num_views
        
        # Load the base YOLOv5 backbone and detection layers
        self.base_model = base_model
        
        # Feature fusion module for multi-view integration
        self.fusion_layer = torch.nn.Sequential(
            torch.nn.Conv2d(1024 * num_views, 1024, kernel_size=1),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU()
        )
        
        # Pose estimation head (reuses base model's detection head connections)
        self.connect_fusion_to_detection()
        
    def connect_fusion_to_detection(self):
        """Connect fusion layer output to the existing detection head"""
        # This will depend on the specific YOLOv5 architecture
        # In practice, you would need to modify the model's internal connections
        pass
    
    def forward(self, images, transforms=None):
        """
        Args:
            images: Tensor of shape [batch_size, num_views, 3, H, W]
            transforms: Tensor of shape [batch_size, num_views, 4, 4] - relative camera transforms
        """
        batch_size = images.shape[0]
        num_views = images.shape[1]
        
        # Process each view to get features
        all_features = []
        for view_idx in range(min(num_views, self.num_views)):
            # Extract features from this view
            view_images = images[:, view_idx]
            
            # Get backbone features only (without running detection head)
            with torch.no_grad():
                # This implementation will depend on specific YOLOv5 structure
                view_features = self.extract_backbone_features(view_images)
            
            # Apply transform-aware weighting if transforms are provided
            if transforms is not None:
                view_transform = transforms[:, view_idx]
                # Apply transform-based weighting to features (custom implementation)
                
            all_features.append(view_features)
        
        # Fill any missing views with zeros if less than num_views provided
        while len(all_features) < self.num_views:
            all_features.append(torch.zeros_like(all_features[0]))
        
        # Concatenate features along channel dimension
        concat_features = torch.cat(all_features, dim=1)
        
        # Fuse features
        fused_features = self.fusion_layer(concat_features)
        
        # Feed through the rest of the model to get detections
        # This will depend on the specific YOLOv5 architecture
        outputs = self.feed_features_to_detection_head(fused_features)
        
        return outputs
    
    def extract_backbone_features(self, images):
        """Extract features from the backbone only"""
        # Implementation depends on the specific YOLOv5 structure
        # This is a simplified example
        features = images
        for i, layer in enumerate(self.base_model.model[:24]):  # Adjust slice as needed
            features = layer(features)
            if i == 23:  # Adjust index as needed for feature extraction point
                return features
        return features
    
    def feed_features_to_detection_head(self, features):
        """Feed features through detection head"""
        # Implementation depends on the specific YOLOv5 structure
        # This is a simplified example
        x = features
        for i, layer in enumerate(self.base_model.model[24:]):  # Adjust slice as needed
            x = layer(x)
        return x


def detect(save_img=False):
    source, weights, view_img, imgsz, mesh_data, cam_intrinsics, num_views = opt.source, opt.weights, opt.view_img, opt.img_size, opt.mesh_data, opt.static_camera, opt.num_views
    camera_transform_file = opt.camera_transforms
    
    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()

    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    base_model = attempt_load(weights, map_location=device)  # load FP32 model
    
    # Create multi-view model wrapper
    model = MultiViewPoseNet(base_model, num_views=num_views).to(device)
    
    stride = int(base_model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Load camera transforms if provided
    camera_transforms = None
    if camera_transform_file:
        try:
            with open(camera_transform_file, 'r') as f:
                camera_transforms_data = yaml.load(f, Loader=yaml.SafeLoader)
                camera_transforms = np.array(camera_transforms_data['transforms'])
                print(f"Loaded camera transforms with shape: {camera_transforms.shape}")
        except Exception as e:
            print(f"Error loading camera transforms: {e}")
            camera_transforms = None

    if cam_intrinsics:
        with open(cam_intrinsics) as f:
            cam_intrinsics = yaml.load(f, Loader=yaml.FullLoader)

        dtx = np.array(cam_intrinsics["distortion"])
        mtx = np.array(cam_intrinsics["intrinsic"])

        fx = mtx[0,0]
        fy = mtx[1,1]
        u0 = mtx[0,2]
        v0 = mtx[1,2]

        internal_calibration = get_camera_intrinsic(u0, v0, fx, fy)   

    save_img = True
    # Create multi-view dataset loader
    dataset = LoadMultiViewImages(source, img_size=imgsz, stride=stride, num_views=num_views)

    # Get names and colors
    names = base_model.module.names if hasattr(base_model, 'module') else base_model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    mesh = MeshPly(mesh_data)
    vertices = np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].transpose()
    corners3D = get_3D_corners(vertices)

    edges_corners = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]
    colormap = np.array(['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'xkcd:sky blue'])

    # Run inference
    if device.type != 'cpu':
        # Run dummy inference to initialize
        dummy_input = torch.zeros(1, num_views, 3, imgsz, imgsz).to(device).type_as(next(model.parameters()))
        model(dummy_input)

    predictions = []
    t0 = time.time()
    count = 0
    
    for batch_idx, (paths, imgs, intrinsics_list, shapes) in enumerate(dataset):
        t1 = time_synchronized()
        
        # Process images
        imgs_tensor = torch.from_numpy(imgs).to(device)
        imgs_tensor = imgs_tensor.half() if half else imgs_tensor.float()
        imgs_tensor /= 255.0  # 0 - 255 to 0.0 - 1.0
        
        # Add batch dimension if needed
        if imgs_tensor.ndimension() == 4:  # [num_views, 3, H, W]
            imgs_tensor = imgs_tensor.unsqueeze(0)  # [1, num_views, 3, H, W]
        
        # Prepare transforms tensor if available
        transforms_tensor = None
        if camera_transforms is not None:
            transforms_tensor = torch.from_numpy(camera_transforms).float().to(device)
            transforms_tensor = transforms_tensor.unsqueeze(0)  # Add batch dimension
        
        # Compute intrinsics if not provided
        if cam_intrinsics is None:
            # Use the first view's intrinsics for now (ideally would use all)
            first_view_intrinsics = intrinsics_list[0]
            fx, fy, det_height, u0, v0, im_native_width, im_native_height = first_view_intrinsics
            internal_calibration = get_camera_intrinsic(u0, v0, fx, fy)

        # Inference
        t2 = time_synchronized()
        with torch.no_grad():
            pred, _ = model(imgs_tensor, transforms_tensor)
        
        # Using confidence threshold, eliminate low-confidence predictions
        pred = box_filter(pred, conf_thres=opt.conf_thres)
        t3 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if len(det) == 0:
                continue
                
            print(f"Detection: {det}")
            
            # Get paths and image for the first view (for visualization)
            p = paths[0] if isinstance(paths, list) else paths
            im0s = cv2.imread(str(p))  # Get first view for visualization
            shapes_first = shapes[0] if isinstance(shapes, list) else shapes
            
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{batch_idx}')  # img.txt
            (Path(str(save_dir / 'labels'))).mkdir(parents=True, exist_ok=True) 
            s = '%gx%g ' % (imgsz, imgsz)  # print string
            
            det = det.cpu()
            
            # Rescale boxes from img_size to im0 size (using first view for now)
            scale_coords(imgs_tensor[0, 0].shape[1:], det[:, :18], shapes_first[0], shapes_first[1])  # native-space pred
            
            # Process each detection
            for j in range(len(det)):
                prediction_confidence = det[j, 18]
                box_predn = det[j, :18].clone()
                
                # Denormalize the corner predictions 
                corners2D_pr = np.array(np.reshape(box_predn, [9, 2]), dtype='float32')
                
                # Calculate rotation and tranlation in rodriquez format
                R_pr, t_pr = pnp(np.array(np.transpose(np.concatenate((np.zeros((3, 1)), corners3D[:3, :]), axis=1)), dtype='float32'),  
                                corners2D_pr, np.array(internal_calibration, dtype='float32'))
                
                pose_mat = cv2.hconcat((R_pr, t_pr))
                euler_angles = cv2.decomposeProjectionMatrix(pose_mat)[6]
                predictions.append([det[j], euler_angles, t_pr])
                
                # Print results
                if len(det):
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Visualization (using first view)
                if save_img:
                    local_img = cv2.cvtColor(im0s, cv2.COLOR_BGR2RGB)
                    figsize = (shapes_first[0][1]/96, shapes_first[0][0]/96)
                    fig = plt.figure(frameon=False, figsize=figsize)
                    ax = plt.Axes(fig, [0., 0., 1., 1.])
                    ax.set_axis_off()
                    fig.add_axes(ax)
                    
                    ax.imshow(local_img, aspect='auto')
                    corn2D_pr = corners2D_pr[1:, :]

                    # Plot projection corners
                    for edge in edges_corners:
                        ax.plot(corn2D_pr[edge, 0], corn2D_pr[edge, 1], color='b', linewidth=0.5)
                    ax.scatter(corners2D_pr.T[0], corners2D_pr.T[1], c=colormap, s=10)
                    
                    min_x, min_y = np.amin(corners2D_pr.T[0]), np.amin(corners2D_pr.T[1])
                    max_x, max_y = np.amax(corners2D_pr.T[0]), np.amax(corners2D_pr.T[1])

                    ax.text(min_x, min_y-10, f"Conf: {prediction_confidence:.3f}, Rot: {euler_angles}", 
                            style='italic', bbox={'facecolor': 'blue', 'alpha': 0.5, 'pad': 10})
                    
                    filename = os.path.basename(str(p)).split('.')[0] + "_" + str(count) + "_predicted.png"
                    file_path = os.path.join(save_dir, filename)
                    fig.savefig(file_path, dpi=96, bbox_inches='tight', pad_inches=0)
                    plt.close()

                    count += 1

                with open(txt_path + '.txt', 'a') as f:
                    f.write(str(det.numpy()) + '\n')

            # Print time (inference + NMS)
            print(f'{s}Done. ({t3 - t1:.3f}s)')
            
            # Save results (first view image with detections)
            if save_img and dataset.mode == 'image':
                cv2.imwrite(save_path, im0s)

    s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}"
    print(f"Results saved to {save_dir}{s}")
    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5l6_pose.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, help='source directory with multiple camera views')
    parser.add_argument('--static-camera', type=str, help='path to static camera intrinsics')
    parser.add_argument('--mesh-data', type=str, help='path to object specific mesh data')  
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--num-views', type=int, default=6, help='number of camera views to process')
    parser.add_argument('--camera-transforms', type=str, default=None, help='path to camera transform data (YAML)')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect()