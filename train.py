class MultiCameraFeatureFusion(nn.Module):
    def __init__(self, base_model, num_cameras=6):
        super(MultiCameraFeatureFusion, self).__init__()
        self.num_cameras = num_cameras
        
        # Use the backbone from the existing model
        self.backbone = base_model.model[:24]  # Adjust based on architecture
        
        # Feature transformation layers
        self.transform_layer = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        
        # Fusion network
        self.fusion_network = nn.Sequential(
            nn.Linear(512 * num_cameras, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU()
        )
        
        # Pose regression head
        self.pose_head = nn.Linear(512, 7)  # position (3) + quaternion (4)
        
    def forward(self, images, camera_transforms=None):
        """
        Args:
            images: Tensor of shape [batch, num_cameras, channels, height, width]
            camera_transforms: Tensor of relative camera transformations 
                            [batch, num_cameras, 4, 4]
        """
        batch_size = images.shape[0]
        all_features = []
        
        # Process each camera view
        for i in range(self.num_cameras):
            # Extract features from backbone (weights shared across all cameras)
            camera_images = images[:, i]  # [batch, channels, height, width]
            features = self.backbone(camera_images)
            features = self.transform_layer(features)
            
            # Global average pooling
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features.view(batch_size, -1)
            
            if camera_transforms is not None:
                # Apply transform-aware weighting (optional)
                transform = camera_transforms[:, i].view(batch_size, -1)
                # This is where you could use the transform to weight features
            
            all_features.append(features)
        
        # Concatenate features from all cameras
        concat_features = torch.cat(all_features, dim=1)
        
        # Fuse features
        fused = self.fusion_network(concat_features)
        
        # Predict 6D pose (as position + quaternion)
        pose = self.pose_head(fused)
        
        return pose