# Lane-detection-and-obstacle-avoidance
This project focuses on robust lane detection and scene understanding using a U-Net based deep learning model combined with Bird’s Eye View (BEV) projection and post-processing techniques for generating structured lane corridors and centerlines. It keeps an understanding of lane markings specific for hard constained environment like warehouses, factories and offices for a robot to navigate itself through a particular region.

What it does mainly:
- Segmented lane marking masks.
- Derived lane corridors and centerlines for controlled and precise maneuvering with Annotated visualization of the detected road structure.

# Project implementation
- Data Annotation & Preparation
- Model Training (U-Net)
- Point generation & BEV Transformation
- Post-Processing & Inference
- Visualization & Evaluation

# Improvrments required
- multilane understanding changes that directly affects working of the model
- Real-time deployment by converting the model to TensorRT or ONNX for embedded inference on raspberry-pi / edge devices.
- Camera-LiDAR Fusion: Fusing visual lane boundaries with LiDAR point clouds for 3D road surface modeling.

This is a prototype level project!
