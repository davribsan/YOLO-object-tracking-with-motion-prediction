# YOLO object tracking with motion prediction

Real-time single-object tracking in MATLAB that combines two YOLOv8 models
with a Kalman filter:

- **Detection model** (`best_640_8.onnx`, 640x640): runs on the full frame to
  find the target at the start or after losing it.
- **Tracking model** (`best_320_8.onnx`, 320x320): runs on a small crop centered
  on the Kalman prediction, so it is much faster.
- **Kalman filter** (2D constant velocity, adaptive noise): smooths the
  trajectory and predicts the target position between inferences or while the
  target is temporarily lost.

Inference runs asynchronously on parallel workers (`parfeval`), so the video
keeps playing at its real frame rate.

## How it works

1. The detector finds the target on the full frame.
2. The detection must be confirmed in consecutive frames (yellow box).
3. Once confirmed, the fast model tracks the target (red box) and the Kalman
   filter is corrected with every measurement.
4. If the target is lost, the Kalman filter keeps predicting for a few frames,
   then the box freezes (green box) while the detector searches again.

## Requirements

- MATLAB with:
  - Deep Learning Toolbox
  - Deep Learning Toolbox Converter for ONNX Model Format
  - Computer Vision Toolbox
  - Parallel Computing Toolbox

## Usage

1. Open MATLAB and set the project folder as the current folder.
2. Put your video in `videos/` and set its name in `videoPath`
   (section 1 of `object_tracking.m`).
3. Run `object_tracking.m`.

## Project structure

```
.
├── object_tracking.m   % main script
├── Models/                              % ONNX models: best_640_8 (detection), best_320_8 (tracking)
├── Classes/                              % YOLOv8 wrappers and custom split layer
├── Functions/                            % model import, inference and worker
├── +utils/                               % pre/post-processing (DFL, anchors, NMS...)
└── videos/                               % your input videos (not tracked by Git)
```
