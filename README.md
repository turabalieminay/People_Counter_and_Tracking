# People_Counter_and_Tracking


# Human Detection and Tracking using YOLOv8 and SORT Algorithm

This project implements human detection and tracking using the YOLOv8 model and the SORT tracking algorithm. The system detects people in a video and tracks their movement frame by frame, assigning unique IDs to individuals and counting the number of people in the scene.

## Project Overview

In this project, I've developed a solution that detects humans in video feeds and tracks them across frames. I used the YOLOv8s model for detection and the SORT (Simple Online and Realtime Tracking) algorithm for tracking. This solution is highly efficient for real-time human detection and tracking tasks, especially in surveillance or similar applications.

## Features

- **Human Detection**: Detects humans in each frame using the pre-trained YOLOv8s model.
- **Tracking**: Tracks detected humans across multiple frames using the SORT algorithm.
- **Unique ID Assignment**: Assigns a unique ID to each detected person and maintains that ID across frames.
- **People Counting**: Displays the number of people detected in the current frame.

## How It Works

1. **Human Detection**: 
   - The YOLOv8 model is used to detect humans in each frame of the video. I chose the `yolov8s.pt` model for its balance of accuracy and speed.
   
2. **Tracking**:
   - The SORT algorithm is used to track detected humans across frames. SORT is a simple yet effective method that uses Kalman filters and the Hungarian algorithm to match detections across frames.
   
3. **ID Assignment**:
   - Each detected person is assigned a unique ID. If a person leaves the frame, their ID is made available for reuse by new people entering the frame.
   
4. **Counting**:
   - The number of people detected in the current frame is displayed on the video. The IDs of people are tracked until they leave the scene.

## Dependencies

To run this project, you'll need the following libraries:
- `opencv-python`
- `ultralytics`
- `numpy`
- `sort` (Install the SORT algorithm from its repository)

You can install the dependencies using `pip`:
```bash
pip install opencv-python ultralytics numpy
