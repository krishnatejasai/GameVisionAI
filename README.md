# GameVision AI

GameVision AI is an end-to-end computer vision pipeline for football video analytics. The system detects players, referees, and the ball from match footage, assigns persistent tracking IDs across frames, exports structured tracking data, and generates movement analytics including player trajectories, occupancy heatmaps, ball trajectories, and per-track motion statistics.

## Features

- Custom YOLO-based object detection for:
  - Players
  - Referees
  - Ball
- Multi-object tracking using ByteTrack
- Persistent object IDs across video frames
- Automated CSV export of tracking results
- Player trajectory visualization
- Player occupancy heatmap generation
- Ball trajectory analysis
- Per-track distance and speed estimation in image coordinates
- End-to-end pipeline execution from a single entry point

## Pipeline

Input Match Video  
→ Object Detection  
→ Multi-Object Tracking  
→ Tracking Data Export  
→ Movement Analytics  
→ Trajectory and Heatmap Visualization

## Project Structure

```text
gamevision-ai/
├── configs/
│   ├── config.yaml
│   └── football_players.yaml
├── data/
├── models/
├── notebooks/
├── outputs/
│   ├── metrics/
│   └── plots/
├── src/
│   ├── analytics.py
│   ├── detect.py
│   ├── inspect_dataset.py
│   ├── track.py
│   ├── utils.py
│   └── visualize.py
├── main.py
├── requirements.txt
└── README.md