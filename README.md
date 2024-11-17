# Autonomous-Navigation-and-Collision-Avoidance-Sytem-for-car

## Introduction
1. **Project Overview**
   
The "Autonomous-Navigation-System-for-Car" project focuses on developing a self-driving system that integrates two key functionalities: automatic lane-keeping and obstacle detection. The system enables a vehicle to autonomously navigate while staying within lane boundaries and avoiding collisions with obstacles.

3. **Objectives**
   
Implement an automatic navigation system that ensures the vehicle remains within its designated lane without causing lane invasions.
Develop an obstacle detection system capable of identifying and stopping the vehicle upon detecting objects within its path.

## System Architecture 
1. **Overall Structure**
    - Automatic Navigation System: Controls the vehicle's path, keeping it on track using real-time data processing.
    - Obstacle Detection System: Detects obstacles using sensors and halts vehicle motion when necessary.

2. **Data Flow and Interactions**
    - Sensors such as cameras and LiDAR collect data from the environment.
    - Image processing and point cloud data are processed using trained deep learning models to control vehicle movements.

## System Design and Implementation
### Automatic Navigation System
1. **Data Generation for Training**
    - Environment Setup: The system uses the CARLA simulator to generate training data.
    - Sensors: The vehicle is equipped with RGB and semantic segmentation cameras.
    - Route Simulation: Random routes are generated to simulate real-world driving conditions.
    - Data Collection: Data includes semantic segmentation images and steering angles.

2. **Model Architecture**
    - Model Used: The navigation model is a convolutional neural network (CNN) designed to predict the steering angle required to maintain the lane.
    - Training Process: The model is trained using labeled data to predict steering directions based on image inputs.

### Obstacle Detection System
1. **Data Generation**
    - Environment Setup: The system uses the CARLA simulator to generate training data.
    - Sensors used:
      - LiDAR sensor collects 3D point cloud data and converts it to PGM files, which store grayscale 2D images.
      - The Segmentation Camera collects segmentation images.
      - The Obstacle Detector sensor detects obstacles in front of the car.
    - Data collection includes semantic segmentation photos, PGM files, and labels that indicate whether or not there is an obstacle in front of the car.

*Note: All of these sensors are built-in sensors of the CARLA simulator.

2. **Model Architecture**
    - Model Used: The obstacle detection model is a CNN-based architecture trained to predict the presence of obstacles using combined inputs from PGM files and semantic images.







