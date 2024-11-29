# Introduction
* Project Overview
      * The "Autonomous-Navigation-System-for-Car" project focuses on developing a self-driving system that integrates two key functionalities: automatic lane-keeping and obstacle detection. 
      * The system enables a vehicle to autonomously navigate while staying within lane boundaries and avoiding collisions with obstacles.

* Objectives
      * Implement an automatic navigation system that ensures the vehicle remains within its designated lane without causing lane invasions.
      * Develop an obstacle detection system capable of identifying and stopping the vehicle upon detecting objects within its path.

# Features
* Automatic Lane Navigation:
    * Generates and follows a random navigation route using semantic segmentation data.
    * Maintains the car's position within lanes using a deep learning-based steering angle prediction model.

* Obstacle Detection:
    * Utilizes LiDAR point cloud data to detect obstacles in the car's path.
    * Stops the car when obstacles are detected.

* Deep Learning Integration:
    * A neural network is trained on semantic segmentation images and navigation direction labels to predict steering angles.

# System Architecture
**The system consists of two main components:**
* Perception Module:
    * LiDAR sensor for obstacle detection.
    * Semantic segmentation camera for lane and navigation detection.

* Control Module:
    * A deep learning-based controller trained to predict the steering angle based on semantic images and navigation labels.
    * Speed control using a throttle-braking mechanism.


# Dataset Preparation
* Semantic Segmentation Images:
    * Generated using CARLA's semantic segmentation camera.
    * Images are labeled with steering angles and navigation directions.
  
<div align=center>
  <img src="/images/1728788476498592800_0_0.0.png" width="300" />
</div>

* LiDAR Point Cloud Data:
    * Captures real-time obstacle data.
<div align=center>
  <img src="/images/cloud_point_data.png" width="300" />
</div>

* Data Statistics:
**Distribution of steering angles and navigation directions:**
<div align=center>
  <img src="/images/data_distribution.png" width="400" />
</div>


# Implementation Details
* Sensors Used:
    * LiDAR: Detects obstacles and processes point cloud data.
    * Semantic Segmentation Camera: Captures road lanes and other objects in a segmented format.

* Key Algorithms:
    * Route Selection: Uses CARLA's GlobalRoutePlanner to generate navigation routes.
    * Obstacle Detection: Filters LiDAR points within a defined region of interest (ROI).

* Control Logic:
    * Predicted steering angles control the car's direction.
    * Throttle-braking logic maintains a preferred speed.

# Results
* Steering Angle Prediction:
    * Achieved smooth lane following with minimal deviations.

* Obstacle Detection:
    * Accurate detection of obstacles using LiDAR.
    * Immediate stopping when an obstacle is detected.


<div align=center>
  <img src="/images/training_validation_loss.png" width="400" />
</div>

<div align=center>
  <img src="/images/demo.png" width="400"/>
  <img src="/images/demo1.png" width="400" height="238"/> 
</div>








