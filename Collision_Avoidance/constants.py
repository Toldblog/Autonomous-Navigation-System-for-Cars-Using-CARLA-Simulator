import numpy as np
from matplotlib import cm

# CARLA SIMULATOR
PORT = 2000
TIMEOUT_MAX = 5.0 # seconds
DELTA = 0.05 # simulation frame rate of 1 / delta = 1 / 0.05 = 20 frames per second (FPS)

CAR_MODEL = "*model3*" 

# DBSCAN parameters
EPS = 0.5
MIN_SAMPLES = 10
CLUSTER_UPPER_EXCLUDE = 2 # Exclude any cluster have points have z higher than this value

# Default scale factor for arrows
ARROW_SCALE_FACTOR = 15.0

# Default frame rate for video
FRAME_RATE = 20

# Velocity threshold for drawing arrows (in km/h)
VELOCITY_THRESHOLD = 4

# Apply EMA in update the movement diretion when drawing arrow
UPDATED_DIRECTION_PARAM = 0.7
PREVIOUS_DIRECTION_PARAM = 0.3

# Distance threshold for tracking centroid of cluster (in meters)
DIST_THRESHOLD_TRACKING = 1

# LiDAR sensor configuration
LIDAR_UPPER_FOV = 10.0
LIDAR_LOWER_FOV = -30.0
LIDAR_CHANNELS = 64
LIDAR_RANGE = 100.0
LIDAR_ROTATION_FREQUENCY = 20
LIDAR_POINTS_PER_SECOND = 1000000

LIDAR_POSITION_HEIGHT = 2 # z = 2

# Vehicle speed control parameters
PREFERRED_SPEED = 30  # Desired speed in km/h
SPEED_THRESHOLD = 2

VIRIDIS = np.array(cm.get_cmap('plasma').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])

HEIGHT_UPPER_FILTER = 4 - LIDAR_POSITION_HEIGHT # Accept cloud point have height lower than 4 meter
HEIGHT_LOWER_FILTER = 0.3 - LIDAR_POSITION_HEIGHT # Accept cloud point have height higher than 0.3 meter

ZOOM_LIDAR_BIRD_EYE_PARAM = 0.1

VEHICLE_SIZE_RANGES = {
    'car': {'0': (1, 4.9), '1': (1.0, 2.0), '2': (1, 1.5)},
    'truck': {'0': (2.5, 6.0), '1': (1, 3.0), '2': (1.0, 3.0)},
    'van': {'0': (2.0, 5.5), '1': (0.5, 2.0), '2': (1.0, 1.8)},
    'motorcycle': {'0': (1.0, 2.5), '1': (0.3, 1.8), '2': (0.4, 1.7)}
}

VEHICLE_CENTROID_COLORS = {
    'car': [0, 0, 1],        # Blue for cars
    'truck': [1, 1, 0],      # Yellow for trucks
    'motorcycle': [0, 1, 0],  # Green for motorcycles
    'van': [0.5, 0.4, 0.4]
}

# Predefined color names and their RGB values
COLOR_NAMES = [
    'red', 'green', 'blue', 'yellow', 'orange', 'purple', 'cyan', 'magenta', 
    'pink', 'brown', 'lime', 'navy', 'olive', 'teal', 'maroon', 'turquoise', 
    'gold', 'silver', 'indigo', 'violet'
]

COLOR_VALUES = {
    'red': [1, 0, 0], 'green': [0, 1, 0], 'blue': [0, 0, 1], 'yellow': [1, 1, 0],
    'orange': [1, 0.5, 0], 'purple': [0.5, 0, 0.5], 'cyan': [0, 1, 1], 'magenta': [1, 0, 1],
    'pink': [1, 0.75, 0.8], 'brown': [0.6, 0.3, 0.1], 'lime': [0.75, 1, 0], 'navy': [0, 0, 0.5],
    'olive': [0.5, 0.5, 0], 'teal': [0, 0.5, 0.5], 'maroon': [0.5, 0, 0], 'turquoise': [0.25, 0.88, 0.82],
    'gold': [1, 0.84, 0], 'silver': [0.75, 0.75, 0.75], 'indigo': [0.29, 0, 0.51], 'violet': [0.93, 0.51, 0.93]
}
