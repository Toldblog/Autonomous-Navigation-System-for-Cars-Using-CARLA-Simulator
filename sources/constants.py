
HEIGHT, WIDTH = 180, 320

BATCH_SIZE = 32

SPEED_THRESHOLD = 2 #defines when we get close to desired speed so we drop the throttle

# Max steering angle
MAX_STEER_DEGREES = 40
# This is max actual angle with Mini under steering input=1.0
STEERING_CONVERSION = 75
PREFERRED_SPEED = 40
#camera mount offset on the car - this mimics Tesla Model 3 view 
CAMERA_POS_Z = 1.3 
CAMERA_POS_X = 1.4 

CAM_HEIGHT = 480
CAM_WIDTH = 640
FOV = 90 # field of view = focal length

YAW_ADJ_DEGREES = 25 #random spin angle max