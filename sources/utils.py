import carla 
import numpy as np
import math
import sys
import random

from .constants import *

sys.path.append('E:\Download\Hung\WindowsNoEditor\PythonAPI\carla') # tweak to where you put carla
from agents.navigation.global_route_planner import GlobalRoutePlanner


def cleanup(world):
    for actor in world.get_actors().filter('*vehicle*'):
        actor.destroy()
    for actor in world.get_actors().filter('*sensor*'):
        actor.destroy()

def sem_callback(image,data_dict):
    image.convert(carla.ColorConverter.CityScapesPalette)
    data_dict['sem_image'] = np.reshape(np.copy(image.raw_data),(image.height,image.width,4))

def collision_callback(event,data_dict):
    data_dict['collision']=True

# maintain speed function
def maintain_speed(s):
    if s >= PREFERRED_SPEED:
        return 0
    elif s < PREFERRED_SPEED - SPEED_THRESHOLD:
        return 0.9 # think of it as % of "full gas"
    else:
        return 0.4 # tweak this if the car is way over or under preferred speed 


# function to get angle between the car and target waypoint
def get_angle(car,wp):
    vehicle_pos = car.get_transform()
    car_x = vehicle_pos.location.x
    car_y = vehicle_pos.location.y
    wp_x = wp.transform.location.x
    wp_y = wp.transform.location.y
    
    # vector to waypoint
    x = (wp_x - car_x)/((wp_y - car_y)**2 + (wp_x - car_x)**2)**0.5
    y = (wp_y - car_y)/((wp_y - car_y)**2 + (wp_x - car_x)**2)**0.5
    
    #car vector
    car_vector = vehicle_pos.get_forward_vector()
    degrees = math.degrees(np.arctan2(y, x) - np.arctan2(car_vector.y, car_vector.x))
    # extra checks on predicted angle when values close to 360 degrees are returned
    if degrees<-180:
        degrees = degrees + 360
    elif degrees > 180:
        degrees = degrees - 360
    return degrees

def get_proper_angle(car,wp_idx,rte):
    next_angle_list = []
    for i in range(10):
        if wp_idx + i*3 <len(rte)-1:
            next_angle_list.append(get_angle(car,rte[wp_idx + i*3][0]))
    idx = 0
    while idx<len(next_angle_list)-2 and abs(next_angle_list[idx])>40:
        idx +=1
    return wp_idx+idx*3,next_angle_list[idx]  

def get_distant_angle(car,wp_idx,rte, delta):
    if wp_idx + delta < len(rte)-1:
        i = wp_idx + delta
    else:
        i = len(rte)-1
    intersection_detected = False
    for x in range(i-wp_idx):
        if rte[wp_idx+x][0].is_junction:
             intersection_detected = True
    angle = get_angle(car,rte[i][0])
    if not intersection_detected:
        result = 1
    elif angle <-10:
        result = 0
    elif angle>10:
        result = 2
    else:
        result = 1  
    return int(result)

def draw_route(world, wp, route,seconds=3.0):
    if len(route)-wp <25: # route within 25 points from end is red
        draw_colour = carla.Color(r=255, g=0, b=0)
    else:
        draw_colour = carla.Color(r=0, g=0, b=255)
    for i in range(10):
        if wp+i<len(route)-2:
            world.debug.draw_string(route[wp+i][0].transform.location, '^', draw_shadow=False,
                color=draw_colour, life_time=seconds,
                persistent_lines=True)
    return None


def select_random_route(position,locs):
    point_a = position.location #we start at where the car is or last waypoint
    sampling_resolution = 1
    grp = GlobalRoutePlanner(world.get_map(), sampling_resolution)
    # now let' pick the longest possible route
    min_distance = 100
    result_route = None
    route_list = []
    for loc in locs: # we start trying all spawn points 
                                #but we just exclude first at zero index
        cur_route = grp.trace_route(point_a, loc.location)
        if len(cur_route) > min_distance:
            route_list.append(cur_route)
    result_route = random.choice(route_list)
    return result_route


def generate_lidar_bp(blueprint_library):
    """Generates a CARLA blueprint based on the script parameters"""
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')

    lidar_bp.set_attribute('upper_fov', str(10.0))
    lidar_bp.set_attribute('lower_fov', str(-30.0))
    lidar_bp.set_attribute('channels', str(64.0))
    lidar_bp.set_attribute('range', str(100.0))
    lidar_bp.set_attribute('rotation_frequency', str(20))
    lidar_bp.set_attribute('points_per_second', str(1000000))
    return lidar_bp


def convert_lidar_to_pgm_polar(lidar_data, num_layers=64, num_angular_steps=360, max_range=100.0, output_filename='lidar_pgm_image.pgm'):
    x, y, z = lidar_data[:, 0], lidar_data[:, 1], lidar_data[:, 2]
    r = np.sqrt(x**2 + y**2)  # Radial distance
    theta = np.arctan2(y, x)  # Angle in radians
    theta_degrees = np.degrees(theta)  # Convert to degrees
    theta_degrees = (theta_degrees + 360) % 360  # Ensure range [0, 360)

    # Discretize theta into bins
    angular_step_size = 360.0 / num_angular_steps
    theta_bins = (theta_degrees / angular_step_size).astype(int)
    
    # Normalize z values into layer indices
    layer_indices = ((z - np.min(z)) / (np.max(z) - np.min(z)) * (num_layers - 1)).astype(int)
    
    # Initialize a 2D grid (n x m) for PGM image
    pgm_grid = np.full((num_layers, num_angular_steps), 0, dtype=np.uint8)

    # Populate the PGM grid
    for i in range(len(lidar_data)):
        layer = layer_indices[i]
        angle_bin = theta_bins[i]
        distance = min(r[i], max_range)  # Clip the distance to max_range
        normalized_distance = int((distance / max_range) * 255)  # Normalize to [0, 255]
        pgm_grid[layer, angle_bin] = max(pgm_grid[layer, angle_bin], normalized_distance)

    # Save as a PGM file
    with open(output_filename, 'wb') as f:
        f.write(b'P5\n')
        f.write(f"{num_angular_steps} {num_layers}\n255\n".encode())
        pgm_grid.tofile(f)
        
        
def generate_traffic(world, client, num_vehicles=60, num_pedestrians=10):

    vehicles_list = []
    walkers_list = []
    all_id = []

    # Setup Traffic Manager
    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)
    traffic_manager.set_synchronous_mode(True)
    world_settings = world.get_settings()
    world_settings.synchronous_mode = True
    world.apply_settings(world_settings)

    # Get blueprints
    vehicle_blueprints = world.get_blueprint_library().filter('vehicle.*')
    walker_blueprints = world.get_blueprint_library().filter('walker.pedestrian.*')

    # Spawn vehicles
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)
    batch = []

    for n, transform in enumerate(spawn_points[:num_vehicles]):
        blueprint = random.choice(vehicle_blueprints)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        blueprint.set_attribute('role_name', 'autopilot')
        batch.append(carla.command.SpawnActor(blueprint, transform)
                        .then(carla.command.SetAutopilot(carla.command.FutureActor, True, traffic_manager.get_port())))

    for response in client.apply_batch_sync(batch, True):
        if not response.error:
            vehicles_list.append(response.actor_id)

    # Spawn pedestrians
    walker_spawn_points = []
    for _ in range(num_pedestrians):
        spawn_point = carla.Transform()
        loc = world.get_random_location_from_navigation()
        if loc:
            spawn_point.location = loc
            walker_spawn_points.append(spawn_point)

    walker_batch = []
    walker_speeds = []

    for spawn_point in walker_spawn_points:
        walker_bp = random.choice(walker_blueprints)
        walker_bp.set_attribute('is_invincible', 'false')
        walker_batch.append(carla.command.SpawnActor(walker_bp, spawn_point))

    walker_results = client.apply_batch_sync(walker_batch, True)
    for result in walker_results:
        if not result.error:
            walkers_list.append({"id": result.actor_id})

    # Add walker controllers
    walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
    controller_batch = []
    for walker in walkers_list:
        controller_batch.append(carla.command.SpawnActor(walker_controller_bp, carla.Transform(), walker["id"]))

    controller_results = client.apply_batch_sync(controller_batch, True)
    for i, result in enumerate(controller_results):
        if not result.error:
            walkers_list[i]["con"] = result.actor_id
            all_id.append(walkers_list[i]["con"])
            all_id.append(walkers_list[i]["id"])

    all_actors = world.get_actors(all_id)

    # Start the walkers
    for i in range(0, len(all_id), 2):
        all_actors[i].start()
        all_actors[i].go_to_location(world.get_random_location_from_navigation())
        all_actors[i].set_max_speed(1 + random.random())  # Random speed for walkers

    print(f"Spawned {len(vehicles_list)} vehicles and {len(walkers_list)} pedestrians.")
