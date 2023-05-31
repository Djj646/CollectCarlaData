import sys
import glob
import os
import logging
from utils import debug

from os.path import join, dirname
sys.path.insert(0, join(dirname(__file__), '..'))

path = 'D:/CARLA_0.9.13/WindowsNoEditor/PythonAPI'
sys.path.append(path) # 包括carla
sys.path.append(path+'/carla') # 包括agents

try:
    sys.path.append(glob.glob(path + '/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from carla import VehicleLightState as vls # 
SpawnActor = carla.command.SpawnActor
SetAutopilot = carla.command.SetAutopilot
SetVehicleLightState = carla.command.SetVehicleLightState
FutureActor = carla.command.FutureActor
DestroyActor = carla.command.DestroyActor

from agents.navigation.behavior_agent import BehaviorAgent
from agents.navigation.basic_agent import BasicAgent

from sensor_manager import ClossionDetected, SensorManager, SaveData, visualize_data
from utils.navigator_sim import get_random_destination, get_map, get_nav, get_global_nav_map, get_global_nav, replan, close2dest

import os
import cv2
import time
import copy
import random
import argparse
import numpy as np
from tqdm import tqdm

world_config = {
    'host': 'localhost',
    'port': 2000,
    'timeout': 10.0,
    'town': 'Town01',
    'weather': carla.WeatherParameters.ClearNoon,
    'check_arrival_distance': 10,
    'min_route_lenth': 200,
    'collision_time_frames_allowed': 200,
}

# 传感器回馈函数暂存数据
global_img = None
global_pcd = None
global_pcd_unfilted = None
global_seg_img = None
global_lidar_viz = None
global_lidar_viz_unfilted = None

# 碰撞持续时间帧数检测
collision_time_frame = 0

argparser = argparse.ArgumentParser(description=__doc__)
argparser.add_argument(
    '--enable-save',
    action='store_true',
    help='Save the data(default False)')
argparser.add_argument(
    '-M','--map-number',
    metavar='SERIAL NUMBER',
    default=1,
    type=int,
    help='Map Serial Number (1-5, default: 1)')
argparser.add_argument(
    '--vel',
    metavar='SPEED',
    default=30,
    type=int,
    help='speed of the host vehilce (default: 30)')
argparser.add_argument(
    '--vehicles',
    metavar='NUMBER',
    default=10,
    type=int,
    help='number of vehicles (default: 6)')
argparser.add_argument(
    '--walkers',
    metavar='NUMBER',
    default=0,
    type=int,
    help='number of walkers (default: 10)')
argparser.add_argument(
    '--safe',
    action='store_true',
    help='avoid spawning vehicles prone to accidents')
argparser.add_argument(
    '--filterv',
    metavar='Vehicle PATTERN',
    default='vehicle.*',
    help='vehicles filter (default: "vehicle.*")')
argparser.add_argument(
    '--filterw',
    metavar='Walker PATTERN',
    default='walker.pedestrian.*',
    help='pedestrians filter (default: "walker.pedestrian.*")')
argparser.add_argument(
    '--tm-port',
    metavar='P',
    default=8000,
    type=int,
    help='port to communicate with TM (default: 8000)')
argparser.add_argument(
    '--sync',
    action='store_false',
    default=True,
    help='Not Synchronous mode execution(default: True)')
argparser.add_argument(
    '--hybrid',
    action='store_true',
    help='Enanble Hybrid Mode')
argparser.add_argument(
    '--car-lights-on',
    action='store_true',
    default=False,
    help='Enanble car lights')
argparser.add_argument(
    '--traffic-lights',
    action='store_true',
    default=False,
    help='Ignore Traffic Lights(make sure there is no npc)')
argparser.add_argument(
    '--data-num',
    type=int,
    default=100000,
    help='Total Number'
)
argparser.add_argument(
    '--data-index',
    type=int,
    default=1,
    help='Data Index'
)
args = argparser.parse_args()

# 图像反馈函数 将相机data存入全局变量global_img中，NumPy数组形状(h, w, 4)
def image_callback(data):
    global global_img
    array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
    # (samples, channels, height, width)
    # 4 channels: R, G, B, A
    array = np.reshape(array, (data.height, data.width, 4))  # RGBA format
    global_img = array
    
# 激光雷达反馈函数 将激光雷达data的points中选择符合两个坐标条件的
# 存入全局变量global_pcd中
def lidar_callback(data):
    global global_pcd, global_pcd_unfilted, global_lidar_viz, global_lidar_viz_unfilted
    # (samples, (dim1, dim2, dim3)) NumPy二维数组 点集points
    lidar_data = np.frombuffer(data.raw_data, dtype=np.float32).reshape([-1, 4])
    # np.stack用于堆叠数组
    # 第一维度为sample 分别取第1，0，2列坐标，对应y，x，z，变成3行的二维数组point_cloud
    point_cloud = np.stack([-lidar_data[:, 1], -lidar_data[:, 0], -lidar_data[:, 2]])
    mask = \
        np.where((point_cloud[0] > 1.0) | (point_cloud[0] < -4.0) | (point_cloud[1] > 1.2) | (point_cloud[1] < -1.2))[0]
    # np.where(condition)返回符合第一行坐标条件的序号（以元组的形式） 取元组第一位保存到mask中
    # 取满足条件的sample序号mask 对x, y坐标做出维度坐标限定
    point_cloud = point_cloud[:, mask]
    # 在满足第一行条件下（只取一列）
    # 取满足第三行坐标条件的序号（最多一列）
    mask = np.where(point_cloud[2] > -1.95)[0]
    point_cloud = point_cloud[:, mask]
    
    global_lidar_viz = visualize_data(point_cloud.transpose())
    global_lidar_viz_unfilted = visualize_data(lidar_data)
    global_pcd = point_cloud
    global_pcd_unfilted = lidar_data
    
def seg_image_callback(data):
    global global_seg_img
    data.convert(carla.ColorConverter.CityScapesPalette)
    array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
    # (samples, channels, height, width)
    # 4 channels: R, G, B, A
    array = np.reshape(array, (data.height, data.width, 4))  # RGBA format
    global_seg_img = array[:, :, :3]

def collision_callback(event):
    global collision_time_frame
    collision_time_frame = collision_time_frame+1

def add_vehicle(world, blueprint, vehicle_type='vehicle.bmw.grandtourer'):
    bp = random.choice(blueprint.filter(vehicle_type))
    if bp.has_attribute('color'):
        color = random.choice(bp.get_attribute('color').recommended_values)
        bp.set_attribute('color', color)
    transform = random.choice(world.get_map().get_spawn_points())
    vehicle = world.spawn_actor(bp, transform)
    return vehicle

def add_npc_vehicles(client, world, tm, filter, safe, sync, light, num):
    vehicles_list = []
    vehicles_blueprint = world.get_blueprint_library().filter(filter)
    
    # 保留四轮车移除危险车
    if safe:
        vehicles_blueprint = [x for x in vehicles_blueprint if int(x.get_attribute('number_of_wheels')) == 4]
        vehicles_blueprint = [x for x in vehicles_blueprint if not x.id.endswith('isetta')]
        vehicles_blueprint = [x for x in vehicles_blueprint if not x.id.endswith('carlacola')]
        vehicles_blueprint = [x for x in vehicles_blueprint if not x.id.endswith('cybertruck')]
        vehicles_blueprint = [x for x in vehicles_blueprint if not x.id.endswith('t2')]
    
    spawn_points = world.get_map().get_spawn_points()
    num_of_spawn_points = len(spawn_points)
    
    if num < num_of_spawn_points:
        random.shuffle(spawn_points)
    elif num > num_of_spawn_points:
        msg = 'requested %d vehicles, but could only find %d spawn points'
        logging.warning(msg, num, num_of_spawn_points)
        num = num_of_spawn_points
    
    batch = []
    for n, transform in enumerate(spawn_points):
        if n >= num:
            break
        # 根据属性选择生成角色蓝图
        blueprint = random.choice(vehicles_blueprint)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        blueprint.set_attribute('role_name', 'autopilot')

        light_state = vls.NONE
        if light:
            light_state = vls.Position | vls.LowBeam | vls.LowBeam
        
        batch.append(SpawnActor(blueprint, transform)
                         .then(SetAutopilot(FutureActor, True, tm.get_port()))
                         .then(SetVehicleLightState(FutureActor, light_state)))
    
    for response in client.apply_batch_sync(batch, sync):
            if response.error:
                logging.error(response.error)
            else:
                vehicles_list.append(response.actor_id)
    
    return vehicles_list
    
def add_npc_walkers(client, world, filter, sync, num):
    all_id = []
    walkers_list = []
    walkers_blueprint = world.get_blueprint_library().filter(filter)
    
    percentagePedestriansRunning = 0.0  # how many pedestrians will run
    percentagePedestriansCrossing = 0.0  # how many pedestrians will walk through the road

    spawn_points = []
    for i in range(num):
        spawn_point = carla.Transform()
        loc = world.get_random_location_from_navigation()
        if (loc != None):
            spawn_point.location = loc
            spawn_points.append(spawn_point)
    
    batch = []
    walker_speed = []
    
    for spawn_point in spawn_points:
        walker_bp = random.choice(walkers_blueprint)
        # 无敌状态
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')
        if walker_bp.has_attribute('speed'):
            if (random.random() > percentagePedestriansRunning):
                # walking
                walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
            else:
                # running
                walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
        else:
            print("Walker has no speed")
            walker_speed.append(0.0)
        batch.append(SpawnActor(walker_bp, spawn_point))
    
    results = client.apply_batch_sync(batch, True)
    walker_speed2 = []
    for i in range(len(results)):
        if results[i].error:
            logging.error(results[i].error)
        else:
            walkers_list.append({"id": results[i].actor_id})
            walker_speed2.append(walker_speed[i])
    walker_speed = walker_speed2
    
    # walker controller
    batch = []
    walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
    for i in range(len(walkers_list)):
        batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))

    results = client.apply_batch_sync(batch, True)
    for i in range(len(results)):
        if results[i].error:
            logging.error(results[i].error)
        else:
            walkers_list[i]["con"] = results[i].actor_id
    # put altogether the walkers and controllers id to get the objects from their id
    for i in range(len(walkers_list)):
        all_id.append(walkers_list[i]["con"])
        all_id.append(walkers_list[i]["id"])
    all_actors = world.get_actors(all_id)
    
    if not sync:
        world.wait_for_tick()
    else:
        world.tick()
        
    world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
    for i in range(0, len(all_id), 2):
        # start walker
        all_actors[i].start()
        # set walk to random point
        all_actors[i].go_to_location(world.get_random_location_from_navigation())
        # max speed
        all_actors[i].set_max_speed(float(walker_speed[int(i / 2)]))

    return walkers_list, all_actors, all_id

def main():
    global args, global_pcd, global_pcd_unfilted, global_img, global_seg_img, \
        global_lidar_viz, global_lidar_viz_unfilted, collision_time_frame
    
    world_config['town'] = 'Town' + '{:02d}'.format(args.map_number)
    debug(info='Collect From Map: '+world_config['town'], info_type='message')

    sdata = SaveData(save_path='/data2/wanghejun/NewCICT/datacollect/DATASET/CARLA/Segmentation/' \
        + world_config['town'] + '/' + str(args.data_index) + '/', enable_save=args.enable_save)

    client = carla.Client(world_config['host'], world_config['port'])
    client.set_timeout(world_config['timeout'])
    world = client.load_world(world_config['town'])
    world.set_weather(world_config['weather'])
    
    blueprints = world.get_blueprint_library()
    world_map = world.get_map()
    spectator = world.get_spectator()# 监视器

    ego_vehicle = add_vehicle(world, blueprints, vehicle_type='vehicle.audi.a2')
    ego_vehicle.set_simulate_physics(True)
    
    sensor_dict = {
        'camera': {
            'transform': carla.Transform(carla.Location(x=0.5, y=0.0, z=2.5)),
            'callback': image_callback,
        },
        'lidar': {
            'transform': carla.Transform(carla.Location(x=0.5, y=0.0, z=2.5)),
            'callback': lidar_callback,
        },
        'semantic': {
            'transform': carla.Transform(carla.Location(x=0.5, y=0.0, z=2.5)),
            'callback': seg_image_callback,
        },
        'collision': {
            'transform': carla.Transform(carla.Location(x=0.5, y=0.0, z=2.5)),
            'callback': collision_callback,
        },
    }
    
    sm = SensorManager(world, blueprints, ego_vehicle, sensor_dict)
    sm.init_all()
    
    des_spawn_points = world_map.get_spawn_points()

    # 返回openDRIVE文件的拓扑的最小图元祖列表
    waypoint_tuple_list = world_map.get_topology()
    origin_map = get_map(waypoint_tuple_list)
    
    if args.traffic_lights:
        debug(info='Ignore Traffic Lights (Only With No Npc)', info_type='message')
        
    agent = BasicAgent(ego_vehicle, target_speed=args.vel, opt_dict={'ignore_traffic_lights': args.traffic_lights})
    
    # 第一次导航
    destination = get_random_destination(des_spawn_points)
    plan_map, route_lenth = replan(agent, destination, copy.deepcopy(origin_map))
    global_nav_map = get_global_nav_map(plan_map)
    
    settings = world.get_settings()
    traffic_manager = client.get_trafficmanager(args.tm_port)
    traffic_manager.set_global_distance_to_leading_vehicle(2.0)
    
    if args.hybrid:
        traffic_manager.set_hybrid_physics_mode(True)
    
    if args.sync:
        traffic_manager.set_synchronous_mode(True)
        debug(info='Synchronous Mode', info_type='message')
        if not settings.synchronous_mode:
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            world.apply_settings(settings)
    else:
        debug(info='Not Synchronous Mode', info_type='message')
    
    vehicles_list = add_npc_vehicles(client, world, traffic_manager, args.filterv, args.safe, args.sync, args.car_lights_on, args.vehicles)
    walkers_list, all_actors, all_id = add_npc_walkers(client, world, args.filterw, args.sync, args.walkers)

    debug(info='spawned %d vehicles and %d walkers !' % (len(vehicles_list), len(walkers_list)), info_type='success')
    
    try:
        for cnt in tqdm(range(args.data_num)):
            # world.tick() # sync_mode
            # world.wait_for_tick() # not sync_mode
            (world.tick() if args.sync else world.wait_for_tick())
            
            if collision_time_frame>world_config['collision_time_frames_allowed']:
                raise ClossionDetected
            
            if close2dest(ego_vehicle, destination, world_config['check_arrival_distance']):
                destination = get_random_destination(des_spawn_points)
                plan_map, route_lenth = replan(agent, destination, copy.deepcopy(origin_map))
                
                while route_lenth<world_config['min_route_lenth']:
                    # debug(info='New destination too close ! Replaning...', info_type='warning')
                    destination = get_random_destination(des_spawn_points)
                    plan_map, route_lenth = replan(agent, destination, copy.deepcopy(origin_map))

                global_nav_map = get_global_nav_map(plan_map)

            control = agent.run_step()
            # control = carla.VehicleControl(throttle=30, steer=0)
            ego_vehicle.apply_control(control)
            sdata.control = control
            # 获得卫星导航图
            sdata.nav = get_nav(ego_vehicle, plan_map)
            # 获得全局位置
            global_nav = get_global_nav(ego_vehicle, global_nav_map)
            # 位置与姿态信息：x, y, z, pitch(y), yaw(z), roll(x) 
            sdata.pos = ego_vehicle.get_transform()
            sdata.vel = ego_vehicle.get_velocity()
            sdata.acceleration = ego_vehicle.get_acceleration()
            sdata.angular_velocity = ego_vehicle.get_angular_velocity()
            sdata.img = global_img
            sdata.pcd = global_pcd
            sdata.seg_img = global_seg_img
            
            cv2.imshow('Nav', sdata.nav)
            cv2.imshow('Global_Nav', global_nav)
            cv2.imshow('Vision', sdata.img)
            # cv2.imshow('SegVision', sdata.seg_img)
            # cv2.imshow('Lidar', global_lidar_viz)
            # cv2.imshow('Lidar_Unfilted', global_lidar_viz_unfilted)
            cv2.waitKey(10)
            time_index = str(time.time())
            sdata.save(time_index)
    except KeyboardInterrupt:
        print("Exit by user !")
    except ClossionDetected:
        print("Clossion detected !")
    finally:
        debug(info='destroying ego_vehicle', info_type='message')
        client.apply_batch([DestroyActor(ego_vehicle)])
        
        debug(info='destroying %d vehicles and %d walkers' % (len(vehicles_list),len(walkers_list)), info_type='message')
        client.apply_batch([DestroyActor(x) for x in vehicles_list])

        for i in range(0, len(all_actors), 2):
            all_actors[i].stop()

        client.apply_batch([DestroyActor(x) for x in all_id])

        sm.close_all()
        sdata.close_all()
        cv2.destroyAllWindows()
        
        # 重置world
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)
        traffic_manager.set_synchronous_mode(False)
        
        time.sleep(0.5)


if __name__ == '__main__':
    try: 
        main()
    except KeyboardInterrupt:
        print("Exit by user !")
