import sys
import os
import cv2
import time
import copy
import numpy as np
from utils import debug, Singleton

# 碰撞异常抛出
class ClossionDetected(Exception):
    pass

sensor_config = {
    # 各相机参数（RGB，语义分割，深度等）
    'camera': {
        'length': 640,
        'width': 360,
        'fov': 120,
        'fps': 30,
    },
    'lidar': {
        'channels': 64,
        'rpm': 30,
        'fps': 30,
        'pps': 100000,
        'range': 20,  # meters for > 0.9.6
        'lower_fov': -30,
        'upper_fov': 10,
    },
    'imu': {
        'fps':400,
    },
    'gnss': {
        'fps':30,
    },
}

# ----------------
# 激光雷达数据可视化
# -----------------
# modify from world on rail code
def visualize_data(lidar, text_args=(0.6)):
    
    lidar_viz = lidar_to_bev(lidar).astype(np.uint8)
    lidar_viz = cv2.cvtColor(lidar_viz,cv2.COLOR_GRAY2RGB)
 
    return lidar_viz

# modify from world on rail code
def lidar_to_bev(lidar, min_x=-60,max_x=60,min_y=-60,max_y=60, pixels_per_meter=6, hist_max_per_pixel=2):
    xbins = np.linspace(
        min_x, max_x+1,
        (max_x - min_x) * pixels_per_meter + 1,
    )
    ybins = np.linspace(
        min_y, max_y+1,
        (max_y - min_y) * pixels_per_meter + 1,
    )
    # Compute histogram of x and y coordinates of points.
    hist = np.histogramdd(lidar[..., :2], bins=(xbins, ybins))[0]
    # Clip histogram
    hist[hist > hist_max_per_pixel] = hist_max_per_pixel
    # Normalize histogram by the maximum number of points in a bin we care about.
    overhead_splat = hist / hist_max_per_pixel * 255.
    # Return splat in X x Y orientation, with X parallel to car axis, Y perp, both parallel to ground.
    return overhead_splat[::-1,:]

# ---------------
# 添加传感器并绑定
# ---------------
# 添加RGB相机
def add_camera(world, blueprint, vehicle, transform):
    # 调用RGB相机蓝图
    camera_bp = blueprint.find('sensor.camera.rgb')
    # 蓝图属性设置 从字典config查表
    camera_bp.set_attribute('image_size_x', str(sensor_config['camera']['length']))
    camera_bp.set_attribute('image_size_y', str(sensor_config['camera']['width']))
    camera_bp.set_attribute('fov', str(sensor_config['camera']['fov']))
    # tick代表捕捉时间间隔（模拟器时间s） 为帧数倒数1./fps
    camera_bp.set_attribute('sensor_tick', str(1. / sensor_config['camera']['fps']))
    # 将蓝图用于spawn函数生成
    camera = world.spawn_actor(camera_bp, transform, attach_to=vehicle)
    return camera

# 添加语义分割相机（不同颜色显示每个对象）
def add_semantic(world, blueprint, vehicle, transform):
    # 调用蓝图
    semantic_bp = blueprint.find('sensor.camera.semantic_segmentation')
    # 属性设置，同普通rgb_camera
    semantic_bp.set_attribute('image_size_x', str(sensor_config['camera']['length']))
    semantic_bp.set_attribute('image_size_y', str(sensor_config['camera']['width']))
    semantic_bp.set_attribute('fov', str(sensor_config['camera']['fov']))
    semantic_bp.set_attribute('sensor_tick', str(1. / sensor_config['camera']['fps']))
    # 应用蓝图
    semantic = world.spawn_actor(semantic_bp, transform, attach_to=vehicle)
    return semantic


# 添加激光雷达
def add_lidar(world, blueprint, vehicle, transform):
    # 调用蓝图
    lidar_bp = blueprint.find('sensor.lidar.ray_cast')
    # 属性设置
    lidar_bp.set_attribute('channels', str(sensor_config['lidar']['channels']))
    lidar_bp.set_attribute('rotation_frequency', str(sensor_config['lidar']['rpm']))
    lidar_bp.set_attribute('points_per_second', str(sensor_config['lidar']['pps']))
    lidar_bp.set_attribute('sensor_tick', str(1. / sensor_config['lidar']['fps']))
    lidar_bp.set_attribute('range', str(sensor_config['lidar']['range']))
    lidar_bp.set_attribute('lower_fov', str(sensor_config['lidar']['lower_fov']))
    lidar_bp.set_attribute('upper_fov', str(sensor_config['lidar']['upper_fov']))
    # 应用蓝图
    lidar = world.spawn_actor(lidar_bp, transform, attach_to=vehicle)
    return lidar


# 添加惯性测量单位（加速度，陀螺仪，指南针）
def add_imu(world, blueprint, vehicle, transform):
    imu_bp = blueprint.find('sensor.other.imu')
    imu_bp.set_attribute('sensor_tick', str(1. / sensor_config['imu']['fps']))
    imu = world.spawn_actor(imu_bp, transform, attach_to=vehicle)
    return imu


# 添加全球导航卫星系统传感器
def add_gnss(world, blueprint, vehicle, transform):
    gnss_bp = blueprint.find('sensor.other.gnss')
    gnss_bp.set_attribute('sensor_tick', str(1. / sensor_config['gnss']['fps']))
    gnss = world.spawn_actor(gnss_bp, transform, attach_to=vehicle)
    return gnss


# 添加碰撞检测器
def add_collision(world, blueprint, vehicle, transform):
    collision_bp = blueprint.find('sensor.other.collision')
    collision = world.spawn_actor(collision_bp, transform, attach_to=vehicle)
    return collision

# 传感器总配置对象
class SensorManager(Singleton):
    # 构造函数 世界，蓝图，车辆，参数字典，已知传感器表初始化
    def __init__(self, world, blueprint, vehicle, param_dict):
        self.world = world
        self.blueprint = blueprint
        self.vehicle = vehicle
        self.param_dict = param_dict
        self.sensor_dict = {}

        self.known_sensors = ['camera', 'lidar', 'imu', 'gnss', 'semantic', 'collision']

    # 初始化传感器key
    def init(self, key):
        if key in self.param_dict:
            # 确认类型为已知传感器且取第一个
            sensor_type = self.get_type(key)
            # globals()全局变量字典
            # ['变量名']访问指定全局变量 即此前定义的各函数
            # 若无则直接创建SensorManager对象sensor 名为add_ + sensor_type
            sensor = globals()['add_' + sensor_type](
                self.world,
                self.blueprint,
                self.vehicle,
                self.param_dict[key]['transform'])
            # lambda表达式匿名函数data 实际即param_dict字典的callback对应的键值
            # listen用于TCP监听 槽函数
            # 目标采集数据
            sensor.listen(lambda data: self.param_dict[key]['callback'](data))            # 传感器字典
            self.sensor_dict[key] = sensor
            debug(info=key + ' successfully initialized !', info_type='success')
        else:
            debug(info='Unknown sensor ' + str(key), info_type='error')
            return None

    # param_dict内的全部传感器初始化
    def init_all(self):
        for key in self.param_dict:
            try:
                self.init(key)
            except:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                debug(info="Error type: {}. Error message: {}".format(key, exc_type, exc_value), info_type='error')
                debug(info=str(key) + ' initialize failed', info_type='error')
                
        time.sleep(0.5)

    # 关闭所有传感器 清除字典sensor_dict内容
    def close_all(self):
        for key in self.param_dict:
            try:
                self.sensor_dict[key].destroy()
                debug(info=str(key) + ' closed', info_type='success')
            except:
                debug(info=str(key) + ' has no attribute called \'close\'', info_type='message')

    # 析构函数
    def __del__(self):
        pass
        # self.close_all()

    # get sensor instance
    # 实现对象SensorManager的迭代功能
    # 返回self.sensor_dict的键表
    def __getitem__(self, key):
        if key in self.sensor_dict:
            return self.sensor_dict[key]
        else:
            debug(info='No sensor called ' + str(key), info_type='error')
            return None

    # set sensor param
    # 存储key映射的值value
    def __setitem__(self, key, value):
        if key in self.param_dict:
            self.param_dict[key] = value
            return True
        else:
            debug(info='No sensor called ' + str(key), info_type='error')
            return None

    # 获取传感器key的类型 若为多个则取冒号前第一个
    def get_type(self, key):
        sensor_type = key.split(':')[0]
        if sensor_type in self.known_sensors:
            return sensor_type
        else:
            debug(info='Unknown sensor type ' + str(key), info_type='error')
            return None
        
class SaveData():
    def __init__(self, save_path, enable_save=True):
        self.img = None
        self.seg_img = None
        self.pcd = None
        self.nav = None
        self.control = None
        self.pos = None
        self.acceleration = None
        self.angular_velocity = None
        self.vel = None
        
        self.path = str(save_path)
        self.enable_save = enable_save
        self.perepare_save()
        
    def perepare_save(self):
        if not self.enable_save:
            return
        os.makedirs(self.path, exist_ok=True)
        os.makedirs(self.path + 'img/', exist_ok=True)
        os.makedirs(self.path + 'segimg/', exist_ok=True)
        os.makedirs(self.path + 'pcd/', exist_ok=True)
        os.makedirs(self.path + 'nav/', exist_ok=True)
        os.makedirs(self.path + 'state/', exist_ok=True)
        os.makedirs(self.path + 'cmd/', exist_ok=True)
        
        # 文件写入指针
        self.cmd_file = open(self.path + 'cmd/cmd.txt', 'w+')
        self.pos_file = open(self.path + 'state/pos.txt', 'w+')
        self.vel_file = open(self.path + 'state/vel.txt', 'w+')
        self.acc_file = open(self.path + 'state/acc.txt', 'w+')
        self.angular_vel_file = open(self.path + 'state/angular_vel.txt', 'w+')
    
    def save(self, time):
        # RGB和NAV
        if not self.enable_save:
            return
        cv2.imwrite(self.path + 'img/' + str(time) + '.png', self.img)
        cv2.imwrite(self.path + 'segimg/' + str(time) + '.png', self.seg_img)
        cv2.imwrite(self.path + 'nav/' + str(time) + '.png', self.nav)
        
        np.save(self.path + 'pcd/' + str(time) + '.npy', self.pcd)
        
        self.cmd_file.write(time + '\t' +
                   str(self.control.throttle) + '\t' +
                   str(self.control.steer) + '\t' +
                   str(self.control.brake) + '\n')
        # 使用时采用0, 1, 2, 3, 5，即ts, x, y, z, yaw
        self.pos_file.write(time + '\t' +
                    str(self.pos.location.x) + '\t' +
                    str(self.pos.location.y) + '\t' +
                    str(self.pos.location.z) + '\t' +
                    str(self.pos.rotation.pitch) + '\t' +
                    str(self.pos.rotation.yaw) + '\t' +
                    str(self.pos.rotation.roll) + '\t' + '\n')
        self.vel_file.write(time + '\t' +
                    str(self.vel.x) + '\t' +
                    str(self.vel.y) + '\t' +
                    str(self.vel.z) + '\t' + '\n')
        self.acc_file.write(time + '\t' +
                    str(self.acceleration.x) + '\t' +
                    str(self.acceleration.y) + '\t' +
                    str(self.acceleration.z) + '\t' + '\n')
        self.angular_vel_file.write(time + '\t' +
                            str(self.angular_velocity.x) + '\t' +
                            str(self.angular_velocity.y) + '\t' +
                            str(self.angular_velocity.z) + '\t' + '\n')
    
    def close_all(self):
        if not self.enable_save:
            return
        try:
            self.acc_file.close()
            self.angular_vel_file.close()
            self.cmd_file.close()
            self.pos_file.close()
            self.vel_file.close()
            
            debug(info='close all files', info_type='success')
        except:
            debug(info='failed to close all files', info_type='error')