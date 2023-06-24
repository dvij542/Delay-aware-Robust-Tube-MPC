#!/usr/bin/env python
'''
This code implements a low level pre-compensator 
for following high level command from robust MPC
'''

# Imports
from random import random


if True :
    import rospy
    import math
    import numpy as np
    from inv_set_calc import *
    import sys
    import os
    import glob
    import time
    import random
    # import matplotlib.pyplot as plt
    from beginner_tutorials.msg import custom_msg
    try:
        sys.path.append(glob.glob('../../../carla/dist/carla-*%d.%d-%s.egg' % (
            sys.version_info.major,
            sys.version_info.minor,
            'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    except IndexError:
        pass


    # ==============================================================================
    # -- imports -------------------------------------------------------------------
    # ==============================================================================


    import carla
    
# Variables
if True :
    Q_robust = np.matrix(np.diag(np.array([1,1,1,1,1])))
    R_robust = np.matrix(np.diag(np.array([1,1])))
    N = 8
    save_path_after = 5

# Global variables for internal communication (Don't change)
if True :
    curr_x = 0
    curr_y = 0
    curr_yaw = 0
    curr_speed = 0
    curr_steering = 0

    path_followed = []
    buffer_cmd = [None]*8 # Buffer
    buffer_times = [None]*8 # Buffer
    buffer_steer = [0]*8 # Buffer
    buffer_throttle = [1]*8 # Buffer
    has_started = False
    save_at = 0
    curr_time = 0

    iter0 = 0
    gt_steering = []
    init_steering_x = 0
    init_steering_y = 0
    init_steering_z = 0
    init_steering_w = 0

def convert_xyzw_to_rpy(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z # in radians




# Called at high frequency with updated position of the vehicle
def callback_feedback(timestamp):
    global has_started,curr_time
    curr_time = timestamp.elapsed_seconds
    if not has_started :
        print("Waiting for commands to be received")
        return
    

# Called when new sequence of commands are received from high level Robust MPC
# Update the buffer with new commands when called
def callback_commands(data):
    global has_started, curr_time, buffer_times, buffer_steer, buffer_throttle
    # global save_path_after
    # global save_at
    # global buffer_cmd
    if buffer_times[0] is not None :
        i = 0
        # tar_delta = data.angular.z
        for t in buffer_times :
            if t > curr_time :
                break
            i = i + 1
        if i!= 0 :
            i = i - 1
        j = 0
        start_time = data.times[0]
        for t in buffer_times :
            if t > start_time :
                break
            j = j + 1
        
        print("Buffer updated :",curr_time,i,j)
        # print("Received buffer", data.poses)
        i=0
        j=0
        # buffer_cmd[:(j-i)] = buffer_cmd[i:j]
        buffer_times[(j-i):] = data.times[:(8-(j-i))]
        buffer_steer[(j-i):] = data.steerings[:(8-(j-i))]
        buffer_throttle[(j-i):] = data.throttles[:(8-(j-i))]
    else :
        buffer_times[:] = data.times[:]
        buffer_steer[:] = data.steerings[:]
        buffer_throttle[:] = data.throttles[:]
    
    if not has_started :
        has_started = True
    
# Initialization
def start():
    global pub
    rospy.init_node('controls', anonymous=True)
    rospy.Subscriber("chatter", custom_msg, callback_commands, queue_size=1)
    global world, player
    world = None
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(2.0)
    world = client.get_world()
    world.on_tick(callback_feedback)
    player = world.get_actors().filter('*mustang*')[0]
    # rospy.spin()
    print("Started")
    _control = carla.VehicleControl()
    global buffer_times, buffer_steer, buffer_throttle, curr_time
    
    while not rospy.is_shutdown() :
        time.sleep(0.04)
        if not has_started :
            continue
        # prius_pub()
        i = 0
        for t in buffer_times :
            if t > curr_time :
                break
            i = i + 1
        if i!= 0 :
            i = i - 1
        _control.steer = buffer_steer[i]
        _control.throttle = buffer_throttle[i]
        player.apply_control(_control)
        print("Sent command at ", curr_time, _control.steer, _control.throttle)
        
        


if __name__ == '__main__':
    start()
