#!/usr/bin/env python
'''
PID_velocity_controller
This code implements a simple PID controller for path tracking.
Author : Dvij Kalaria
'''
import time
import rospy
import math
import json
import xmltodict
from geometry_msgs.msg import Twist
from std_msgs.msg import Int16
from sensor_msgs.msg import Joy
from nav_msgs.msg import Odometry
from prius_msgs.msg import Control
from tf2_msgs.msg import TFMessage
import numpy as np
from nav_msgs.msg import Path
from inv_set_calc import *
# import matplotlib.pyplot as plt

# Node name       - controls
# Published topic  - pid_output (Twist)
# Subscribed topic - base_pose_ground_truth , cmd_vel, cmd_delta

gear_stat = "F"
# target omega
target_speed = 10
tar_delta = 0
error_sum = 0
prev_error = 0
error_diff = 0
output = 0
curr_x = 0
curr_y = 0
curr_yaw = 0
wheelbase = 1.958  # in meters
radius = 0  # radius of path
steering_angle = 0  # steering angle in degrees
kp = 0.80  # proportional gain
ki = 0.0001  # integral gain
kd = 0  # differential gain
acc_thershold = 0  # threshold for acceleration
brake_threshold = 20  # threshold for brake
global pub
global tar_vel
global tar_omega
path_followed = []
save_path_after = 10
buffer_cmd = [None]*8
has_started = False
save_at = 0

Q_robust = np.matrix(np.diag(np.array([1,1,1,1,1])))
R_robust = np.matrix(np.diag(np.array([.1,.1])))
N = 8
curr_speed = 0
curr_steering = 0

def prius_pub():
    '''
    publishes the velocity and steering angle
    published on topic : ackermann_cmd_topic
    '''
    global buffer_cmd
    global curr_x,curr_y,curr_yaw,curr_speed,curr_steering
    prius_vel = Control()
    i = 0
    curr_time = rospy.get_time()
    for pose in buffer_cmd :
        if pose.pose.position.y > curr_time :
            break
        i = i + 1
    if i!= 0 :
        i = i - 1
        dx = curr_x - buffer_cmd[i].pose.orientation.x
        dy = curr_y - buffer_cmd[i].pose.orientation.y
        # udy = dx*math.cos(curr_yaw) - dy*math.sin(curr_yaw)
        dyaw = curr_yaw - buffer_cmd[i].pose.orientation.z
        dvel = curr_speed - buffer_cmd[i].pose.orientation.w
        dsteer = curr_steering - (buffer_cmd[i].header.stamp.secs/1000 - 2)
        # curr_state_diff = np.matrix(np.array([[dx],[dy],[dvel],[dyaw],[dsteer]]))
        # K_mat = get_K(0.1,curr_yaw,curr_speed,Q_robust,R_robust,N)
        # correction = K_mat*curr_state_diff
        # print(correction)
        prius_vel.steer = (buffer_cmd[i].pose.position.x) / 40# - (180/math.pi)*correction[1,0]) / 40
        prius_vel.throttle = max(0,buffer_cmd[i].pose.position.z) # + correction[0,0]
        prius_vel.brake = -min(0,buffer_cmd[i].pose.position.z) # + correction[0,0]

    else :
        rospy.sleep(buffer_cmd[0].pose.position.y - curr_time)
        prius_vel.steer = buffer_cmd[0].pose.position.x / 40
    
    pub.publish(prius_vel)
    print("Published ", i, prius_vel.throttle, prius_vel.brake, prius_vel.steer, "!!!!!!!!!!!!", rospy.get_time())

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

def callback_tf(data):
    global iter0
    global gt_steering
    global times
    global pub
    global init_steering_x,init_steering_y,init_steering_z,init_steering_w
    global curr_steering
    for tf_data in data.transforms :
        if tf_data.header.frame_id == "chassis" and tf_data.child_frame_id == "fr_axle" :
            x = tf_data.transform.rotation.x #- init_steering_x
            y = tf_data.transform.rotation.y #- init_steering_y
            z = tf_data.transform.rotation.z #- init_steering_z
            w = tf_data.transform.rotation.w #- init_steering_w
            # xp = tf_data.transform.translation.x
            # yp = tf_data.transform.translation.y
            # zp = tf_data.transform.translation.z
            _,_,yaw = convert_xyzw_to_rpy(x,y,z,w)
            curr_steering = yaw
            gt_steering.append([yaw*180/3.14,rospy.get_time()])
            
def callback_feedback(data):
    '''
    Applies PID to velcity input from odom readings and publishes.
    :params data [Odometry]
    :params output [Twist]
    :params plot [Twist] 
    '''
    global has_started
    if not has_started :
        print("Waiting for path to get published")
        return
    global save_at
    global tar_delta
    global tar_omega
    global wheelbase
    global error_sum
    global error
    global error_diff
    global output
    global i
    global flag
    global kp
    global ki
    global kd
    global pub
    global prev_error
    global gear_stat
    global acc_thershold
    global brake_threshold
    global act_velocity, gt_steering
    global curr_x, curr_y, curr_yaw, curr_speed
    # conversion of odometry readings from quarternion to euler
    siny = +2.0 * (data.pose.pose.orientation.w *
                   data.pose.pose.orientation.z +
                   data.pose.pose.orientation.x *
                   data.pose.pose.orientation.y)
    cosy = +1.0 - 2.0 * (data.pose.pose.orientation.y *
                         data.pose.pose.orientation.y +
                         data.pose.pose.orientation.z *
                         data.pose.pose.orientation.z)
    yaw = math.atan2(siny, cosy)
    x = data.pose.pose.position.x
    y = data.pose.pose.position.y
    curr_x, curr_y, curr_yaw = x, y, yaw
    
    curr_speed = (data.twist.twist.linear.x**2 +
                 data.twist.twist.linear.y**2) ** 0.5
    path_followed.append([x,y,yaw,curr_speed,data.twist.twist.angular.z,rospy.get_time()])
    # error_sum += (target_speed-curr_speed)
    # throttle = ki*error_sum + kp*(target_speed-curr_speed)
    if save_path_after!=-1 and rospy.get_time()>save_at :
        # np.savetxt("exp6/path_without_compensation.csv", path_followed)
        np.savetxt("exp7/path_test_compensation.csv", path_followed)
        np.savetxt("exp7/steerings.csv", gt_steering)
    # applying PID on the Velocity
    # Thresholding the steering angle between 30 degrees and -30 degrees
    
    rospy.loginfo("linear velocity : %f", curr_speed)
    # publish the msg
    prius_pub()

def callback_cmd_vel(data):
    '''
    Subscribes from cmd_vel and gives target velocity
    :params data [Twist]
    :params tar_vel [float]
    '''
    global tar_vel
    tar_vel = data.linear.x

    # smoothing the velocity and reducing it at greater turning angles.

def callback_delta(data):
    '''
    Subscribes from cmd_delta and returns target steering angle.
    :params data [Twist]
    :params tar_delta [Float] 
    '''
    global has_started
    global save_path_after
    global save_at
    global buffer_cmd
    if not has_started :
        has_started = True
        save_at = rospy.get_time() + save_path_after
    if buffer_cmd[0] is not None :
        i = 0
        # tar_delta = data.angular.z
        curr_time = rospy.get_time()
        for pose in buffer_cmd :
            if pose.pose.position.y > curr_time :
                break
            i = i + 1
        if i!= 0 :
            i = i - 1
        j = 0
        start_time = data.poses[0].pose.position.y
        for pose in buffer_cmd :
            if pose.pose.position.y > start_time :
                break
            j = j + 1
        
        print("Buffer updated :",rospy.get_time(),i,j)
        buffer_cmd[:(j-i)] = buffer_cmd[i:j]
        buffer_cmd[(j-i):] = data.poses[:(8-(j-i))]
    else :
        buffer_cmd[:] = data.poses[:]

def start():
    global pub
    ackermann_cmd_topic = rospy.get_param('~ackermann_cmd_topic', '/prius')
    print(ackermann_cmd_topic)
    print("YES changed")
    test_cmd = Control()
    test_cmd.brake = 0
    test_cmd.throttle = 0
    test_cmd.steer = -1.57
    rospy.init_node('controls', anonymous=True)
    pub = rospy.Publisher(ackermann_cmd_topic, Control, queue_size=1)
    rospy.Subscriber("base_pose_ground_truth", Odometry, callback_feedback, queue_size=1)
    rospy.Subscriber("cmd_delta", Path, callback_delta, queue_size=1)
    rospy.Subscriber("/tf", TFMessage, callback_tf,queue_size=1)
    
    rospy.spin()


if __name__ == '__main__':
    start()
