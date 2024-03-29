#!/usr/bin/env python
'''
This code implements a low level pre-compensator 
for following high level command from robust MPC
'''

# Imports
if True :
    import rospy
    import math
    from nav_msgs.msg import Odometry
    from prius_msgs.msg import Control
    from tf2_msgs.msg import TFMessage
    import numpy as np
    from nav_msgs.msg import Path
    from inv_set_calc import *
    # import matplotlib.pyplot as plt

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
    has_started = False
    save_at = 0


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

# IMPORTANT : Publish to prius vehicle
def prius_pub():
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
        
        # Need to interpolate to get nominal state at current time 
        time_diff = pose.pose.position.y - curr_time
        interp_ratio = 1 - time_diff/.1
        dx = curr_x - (1-interp_ratio)*buffer_cmd[i].pose.orientation.x - interp_ratio*buffer_cmd[i+1].pose.orientation.x
        dy = curr_y - (1-interp_ratio)*buffer_cmd[i].pose.orientation.y - interp_ratio*buffer_cmd[i+1].pose.orientation.y
        dyaw = curr_yaw - (1-interp_ratio)*buffer_cmd[i].pose.orientation.z - interp_ratio*buffer_cmd[i+1].pose.orientation.z
        dvel = curr_speed - (1-interp_ratio)*buffer_cmd[i].pose.orientation.w - interp_ratio*buffer_cmd[i+1].pose.orientation.w
        dsteer = curr_steering - (1-interp_ratio)*(buffer_cmd[i].header.stamp.secs/1000 - 2) - interp_ratio*(buffer_cmd[i+1].header.stamp.secs/1000 - 2)
        
        curr_state_diff = np.matrix(np.array([[dx],[dy],[dvel],[dyaw],[dsteer]]))
        K_mat = get_K(0.1,curr_yaw,curr_speed,Q_robust,R_robust,N)
        correction = K_mat*curr_state_diff
        # print(correction, curr_steering, (1-interp_ratio)*(buffer_cmd[i].header.stamp.secs/1000 - 2) + interp_ratio*(buffer_cmd[i+1].header.stamp.secs/1000 - 2))
        prius_vel.steer = (buffer_cmd[i].pose.position.x - (180/math.pi)*correction[1,0]) / 40
        prius_vel.throttle = max(0,buffer_cmd[i].pose.position.z+ correction[0,0])
        prius_vel.brake = -min(0,buffer_cmd[i].pose.position.z+ correction[0,0])
    else :
        rospy.sleep(buffer_cmd[0].pose.position.y - curr_time)
        prius_vel.steer = buffer_cmd[0].pose.position.x / 40
    
    pub.publish(prius_vel)
    print("Published ", i, prius_vel.throttle, prius_vel.brake, prius_vel.steer, "!!!!!!!!!!!!", rospy.get_time())

# For getting current steering position
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
            print("Updated steering angle")
            _,_,curr_steering = convert_xyzw_to_rpy(x,y,z,w)

# Called at high frequency with updated position of the vehicle
def callback_feedback(data):
    global has_started
    if not has_started :
        print("Waiting for path to get published")
        return
    global save_at
    global i
    global pub
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
    path_followed.append([x,y,yaw,curr_speed,rospy.get_time()])
    if save_path_after!=-1 and rospy.get_time()>save_at :
        np.savetxt("exp3/path_ego_without_actuator_dyn.csv", path_followed)
    # applying PID on the Velocity
    # Thresholding the steering angle between 30 degrees and -30 degrees
    
    rospy.loginfo("linear velocity : %f", curr_speed)
    # publish the msg
    prius_pub()

# Called when new sequence of commands are received from high level Robust MPC
# Update the buffer with new commands when called
def callback_delta(data):
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
        
        print("Buffer updated :",rospy.get_time(),i,j, " at ", rospy.get_time())
        # print("Received buffer", data.poses)
        i=0
        j=0
        # buffer_cmd[:(j-i)] = buffer_cmd[i:j]
        buffer_cmd[(j-i):] = data.poses[:(8-(j-i))]
    else :
        buffer_cmd[:] = data.poses[:]

# Initialization
def start():
    global pub
    ackermann_cmd_topic = rospy.get_param('~ackermann_cmd_topic', '/prius')
    print(ackermann_cmd_topic)
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
