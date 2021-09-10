# -*- coding: utf-8 -*-

#!/usr/bin/env python3
from casadi import *
import numpy as np
import rospy
import math
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseArray
from nav_msgs.msg import Path
from std_msgs.msg import Float64
from tf2_msgs.msg import TFMessage
import pickle
import time
from nav_msgs.msg import Path
# from ackermann_msgs.msg import AckermannDriveStamped
from prius_msgs.msg import Control

from geometry_msgs.msg import PointStamped

import numpy as np
import matplotlib.pyplot as plt
import copy
import math
import sys
import os


try:
    from quintic_polynomials_planner import QuinticPolynomial
    import cubic_spline_planner
except ImportError:
    raise

################# PARAMS ####################

# Parameter
speed_less_at = [5,1]
speed_increase_at = [150,5]

start_point = [-81,-12]
end_point = [100,-12]
target_speed = 7
kp = 0.75
ki = 0.0001
kd = 0
err_sum = 0
path_followed = []
throttles = []
# start_time = 0
started = False
curr_x = 0
curr_y = 0
curr_speed_x = 0
curr_speed_y = 0
SIM_LOOP = 500
x_obs = 0
y_obs = 0
curr_vel = 0 
not_turned = False
# Parameter
MAX_SPEED = 50.0 / 3.6  # maximum speed [m/s]
MAX_ACCEL = 5.0  # maximum acceleration [m/ss]
MAX_CURVATURE = 5.0  # maximum curvature [1/m]
MAX_ROAD_WIDTH = 20.0  # maximum road width [m]
D_ROAD_W = 2.0  # road width sampling length [m]
DT = 0.2  # time tick [s]
MAX_T = 5.0  # max prediction time [m]
MIN_T = 4.0  # min prediction time [m]
TARGET_SPEED = 36.0 / 3.6  # target speed [m/s]
D_T_S = 1.0 / 3.6  # target speed sampling length [m/s]
N_S_SAMPLE = 1  # sampling number of target speed
ROBOT_RADIUS = 2.2  # robot radius [m]

# cost weights
K_J = 0.1
K_T = 0.1
K_D = 1.0
K_LAT = 1.0
K_LON = 1.0
K_obs_dist = 50.0

show_animation = True


class QuarticPolynomial:

    def __init__(self, xs, vxs, axs, vxe, axe, time):
        # calc coefficient of quartic polynomial

        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[3 * time ** 2, 4 * time ** 3],
                      [6 * time, 12 * time ** 2]])
        b = np.array([vxe - self.a1 - 2 * self.a2 * time,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t ** 2 + \
             self.a3 * t ** 3 + self.a4 * t ** 4

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + \
             3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2

        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t

        return xt

class FrenetPath:

    def __init__(self):
        self.t = []
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.d_ddd = []
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.s_ddd = []
        self.cd = 0.0
        self.cv = 0.0
        self.cf = 0.0

        self.x = []
        self.y = []
        self.yaw = []
        self.ds = []
        self.c = []

def calc_frenet_paths(c_speed, c_d, c_d_d, c_d_dd, s0):
    frenet_paths = []

    # generate path to each offset goal
    for di in np.arange(0, MAX_ROAD_WIDTH, D_ROAD_W):

        # Lateral motion planning
        for Ti in np.arange(MIN_T, MAX_T, DT):
            fp = FrenetPath()

            # lat_qp = quintic_polynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)
            lat_qp = QuinticPolynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)

            fp.t = [t for t in np.arange(0.0, Ti, DT)]
            fp.d = [lat_qp.calc_point(t) for t in fp.t]
            fp.d_d = [lat_qp.calc_first_derivative(t) for t in fp.t]
            fp.d_dd = [lat_qp.calc_second_derivative(t) for t in fp.t]
            fp.d_ddd = [lat_qp.calc_third_derivative(t) for t in fp.t]

            # Longitudinal motion planning (Velocity keeping)
            for tv in np.arange(TARGET_SPEED - D_T_S * N_S_SAMPLE,
                                TARGET_SPEED + D_T_S * N_S_SAMPLE, D_T_S):
                tfp = copy.deepcopy(fp)
                lon_qp = QuarticPolynomial(s0, c_speed, 0.0, tv, 0.0, Ti)

                tfp.s = [lon_qp.calc_point(t) for t in fp.t]
                tfp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
                tfp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
                tfp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]

                Jp = sum(np.power(tfp.d_ddd, 2))  # square of jerk
                Js = sum(np.power(tfp.s_ddd, 2))  # square of jerk

                # square of diff from target speed
                ds = (TARGET_SPEED - tfp.s_d[-1]) ** 2

                tfp.cd = K_J * Jp + K_T * Ti + K_D * tfp.d[-1] ** 2
                tfp.cv = K_J * Js + K_T * Ti + K_D * ds
                tfp.cf = K_LAT * tfp.cd + K_LON * tfp.cv

                frenet_paths.append(tfp)

    return frenet_paths

def calc_global_paths(fplist, csp):
    for fp in fplist:

        # calc global positions
        # print("len of path:", len(fp.s))
        for i in range(len(fp.s)):
            # print(fp.s[i])
            try :
                ix, iy = csp.calc_position(fp.s[i])
                if ix is None:
                    break
            except :
                continue
            i_yaw = csp.calc_yaw(fp.s[i])
            di = fp.d[i]
            fx = ix + di * math.cos(i_yaw + math.pi / 2.0)
            fy = iy + di * math.sin(i_yaw + math.pi / 2.0)
            fp.x.append(fx)
            fp.y.append(fy)

        # calc yaw and ds
        # print(len(fp.x))
        for i in range(len(fp.x) - 1):
            dx = fp.x[i + 1] - fp.x[i]
            dy = fp.y[i + 1] - fp.y[i]
            fp.yaw.append(math.atan2(dy, dx))
            fp.ds.append(math.hypot(dx, dy))

        # print(len(fp.yaw))
        fp.yaw.append(fp.yaw[-1])
        fp.ds.append(fp.ds[-1])

        # calc curvature
        for i in range(len(fp.yaw) - 1):
            fp.c.append((fp.yaw[i + 1] - fp.yaw[i]) / fp.ds[i])

    return fplist

def check_collision(fp, ob, ob_speed):
    d_avg = []
    for i in range(len(ob[:, 0])):
        d = []
        for j in range(len(fp.x)) :
            ix = fp.x[j]
            iy = fp.y[j]
            obx = ob[i,0] + ob_speed[i,0]*DT*j
            oby = ob[i,1] + ob_speed[i,1]*DT*j
            d += [(((ix - obx)/3) ** 2 + (iy - oby) ** 2)]
        collision = any([di <= ROBOT_RADIUS ** 2 for di in d])
        d_avg += [min(d)]
        if collision:
            return False, 0
    d_avg = min(d_avg)
    return True, d_avg

def check_paths(fplist, ob, ob_speed):
    ok_ind = []
    dists = []
    for i, _ in enumerate(fplist):
        if any([v > MAX_SPEED for v in fplist[i].s_d]):  # Max speed check
            # print("1")
            continue
        elif any([abs(a) > MAX_ACCEL for a in
                  fplist[i].s_dd]):  # Max accel check
            # print("2")
            continue
        elif any([abs(c) > MAX_CURVATURE for c in
                  fplist[i].c]):  # Max curvature check
            # print("3")
            continue
        cond, min_dist = check_collision(fplist[i], ob, ob_speed)
        if not cond:
            # print("4")
            continue
        dists.append(1/min_dist)
        ok_ind.append(i)

    return [fplist[i] for i in ok_ind],dists

def frenet_optimal_planning(csp, s0, c_speed, c_d, c_d_d, c_d_dd, ob, ob_speed):
    fplist = calc_frenet_paths(c_speed, c_d, c_d_d, c_d_dd, s0)
    # print("No of paths:", len(fplist))
    fplist = calc_global_paths(fplist, csp)
    # print("a", len(fplist))
    fplist,dists = check_paths(fplist, ob, ob_speed)
    # print("b", len(fplist))

    # find minimum cost path
    min_cost = float("inf")
    best_path = None
    ind = 0
    for fp in fplist:
        # print(fp.cf)
        if min_cost >= ( fp.cf + K_obs_dist*dists[ind]):
            min_cost = ( fp.cf + K_obs_dist*dists[ind])
            best_path = fp
        ind = ind + 1
    return best_path

def generate_target_course(x, y):
    csp = cubic_spline_planner.Spline2D(x, y)
    s = np.arange(0, csp.s[-1], 0.1)

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = csp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(csp.calc_yaw(i_s))
        rk.append(csp.calc_curvature(i_s))

    return rx, ry, ryaw, rk, csp

def callback_ego_feedback(odom_data1) :
    global curr_x,curr_y,curr_speed_x,curr_speed_y, curr_vel, last_speed_y, last_time_t, not_turned
    curr_x = odom_data1.pose.pose.position.x
    curr_y = odom_data1.pose.pose.position.y
    curr_speed_x = odom_data1.twist.twist.linear.x
    curr_speed_y = odom_data1.twist.twist.linear.y
    omega =odom_data1.twist.twist.angular.z
    # print("Got speed ego :", curr_x,curr_y,curr_speed_x, curr_speed_y)
    # initial state
    
    c_speed = min((curr_speed_x**2 + curr_speed_y**2)**0.5+1,10)  # current speed [m/s]
        
    c_d = curr_y - ty[-1] # current lateral position [m]
    c_d_d = max(curr_speed_y,0)  # current lateral speed [m/s]
    c_d_dd = curr_speed_x*omega  # current lateral acceleration [m/s]
    s0 = curr_x - tx[0]  # current course position
    ob = np.array([
                   [x_obs, y_obs],
                   ])
    print("##############################", curr_vel)
    ob_speed = np.array([
                   [curr_vel, 0],
                   ])
    if rospy.get_time() < start_time + speed_less_at[0] or not started:
        c_d = 0
        c_d_d = 0
        c_d_dd = 0
        print("dummied")
        ob_speed = np.array([
                   [7, 0],
                   ])
        path = FrenetPath()
        path.y = [ty[-1]]*3
        path.x = [curr_x,curr_x+2*TARGET_SPEED,curr_x+2*TARGET_SPEED]
    else :
        # print(s0, c_speed, c_d, c_d_d, c_d_dd, ob, ob_speed)
        path = frenet_optimal_planning(
                csp, s0, c_speed, c_d, c_d_d, c_d_dd, ob, ob_speed)
    # print("Received data!",x_obs,y_obs)
    dyn_path = Path()
    dyn_path.header.frame_id='map'
    # print(path.x)
    for itr in range(len(path.x[1:])):
        pose=PoseStamped()
        pose.pose.orientation.x=0
        pose.pose.orientation.y=0
        pose.pose.orientation.z=0
        pose.pose.orientation.w=1
        pose.pose.position.x = path.x[itr]
        pose.pose.position.y = path.y[itr]
        dyn_path.poses.append(pose)
    
    dyn_path.header.stamp = rospy.Time.now()
    # rospy.sleep(0.2)
    pub1.publish(dyn_path)
    print("Successfully published")
    not_turned = True

def callback_feedback(odom_data) :
    global pub2
    global err_sum
    global path_followed,throttles
    global t1
    global x_obs,y_obs,curr_vel
    global start_time, target_speed, started
    x_obs = odom_data.pose.pose.position.x
    y_obs = odom_data.pose.pose.position.y

    # print("Frame :", odom_data.header.frame_id)
    # conversion of odometry readings from quarternion to euler
    siny = +2.0 * (odom_data.pose.pose.orientation.w *
                   odom_data.pose.pose.orientation.z +
                   odom_data.pose.pose.orientation.x *
                   odom_data.pose.pose.orientation.y)
    cosy = +1.0 - 2.0 * (odom_data.pose.pose.orientation.y *
                         odom_data.pose.pose.orientation.y +
                         odom_data.pose.pose.orientation.z *
                         odom_data.pose.pose.orientation.z)
    yaw_obs = math.atan2(siny, cosy)
    curr_vel = ((odom_data.twist.twist.linear.x)**2 + (odom_data.twist.twist.linear.y)**2)**0.5
    print("Current speed is : ", curr_vel)
    path_followed.append(np.array([x_obs,y_obs,yaw_obs,curr_vel,rospy.get_time()]))
    if not started :
        started = True
        start_time = rospy.get_time()
        t1 = rospy.get_time() + 10

    print(start_time, rospy.get_time() - start_time)
    if rospy.get_time() > start_time + speed_less_at[0] :
        target_speed = speed_less_at[1]
    if rospy.get_time() > start_time + speed_increase_at[0] :
        target_speed = speed_increase_at[1]
    print("Target speed is : ", target_speed)
    err = (target_speed-curr_vel)
    cmd = kp*err + ki*err_sum
    err_sum += err
    cmd = max(-0.9,min(0.9,cmd))
    cmd_pub = Control()
    # print(cmd)
    if cmd > 0:
        cmd_pub.throttle = cmd
        cmd_pub.brake = 0
    else :
        cmd_pub.throttle = 0
        cmd_pub.brake = -cmd
        
    cmd_pub.steer = 0
    throttles.append([cmd,curr_vel,rospy.get_time()])
    if rospy.get_time() > t1 :
        # np.savetxt('a',)
        np.savetxt('exp5/throttles.txt',np.array(throttles))
        np.savetxt('exp5/path_opp_with_comp.csv',np.array(path_followed))
        print("Saved")

    if not_turned :
        print(cmd_pub.throttle, cmd_pub.brake)
        pub2.publish(cmd_pub)

   
def start():
    global pub1
    global pub2
    global t1 
    global tx,ty,tyaw,tc,csp,start_time
    
    tx, ty, tyaw, tc, csp = generate_target_course([start_point[0],end_point[0]], [start_point[1],end_point[1]])
    # print(f([1,2,3],[1,2]))
    # print(tx,ty,csp)
    rospy.init_node('dummy_path_publisher', anonymous=True)
    pub1 = rospy.Publisher('astroid_path', Path,queue_size=1)
    pub2 = rospy.Publisher('/obstacle/prius', Control, queue_size=1)
    rospy.Subscriber("/obstacle/base_pose_ground_truth", Odometry, callback_feedback,queue_size=1)
    rospy.Subscriber("base_pose_ground_truth", Odometry, callback_ego_feedback,queue_size=1)
    # start_time = rospy.get_time()
    # t1 = rospy.get_time() + 10
    # print(start_time,t1)
    rospy.spin()
    r=rospy.Rate(50)
    r.sleep()

if __name__ == '__main__':
    start()
