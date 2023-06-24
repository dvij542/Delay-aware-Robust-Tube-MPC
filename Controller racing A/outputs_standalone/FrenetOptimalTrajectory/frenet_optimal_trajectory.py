# -*- coding: utf-8 -*-

#!/usr/bin/env python3
from casadi import *
import numpy as np
# import rospy
import math
import pickle
import time
# from ackermann_msgs.msg import AckermannDriveStamped
import matplotlib.pyplot as plt
import copy
import sys
import os


try:
    import FrenetOptimalTrajectory.cubic_spline_planner as cubic_spline_planner
    from FrenetOptimalTrajectory.quintic_polynomials_planner import QuinticPolynomial
except ImportError:
    raise

################# PARAMS ####################

# Parameter
speed_less_at = [5,1]
speed_increase_at = [150,5]

start_point = [-81,-12]
end_point = [100,-12]
target_speed = 50
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
MAX_SPEED = 150.0 / 3.6  # maximum speed [m/s]
MAX_ACCEL = 5.0  # maximum acceleration [m/ss]
MAX_CURVATURE = 5.0  # maximum curvature [1/m]
ROAD_WIDTH = 11.0
MAX_ROAD_WIDTH_L = 7.0  # maximum road width [m]
MAX_ROAD_WIDTH_R = 7.0  # maximum road width [m]
D_ROAD_W = 2.0  # road width sampling length [m]
DT = 0.2  # time tick [s]
MAX_T = 1.8  # max prediction time [m]
MIN_T = 1.4  # min prediction time [m]
TARGET_SPEED = 90.0 / 3.6  # target speed [m/s]
D_T_S = 1.0 / 3.6  # target speed sampling length [m/s]
N_S_SAMPLE = 1  # sampling number of target speed
ROBOT_RADIUS = 5  # robot radius [m]

# cost weights
K_J = 0.1
K_T = 0.1
K_D = 100.0
K_LAT = 1.0
K_LON = 1.0
K_obs_dist = 100.0

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
    for di in np.arange(-MAX_ROAD_WIDTH_R, MAX_ROAD_WIDTH_L, D_ROAD_W):

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
            d += [(((ix - obx)) ** 2 + (iy - oby) ** 2)]
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
            print("1")
            continue
        elif any([abs(a) > MAX_ACCEL for a in
                  fplist[i].s_dd]):  # Max accel check
            # print("2")
            # continue
            pass
        elif any([abs(c) > MAX_CURVATURE for c in
                  fplist[i].c]):  # Max curvature check
            print("3")
            continue
        cond, min_dist = check_collision(fplist[i], ob, ob_speed)
        if not cond:
            print("4")
            continue
        dists.append(1/min_dist)
        ok_ind.append(i)

    return [fplist[i] for i in ok_ind],dists

def frenet_optimal_planning(csp, s0, c_speed, c_d, c_d_d, c_d_dd, ob, ob_speed, rel_distance):
    global MAX_ROAD_WIDTH_L, MAX_ROAD_WIDTH_R
    MAX_ROAD_WIDTH_L = ROAD_WIDTH/2-rel_distance
    MAX_ROAD_WIDTH_R = ROAD_WIDTH/2+rel_distance
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
    s = np.arange(0, csp.s[-1], 1)

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = csp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(csp.calc_yaw(i_s))
        rk.append(csp.calc_curvature(i_s))

    return rx, ry, ryaw, rk, s, csp


