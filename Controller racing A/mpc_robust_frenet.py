# -*- coding: utf-8 -*-

#!/usr/bin/env python3

'''
This code implements the high level Delay aware MPC controller 
for following higher level trajectory from motion planner
'''
from __future__ import print_function
from __future__ import division

from yaml import error
# Imports
if True :
    from adaptive_kalman_filter import update_delay_time_estimate
    from casadi import *
    from IRIS import *
    from inv_set_calc import *
    from adaptive_kalman_filter import *
    import matplotlib.pyplot as plt
    
    import live_plotter as lv
    import numpy as np
    import math
    # import rospy
    # import math
    from FrenetOptimalTrajectory.frenet_optimal_trajectory import *
    from sys import path as sys_path
    from os import path as os_path, times
    from beginner_tutorials.msg import custom_msg
    import rospy
    from carla_utils import *
    from utils import *

# Parameters
if True :
    T = .1     # Planning time horizon
    N = 8 # No of control intervals
    save_path_after = 55 # Start saving path after this much time from start (in s)
    wait_time = -1 # If -1, adaptive kalman filter will be used to predict it
    extra_time_to_wait = 0.05 # Wait for this additional extra time (in s) after each computation cycle to simulate and verify safety on slow systems
    Td = 0.0 # Actuator processing delay (in s)
    steer_to_wheel_ratio = (70*math.pi/180)
    has_start=False
    curr_time_est = 0  # Estimated mean value of computation time
    curr_time_est_var = 0 # Estimated variance of computation time
    lambda_ = 10
    LANE_WIDTH = 5.5
    # START_SPEED = 28
    START_SPEED = 32
    KP_OPP = 0.1
    OPP_SPEED = 15
    SIM_RATE = 50
    g_const = 9.8
    obstacle_points = np.array([[22,-18],[22,-10],[32,-10],[32,-18]]) # Static obstacle
    L = 3 # Length of the vehicle in m
    Ka = 5 # Pedal constant (F_x = Ka*mass*pedal_value)
    Kf = -0.134 # Friction resistance (F_{friction,x} = Kf*mass)
    vehicle_footprint = [[2,1],[-2,1],[-2,-1],[2,-1]] # Vehicle dimensions in m
    CONTROLLER_OUTPUT_FOLDER = './outputs_overtaking_turn'
    if not os.path.exists(CONTROLLER_OUTPUT_FOLDER) :
        os.makedirs(CONTROLLER_OUTPUT_FOLDER)
    SAVE_SUFFIX = 'without_comp'
    Q_robust = np.matrix(np.diag(np.array([1,1,1,1,1]))) 
    R_robust = np.matrix(np.diag(np.array([.1,.1])))

    ############ Calculated offline from inv_set_calc.py (Last commented part) ###############
    steering_limits = [0.5511084632063113, 0.5511084632063113, 0.5511084632063113, \
        0.5185237587941808, 0.5511084632063113, 0.5727896385850489, 0.5896766286658156, \
        0.6037252485785009, 0.616120511291855, 0.6266117297066048, 0.6266117297066048]
    acc_limits = [3.7014163903050914, 3.700715491966788, 3.7157664426919617, \
        3.7346625840889347, 3.751783194067104, 3.7654178037240746, 3.7756355027001733, \
        3.7829216295990125, 3.7880532616963265, 3.791426044016998, 3.791426044016998]
    #################### Function of speeds (step size 1 m/s) ################################

    DONT_CONSIDER_COMP_DELAY = True # If True, computation delay compensation will not be considered
    DONT_CONSIDER_STEERING_DYNAMICS = True # If True, actuator dynamic delay compensation will not be considered
    file_centre_line = "./racetrack_waypoints_reverse.txt"  # File to read the global reference line, if None then centre line will be taken
    file_path_follow = "./waypoints_new_reverse.csv"  # File to read the global reference line, if None then centre line will be taken
    file_new_path = "./coordinates_nc2.txt" # File in which the new coordinates will be saved
    Q_along=2  # Weight for progress along the road
    Q_dist=0  # Weight for distance for the center of the road
    penalty_out_of_road = 6 # Penalise for planning path outside the road
    no_iters = 3
    max_no_of_vehicles = 4
    max_vehicle_id = 10

# Global variables for internal communication (Don't change)
if True :
    buff_con = [0]*N # buffer sequence of commands
    gt_steering = 0 # Variable to communicate current value of steering angle
    inv_set = [] # Invariant set, Z
    time_estimates = []
    planned_paths = []
    time_to_finish = 0
    FIRST_TIME = True

# Definitions of optimization objective 
if True : 
    ###########   States    ####################
    Cf = 2000
    Cr = 2000
    lf = 1.5
    lr = 1.5
    mass = 1600
    moi = 1600
    e1=SX.sym('e1')
    e2=SX.sym('e2')
    e1_=SX.sym('e1\'')
    e2_=SX.sym('e2\'')
    delta_ac=SX.sym('delta_ac')
    vx=SX.sym('vx')
    Cm = SX.sym('Cm')
    Ca = SX.sym('Cm')
    s = SX.sym('s')
    states=vertcat(e1,e1_,e2,e2_,vx,delta_ac,s,Ca,Cm)
    pedal=SX.sym('pedal')
    delta=SX.sym('delta')
    controls=vertcat(pedal,delta)
    EPSILON = 0
    if DONT_CONSIDER_STEERING_DYNAMICS :
        K = 1/T
    else :
        K = (1-math.e**(-11*T))/T
    n_states=9
    n_controls=2
    U=SX.sym('U',n_controls+1,N)
    g=SX.sym('g',(1+len(vehicle_footprint))*(N+1))
    P=SX.sym('P',n_states + 3*N + 3)
    X=SX.sym('X',n_states,(N+1))
    X[:,0]=P[0:n_states]    
    # X[-1,0]=P[-4]
    ###########    Model   #####################
    if DONT_CONSIDER_STEERING_DYNAMICS : 
        rhs=[
                e1_,
                -e1_*(2*Cf+2*Cr)/(mass*vx) + e2*(2*Cf+2*Cr)/mass + e2_*(-2*Cf*lf+2*Cr*lr)/(mass*vx) + 2*delta*Cf*steer_to_wheel_ratio/mass -(vx*(Ca+Cm*s))*((2*Cf*lf-2*Cr*lr)/(mass*vx)+vx),
                e2_,
                -e1_*(2*Cf*lf-2*Cr*lr)/(moi*vx) + e2*(2*Cf*lf-2*Cr*lr)/moi -e2_*(2*Cf*lf**2+2*Cr*lr**2)/(moi*vx) + 2*Cf*lf*delta*steer_to_wheel_ratio/moi - (vx*(Ca+Cm*s))*(2*Cf*lf**2+2*Cr*lr**2)/(moi*vx),
                Ka*pedal+Kf*vx,
                K*(delta-delta_ac),
                vx*cos(e2)-(e1_-vx*e2)*sin(e2),
                0,
                0
            ]
    else :
        rhs=[
                e1_,
                -e1_*(2*Cf+2*Cr)/(mass*vx) + e2*(2*Cf+2*Cr)/mass + e2_*(-2*Cf*lf+2*Cr*lr)/(mass*vx) + 2*delta_ac*Cf*steer_to_wheel_ratio/mass -(vx*(Ca+Cm*s))*((2*Cf*lf-2*Cr*lr)/(mass*vx)+vx),
                e2_,
                -e1_*(2*Cf*lf-2*Cr*lr)/(moi*vx) + e2*(2*Cf*lf-2*Cr*lr)/moi -e2_*(2*Cf*lf**2+2*Cr*lr**2)/(moi*vx) + 2*Cf*lf*delta_ac*steer_to_wheel_ratio/moi - (vx*(Ca+Cm*s))*(2*Cf*lf**2+2*Cr*lr**2)/(moi*vx),
                Ka*pedal+Kf*vx,
                K*(delta-delta_ac),
                vx*cos(e2)-(e1_-vx*e2)*sin(e2),
                0,
                0
            ]
    rhs=vertcat(*rhs)
    f=Function('f',[states,controls],[rhs])
    # Cm = P[-4]

    for k in range(0,N,1):
        st=X[:,k]
        con=U[:2,k]
        f_value=f(st,con)
        st_next=st+(T*f_value)
        X[:,k+1]=st_next

    ############### Objective function ################

    ff=Function('ff',[U,P],[X])

    obj=0
    Q=SX([[1,0,0,0],
          [0,0,0,0],
          [0,0,10,0],
          [0,0,0,0]])
    Q_speed = 3
    Q_lane = 10
    R=SX([[10,0],
        [0,10]])

    R2=SX([[30,0],
        [0,100]])

    for k in range(0,N,1):
        st=X[:,k+1]
        con=U[:2,k]
        j = n_states + 3*k
        # obj=obj+(((st[:4]).T)@Q)@(st[:4]) + Q_speed*(st[4]-P[j+2])**2 + con.T@R@con + Q_lane*U[2,k]**2
        obj=obj + Q[0,0]*(st[0]-P[j+1])**2 + Q[2,2]*st[2]**2 + Q_speed*(st[4]-P[j+2])**2 + con.T@R@con + Q_lane*U[2,k]**2

    for k in range(0,N-1,1):
        prev_con=U[:2,k]
        next_con=U[:2,k+1]
        obj=obj+(prev_con- next_con).T@R2@(prev_con- next_con)

    OPT_variables = reshape(U[:,:],3*N,1)

    a_eq = P[-3]
    b_eq = P[-2]
    c_eq = P[-1]

    for v in range(0,len(vehicle_footprint)) :
        for k in range (0,N+1,1): 
            g[k+(N+1)*v] = a_eq*(X[6,k]+vehicle_footprint[v][0]*cos(X[2,k])\
                -vehicle_footprint[v][1]*sin(X[2,k])) + b_eq*(X[0,k]+\
                vehicle_footprint[v][0]*sin(X[2,k]) + \
                vehicle_footprint[v][1]*cos(X[2,k])) + c_eq   

    for k in range(0,N,1): 
        e1 = X[0,k]
        e1_dot = X[1,k]
        # g[k+(N+1)*len(vehicle_footprint)] = -X[4,k]**2*tan(X[2,k])/L
        g[k+(N+1)*len(vehicle_footprint)] = 1/lambda_*(e1_dot + lambda_*e1) + U[2,k]
    g[N+(N+1)*len(vehicle_footprint)] = 0
    
    nlp_prob = {'f': obj, 'x':OPT_variables, 'p': P,'g':g}
    options = {
                'ipopt.print_level' : 0,
                'ipopt.max_iter' : 150,
                'ipopt.mu_init' : 0.01,
                'ipopt.tol' : 1e-8,
                'ipopt.warm_start_init_point' : 'yes',
                'ipopt.warm_start_bound_push' : 1e-9,
                'ipopt.warm_start_bound_frac' : 1e-9,
                'ipopt.warm_start_slack_bound_frac' : 1e-9,
                'ipopt.warm_start_slack_bound_push' : 1e-9,
                'ipopt.mu_strategy' : 'adaptive',
                'print_time' : False,
                'verbose' : False,
                'expand' : True
            }

    # solver=nlpsol("solver","ipopt",nlp_prob,options)
    solver=nlpsol("solver","blocksqp",nlp_prob)

    u0=np.random.rand(N,3)
    u0[:,2] = 0
    x0=reshape(u0,3*N,1)

# Utility functions
if True :
    # Get shifted initial pose of the ego vehicle
    def get_future_state_est(current_pose,buff_con,curvature,curr_time_est) :
        # print("Pose before : ", np.array(current_pose))
        if DONT_CONSIDER_COMP_DELAY :
            return np.array(current_pose)
        itr = 0
        while (curr_time_est > T) :
            if buff_con[0] !=0 :
                f_value=f(current_pose,buff_con[itr,:])
            else :
                f_value=f(current_pose,float(buff_con[itr]))
            current_pose = list(np.array(current_pose)+np.array(T*f_value)[:,0])
            itr = itr + 1
            curr_time_est = curr_time_est - T
        if buff_con[0] !=0 :
            f_value=f(current_pose,buff_con[itr,:])
        else :
            f_value=f(current_pose,float(buff_con[itr]))# print("f_value :", f_value)
        return np.array(current_pose)+np.array(curr_time_est*f_value)[:,0]


file_path = os_path.dirname(os_path.realpath(__file__))
sys_path.append(file_path + "/../../../")

def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None
    global inv_set
    
    inv_set = get_inv_set(T,0,2,Q_robust,R_robust,N,without_steering=True)
    
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(2.0)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        # world_load = client.load_world(args.map)
        hud = HUD(args.width, args.height)
        world = World(client.get_world(), hud, args)
        controller = KeyboardControl(world, args.autopilot)
        settings = client.get_world().get_settings()
        settings.synchronous_mode = False # Enables synchronous mode
        settings.fixed_delta_seconds = 1/SIM_RATE
        client.get_world().apply_settings(settings)
        clock = pygame.time.Clock()
        
        while True:
            clock.tick_busy_loop(60)
            if controller.parse_events(client, world, clock):
                break
            world.tick(clock)
            world.render(display)
            client.get_world().tick()
            pygame.display.flip()
        
        print("Starting automated waypoint follower")
        errors_e1 = np.loadtxt('outputs/e1_maxess.csv')
        errors_e1 = errors_e1.reshape((errors_e1.shape[0],-1,2))
        lp_traj = lv.LivePlotter(tk_title="Trajectory Trace")
        FIGSIZE_X_INCHES   = 8      # x figure size of feedback in inches
        FIGSIZE_Y_INCHES   = 8      # y figure size of feedback in inches
        PLOT_LEFT          = 0.1    # in fractions of figure width and height
        PLOT_BOT           = 0.1    
        PLOT_WIDTH         = 0.8
        PLOT_HEIGHT        = 0.8
        TOTAL_EPISODE_FRAMES = 1000
        APPLY_BRAKE_AT = 0.5
        trajectory_fig = lp_traj.plot_new_dynamic_2d_figure(
                title='Vehicle Trajectory',
                figsize=(FIGSIZE_X_INCHES, FIGSIZE_Y_INCHES),
                edgecolor="black",
                rect=[PLOT_LEFT, PLOT_BOT, PLOT_WIDTH, PLOT_HEIGHT])

        waypoints = np.loadtxt('racetrack_waypoints.txt', delimiter=',')
        waypoints[:,1] = -waypoints[:,1]
        # _control = carla.VehicleControl()
        # trajectory_fig.set_invert_x_axis() # Because UE4 uses left-handed 
                                           # coordinate system the X
                                           # axis in the graph is flipped
        trajectory_fig.set_axis_equal()    # X-Y spacing should be equal in size

        # Add waypoint markers
        t = world.player.get_transform()
        t_opp = world.opponent.get_transform()
        angle_heading = t.rotation.yaw * pi/ 180
        world.player.set_velocity(carla.Vector3D(float(START_SPEED*math.cos(angle_heading)),float(START_SPEED*math.sin(angle_heading)),0))
        world.player.apply_control(carla.VehicleControl(throttle=0, brake=0, manual_gear_shift=True, gear=4))
        
        if SCENE == 'one_vehicle' :
            world.opponent.set_velocity(carla.Vector3D(float(START_SPEED*math.cos(angle_heading)),float(START_SPEED*math.sin(angle_heading)),0))
            world.opponent.apply_control(carla.VehicleControl(throttle=1, brake=0, steer=0, manual_gear_shift=True, gear=4))
        if SCENE == 'one_vehicle_turn' :
            world.opponent.set_velocity(carla.Vector3D(float(OPP_SPEED*math.cos(angle_heading)),float(OPP_SPEED*math.sin(angle_heading)),0))
            world.opponent.apply_control(carla.VehicleControl(throttle=0, brake=0, steer=0, manual_gear_shift=True, gear=4))


        start_x = t.location.x
        start_y = t.location.y
        INTERP_MAX_POINTS_PLOT = 20

        # Add trajectory markers
        trajectory_fig.add_graph("trajectory", window_size=TOTAL_EPISODE_FRAMES,
                                 x0=[start_x]*TOTAL_EPISODE_FRAMES, 
                                 y0=[start_y]*TOTAL_EPISODE_FRAMES,
                                 color=[1, 0.5, 0])
        
        trajectory_fig.add_graph("trajectory_opp", window_size=TOTAL_EPISODE_FRAMES,
                                 x0=[t_opp.location.x]*TOTAL_EPISODE_FRAMES, 
                                 y0=[t_opp.location.y]*TOTAL_EPISODE_FRAMES,
                                 color=[0.5, 0, 0.5])
        
        # Add lookahead path
        trajectory_fig.add_graph("lookahead_path", 
                                 window_size=INTERP_MAX_POINTS_PLOT,
                                 x0=[start_x]*INTERP_MAX_POINTS_PLOT, 
                                 y0=[start_y]*INTERP_MAX_POINTS_PLOT,
                                 color=[0, 0.7, 0.7],
                                 linewidth=1)
        trajectory_fig.add_graph("safety_line", 
                                 window_size=2,
                                 x0=[start_x]*2, 
                                 y0=[start_y]*2,
                                 color=[0.7, 0, 0.7],
                                 linewidth=1)
        curr_speed = 0
        all_vehicles = np.ones((max_no_of_vehicles,6))*10000
        if file_path_follow != None:
            trajectory_to_follow = np.loadtxt(file_path_follow,delimiter = ",")
        else :
            trajectory_to_follow=None
        trajectory_to_follow[:,1] = -trajectory_to_follow[:,1]
        if file_centre_line != None:
            centre_line = np.loadtxt(file_centre_line,delimiter = ",")
        else :
            centre_line=None
        centre_line[:,1] = -centre_line[:,1]
        
        tx, ty, tyaw, tc, ts, csp = generate_target_course(trajectory_to_follow[:,0], trajectory_to_follow[:,1])
        tx_center, ty_center, tyaw_center, tc_center, ts_center, csp_center = generate_target_course(centre_line[:,0], centre_line[:,1])
        if SCENE == 'one_vehicle_turn' :
            left_boundary = np.array([tx_center-3.5*np.sin(tyaw_center),ty_center+3.5*np.cos(tyaw_center),[OPP_SPEED]*len(tx_center),tyaw_center,tc_center]).T
        trajectory_fig.add_graph("waypoints", window_size=len(ty),
                                 x0=tx, y0=ty,
                                 linestyle="-", marker="", color='g')
        trajectory_fig.add_graph("left_boundary", window_size=len(ty_center),
                                 x0=tx_center-6.5*np.sin(tyaw_center), y0=ty_center+6.5*np.cos(tyaw_center),
                                 linestyle="-", marker="", color='r')
        trajectory_fig.add_graph("right_boundary", window_size=len(ty_center),
                                 x0=tx_center+6.5*np.sin(tyaw_center), y0=ty_center-6.5*np.cos(tyaw_center),
                                 linestyle="-", marker="", color='r')
        
        itr = 0
        total_itr=0
        trajectory_fig.add_graph("car", window_size=1, 
                                 marker="s", color='b', markertext="Car",
                                 marker_text_offset=1)
        trajectory_fig.add_graph("car_shifted", window_size=1, 
                                 marker="s", color='g', markertext="Car",
                                 marker_text_offset=1)
        trajectory_fig.add_graph("car_opp", window_size=1, 
                                 marker="s", color='b', markertext="Car_opp",
                                 marker_text_offset=1)
        lp_1d = lv.LivePlotter(tk_title="Controls Feedback")
        forward_speed_fig =\
                lp_1d.plot_new_dynamic_figure(title="Forward Speed (m/s)")
        forward_speed_fig.add_graph("forward_speed", 
                                    label="forward_speed", 
                                    window_size=TOTAL_EPISODE_FRAMES)
        forward_speed_fig.add_graph("reference_signal", 
                                    label="reference_Signal", 
                                    window_size=TOTAL_EPISODE_FRAMES)
        started_at = hud.simulation_time
        # debug_helper = carla.DebugHelper()
        while True:
            t1 = hud.simulation_time-started_at
            if SCENE == 'one_vehicle' :
                if t1 > APPLY_BRAKE_AT :
                    print("BBBBBBBBBBBBBBBBBBBBBrakeddddddddddddddddddddddddddd")
                    world.opponent.apply_control(carla.VehicleControl(throttle=0, brake=1, steer=0, manual_gear_shift=True, gear=4))
        
            lp_traj.refresh()
            lp_1d.refresh()
            # clock.tick_busy_loop(60)
            total_itr=total_itr+1
            itr = itr+1
            if total_itr > save_path_after and save_path_after!=-1:
                break
            print("Iteration no", total_itr)
            
            t = world.player.get_transform()
            v = world.player.get_velocity()
            if SCENE == 'one_vehicle' or SCENE == 'one_vehicle_turn' :
                t_opp = world.opponent.get_transform()
                v_opp = world.opponent.get_velocity()
                x_obst = t_opp.location.x
                y_obst = t_opp.location.y
                vx_obst = v_opp.x
                vy_obst = v_opp.y
                yaw_obst = t_opp.rotation.yaw * pi/ 180
            else :
                # DUMMY
                x_obst = 1000
                y_obst = 1000
                vx_obst = 1
                vy_obst = 0
            
            if SCENE == 'one_vehicle_turn' :
                ref_poses,_ = get_path(x_obst, y_obst, yaw_obst, left_boundary.tolist(), int(2), 0.5)
                target_x,target_y = ref_poses[-1][0],ref_poses[-1][1]
                dx = (target_x-x_obst)*cos(yaw_obst) + (target_y-y_obst)*sin(yaw_obst)
                dy = -(target_x-x_obst)*sin(yaw_obst) + (target_y-y_obst)*cos(yaw_obst)
                steer_angle = (2*dy*L)/dx**2
                world.opponent.apply_control(carla.VehicleControl(throttle=KP_OPP*(OPP_SPEED-math.sqrt(vx_obst**2+vy_obst**2)), brake=0, steer=steer_angle, manual_gear_shift=True, gear=4))

            px = t.location.x
            py = t.location.y
            angle_heading = t.rotation.yaw * pi/ 180
            vx = v.x
            vy = v.y

            yaw_obst = 0
            rot_mat = np.array([[math.cos(yaw_obst),math.sin(yaw_obst)],[math.sin(-yaw_obst),math.cos(yaw_obst)]])
            s_obst,t_obst = get_sd_from_xy(trajectory_to_follow.tolist(),x_obst,y_obst,T)
            # s_obst += math.sqrt(vx_obst**2+vy_obst**2)*N*T
            c_obst = np.array([[s_obst,t_obst]])
            dims = np.array([[-2,-1],[-2,1],[2,1],[2,-1]])
            obstacle_points = c_obst + np.matmul(dims,rot_mat)
            # Calculate path using frenet planner
            trajectory_fig.roll("trajectory", px, py)
            trajectory_fig.roll("trajectory_opp", x_obst, y_obst)
            trajectory_fig.roll("car", px, py)
            trajectory_fig.roll("car_opp", x_obst, y_obst)
            
            if True : 
                global curr_x,curr_y,curr_speed_x,curr_speed_y, last_speed_y, last_time_t, not_turned, curr_time_est, curr_time_est_var
                curr_x = px
                curr_y = py
                curr_speed_x = vx*cos(angle_heading) + vy*sin(angle_heading)
                curr_speed_y = vy*cos(angle_heading) - vx*sin(angle_heading)
                ref_poses,near_point_stats = get_path(px, py, angle_heading, trajectory_to_follow.tolist(), int(2*N), T)
                
                # TO BE CHANGED
                a,b,c,s0,curvature,curvature_m = near_point_stats[0],near_point_stats[1],near_point_stats[2],near_point_stats[3],-near_point_stats[4],-near_point_stats[5]
                # print("Got s0 as ", s0)
                yaw_diff = -math.atan2(-a,b)
                c_d = c/((a**2+b**2)**(1/2)) # current lateral position [m]
                c_d_d = math.sqrt(vx**2 + vy**2)*sin(yaw_diff)  # current lateral speed [m/s]
                c_d_dd = 0#curr_speed_x*omega  # current lateral acceleration [m/s]
                e2 = math.atan2(a,b)
                e1 = c/(math.sqrt(a**2+b**2))
                e1_ = curr_speed_y + curr_speed_x*e2
                e2_ = world.player.get_angular_velocity().z*math.pi/180 - curr_speed_x*curvature
                c_speed = curr_speed_x*cos(e2) - curr_speed_y*sin(e2)
                c_d_center = -ref_poses[0][4]
                ob = np.array([
                            [x_obst, y_obst],
                            ])
                ob_speed = np.array([
                            [vx_obst*0.9, vy_obst*0.9],
                            ])
                if not DONT_CONSIDER_COMP_DELAY :
                    # s0 += c_speed*(curr_time_est+beta*curr_time_est_var+Td)
                    print("Increased s0 by ",c_speed*(curr_time_est+beta*curr_time_est_var+Td), c_speed,curr_time_est+beta*curr_time_est_var)
                path = frenet_optimal_planning(
                        csp, s0, c_speed, c_d, c_d_d, c_d_dd, ob, ob_speed,-c_d_center)
                if path is None :
                    path = copy.deepcopy(prev_path)
                
                prev_path = copy.deepcopy(path)
                
                # Trying to plot position arrow
                world.world.debug.draw_line(carla.Location(px,py,t.location.z),carla.Location(px+vx*T*2,py+vy*T*2,t.location.z),life_time=curr_time_est)
            
            trajectory_fig.update("lookahead_path", 
                        path.x,
                        path.y,
                        new_colour=[0, 0.7, 0.7])
                
            # MPC call back
            if True :
                # global curr_time_est, curr_time_est_var
                global time_estimates
                global gt_steering
                global has_start
                global buff_con
                
                has_start=True
                gt_steering = world.player.get_angular_velocity().z*(math.pi/180)*L/(curr_speed_x+0.1)
                curr_speed = math.sqrt(curr_speed_x**2+curr_speed_y**2)
                current_pose=[e1,e1_,e2,e2_,curr_speed_x+0.1,gt_steering,0,curvature,curvature_m]
                # Shift initial state according to the current upper bound on time estimate
                # print("Current state :- e1:",current_pose[0]," e1_:",current_pose[1]," e2:",current_pose[2]," e2_:",current_pose[3]," vx:",current_pose[4]," steering:",current_pose[5]," curvature:",current_pose[6])
                
                # print(current_pose)
                if wait_time < 0 :
                    current_pose = get_future_state_est(current_pose,buff_con,curvature,curr_time_est+curr_time_est_var*beta+Td)
                else :
                    current_pose = get_future_state_est(current_pose,buff_con,curvature,wait_time)
                # print("Pose after : ", np.array(current_pose))
                # print(current_pose)
                # print("Shifted state :- e1:",current_pose[0]," e1_:",current_pose[1]," e2:",current_pose[2]," e2_:",current_pose[3]," vx:",current_pose[4]," steering:",current_pose[5]," curvature:",current_pose[6])
                # s0 += current_pose[6]
                trajectory_fig.roll("car_shifted", px, py)
                forward_speed_fig.roll("forward_speed", 
                                       hud.simulation_time-started_at, 
                                       curr_speed)
                forward_speed_fig.roll("reference_signal", 
                                       hud.simulation_time-started_at, 
                                       ref_poses[0,2])
                
                planner_path = sample_path(s0,path,ref_poses[0,2],(ref_poses[-1,2]-ref_poses[0,2])/(ref_poses[0,2]*T*(2*N)),N,T)
                # Inflate the obstacles according to invariant set, Z
                current_pose[6] = 0
                obstacle_points[:,0] -= s0#get_new_X_constraint_fast(inv_set,obstacle_points,current_pose[2])
                a_eqn, b_eqn, c_eqn, i_eqn = get_safety_line_eqn(obstacle_points,current_pose[6],current_pose[0])
                
                obstacle_points[:,0] += s0#get_new_X_constraint_fast(inv_set,obstacle_points,current_pose[2])
                s_along, d_along = obstacle_points[i_eqn,0],obstacle_points[i_eqn,1]
                angle_along = math.atan2(-a_eqn,b_eqn)
                s1,d1 = s_along + 10*math.cos(angle_along), d_along + 10*math.sin(angle_along) 
                s2,d2 = s_along - 10*math.cos(angle_along), d_along - 10*math.sin(angle_along) 
                x1,y1 = get_xy_from_sd(trajectory_to_follow.tolist(),s1,d1,T)
                x2,y2 = get_xy_from_sd(trajectory_to_follow.tolist(),s2,d2,T)
                trajectory_fig.update("safety_line", 
                        [x1,x2],
                        [y1,y2],
                        new_colour=[0.7, 0, 0.7])
                ref_poses[:,:2] = 0
                p=list(current_pose)+list(planner_path[:N,:3].reshape(3*N))+[a_eqn,b_eqn,c_eqn]
                
                # Steering and acctuator limits
                lbx=np.zeros(3*N)
                ubx=np.zeros(3*N)
                for k in range (1,3*N,3): 
                    # print("Steering limits :", steering_limits[int(v_bot_init)])
                    lbx[k] = -0.25#steering_limits[int(v_bot_init)-40]
                    ubx[k] = 0.25#steering_limits[int(v_bot_init)-40]

                for k in range (0,3*N,3): 
                    # print("Acc limits :", acc_limits[int(v_bot_init)])
                    lbx[k] = -1#acc_limits[int(v_bot_init)]/4
                    ubx[k] = 1#acc_limits[int(v_bot_init)]/4

                for k in range (2,3*N,3): 
                    # print("Acc limits :", acc_limits[int(v_bot_init)])
                    lbx[k] = -math.inf#acc_limits[int(v_bot_init)]/4
                    ubx[k] = math.inf#acc_limits[int(v_bot_init)]/4

                # Solve optimization problem
                errs = errors_e1[min(2,int(abs(curvature)//0.018)+1)]
                i = 0
                while(i<(errs.shape[0]-1) and current_pose[4]>errs[i,0]) :
                    i += 1
                lw = LANE_WIDTH - errs[i,1]
                lr_list = list(-(lw+ref_poses[:(N+1),4]))
                ll_list = list(lw-ref_poses[:(N+1),4])
                print("Debug : ",current_pose,[a_eqn,b_eqn,c_eqn])
                so=solver(x0=x0,p=p,lbx=lbx,ubx=ubx,lbg=([0]*(len(vehicle_footprint)*(N+1))+lr_list),ubg=([math.inf]*(len(vehicle_footprint)*(N+1))+ll_list)) 
                x=so['x']
                u = reshape(x.T,3,N).T
                buff_con = u[:,:2]
                ff_value=ff(u.T,p).T
                
                msg = custom_msg()
                msg.times = [0]*N
                for i in range(0,N,1):
                    msg.times[i] = t1 + curr_time_est+curr_time_est_var*beta + T*i
                
                # For plotting    
                t2 = hud.simulation_time
                if wait_time < 0 :
                    time.sleep(extra_time_to_wait + 0.0*math.sin(t2/3.5))
                else :
                    if (t2-t1)<wait_time :
                        time.sleep(wait_time-(t2-t1))

                t3 = hud.simulation_time
                new_time_est = t3-t1
                curr_time_est, curr_time_est_var = update_delay_time_estimate(min(new_time_est,0.1))
                print("Updated estimates : ", curr_time_est, curr_time_est_var)
                time_estimates.append([curr_time_est,min(new_time_est,0.1),curr_time_est+curr_time_est_var*beta,t2])

                # Wait for remaining time to maintain consistency i.e. (upper bound time - time observed)
                if wait_time < 0 :
                    if curr_time_est+curr_time_est_var*beta-new_time_est > 0 :
                        time.sleep(curr_time_est+curr_time_est_var*beta-new_time_est)
                print("Time taken : ", new_time_est)
                msg.steerings = np.array(buff_con[:,1]).astype(float)
                msg.throttles = np.array(buff_con[:,0]).astype(float)
                msg.e1 = np.array(ff_value[:,0]).astype(float)
                msg.e1_ = np.array(ff_value[:,1]).astype(float)
                msg.e2 = np.array(ff_value[:,2]).astype(float)
                msg.e2_ = np.array(ff_value[:,3]).astype(float)
                msg.vx = np.array(ff_value[:,4]).astype(float)
                msg.delta = np.array(ff_value[:,5]).astype(float)
            
            # world.player.apply_control(_control)
            world.tick(clock)
            pub.publish(msg)
            client.get_world().tick()
            world.render(display)
            pygame.display.flip()
        
        if DONT_CONSIDER_COMP_DELAY :
            suffix = '_without_comp'
        else :
            suffix = '_with_comp'
        store_trajectory_plot(trajectory_fig.fig, 'trajectory'+suffix+'.png')
        store_trajectory_plot(forward_speed_fig.fig, 'forward_speed'+suffix+'.png')
        names_to_save = ['trajectory','left_boundary','right_boundary',"trajectory_opp"]
        
        for name in names_to_save :
            outp = trajectory_fig.get_data(name)
            np.savetxt(CONTROLLER_OUTPUT_FOLDER+'/'+name+suffix+'.csv',outp[-save_path_after:])
        names_to_save = ["forward_speed","reference_signal"]
        for name in names_to_save :
            outp = forward_speed_fig.get_data(name)
            np.savetxt(CONTROLLER_OUTPUT_FOLDER+'/'+name+suffix+'.csv',outp[-save_path_after:])
        np.savetxt(CONTROLLER_OUTPUT_FOLDER+'/comp_times'+suffix+'.csv',np.array(time_estimates))


    finally:

        if (world and world.recording_enabled):
            client.stop_recorder()

        if world is not None:
            world.destroy()

        pygame.quit()

def create_controller_output_dir(output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

def store_trajectory_plot(graph, fname):
    """ Store the resulting plot.
    """
    create_controller_output_dir(CONTROLLER_OUTPUT_FOLDER)

    file_name = os.path.join(CONTROLLER_OUTPUT_FOLDER, fname)
    graph.savefig(file_name)

# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "model3")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)
    global pub
    pub = rospy.Publisher('chatter', custom_msg, queue_size=1)
    rospy.init_node('talker', anonymous=True)
    
    try:
        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()

