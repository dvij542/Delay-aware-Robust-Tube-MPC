# Initialization
def start():
    global inv_set
    
    inv_set = get_inv_set(T,0,2,Q_robust,R_robust,N,without_steering=True)
    with rti.open_connector(
        config_name="MyParticipantLibrary::ObstacleParticipant",
        url=file_path + "/Sensors_ego1.xml") as connector:
        input_speed = connector.get_input("StateSubscriber::stateReader")
        controls = connector.get_output("controlPublisher::controlPub")
        wait_topic = connector.get_input("simWaitSub::simWaitReader")
        done_topic = connector.get_output("simDonePub::simDoneWriter")
        output = connector.get_output("SpeedPublisher::speedPub")
        output_steering = connector.get_output("steeringPublisher::steeringPub")
        input1 = connector.get_input("roadSubscriber::roadReader1")
        input2 = connector.get_input("roadSubscriber::roadReader2")
        input_radar_F = connector.get_input("radarSubscriber_F::radarReader_F")
        input_radar_left = connector.get_input("radarSubscriber_left::radarReader_left")
        input_radar_right = connector.get_input("radarSubscriber_right::radarReader_right")
        input1 = connector.get_input("roadSubscriber::roadReader1")
        
        # Read data from the input, transform it and write it into the output
        print("Waiting for data...")
        
        #Initialise
        all_vehicles = np.ones((max_no_of_vehicles,6))*10000
        opp_vehicle_detected = np.zeros((max_no_of_vehicles),dtype = int)
        opp_vehicle_detected_state = np.zeros((max_vehicle_id),dtype = int)
        
        curr_steering = 0
        curr_speed = 0
        target_speed = 0
        aggregate = 0
        nr_dist = 0
        all_vehicles = np.ones((max_no_of_vehicles,6))*10000
        if file_path_follow != None:
            trajectory_to_follow = np.loadtxt(file_path_follow,delimiter = ",")
        else :
            trajectory_to_follow=None
        if file_centre_line != None:
            centre_line = np.loadtxt(file_centre_line,delimiter = ",")
        else :
            centre_line=None
        traj_followed = []
        
        tx, ty, tyaw, tc, csp = generate_target_course(trajectory_to_follow[:,0], trajectory_to_follow[:,1])

        itr = 0
        total_itr=0
        curr_steering_array = np.zeros((N,2))
        target_speed_array = np.array([0]*N)
        while True:
            total_itr=total_itr+1
            itr = itr+1
            if total_itr > save_path_after and save_path_after!=-1:
                break
            print("Iteration no", total_itr)
            input_speed.wait()

            wait_topic.wait()
            wait_topic.take()
            print("Waiting")
            input1.wait()
            input1.take()
            input2.wait()
            input2.take()
            input_radar_F.wait()
            input_radar_F.take()
            input_radar_left.wait()
            input_radar_left.take()
            input_radar_right.wait()
            input_radar_right.take()
            input1.take()
            radius = 0
            for sample in input1.samples.valid_data_iter:
                st10 = time.time()
                data1 = sample.get_dictionary()
                radius = 1/data1['roadLinesPolynomsArray'][0]['curvatureRadius']
                break
            
            print("Received")
            wait_msg = []
            for sample in wait_topic.samples.valid_data_iter:
                data = sample.get_dictionary()
                wait_msg = data
            
            no_of_vehicles = 0
            all_vehicles[:,:2] = 10000
            all_vehicles[:,2] = 1
            all_vehicles[:,3] = 0
            all_vehicles[:,4:6] = 10000
            opp_vehicle_detected[:] = 0

            
            # Get the opponent vehicle data
            if True :
                for sample in input_radar_F.samples.valid_data_iter:
                    data = sample.get_dictionary()
                    
                    for k in range(len(data['targetsArray'])):
                        all_vehicles[no_of_vehicles,0] = data['targetsArray'][k]['posXInChosenRef'] - 3.2
                        all_vehicles[no_of_vehicles,1] = data['targetsArray'][k]['posYInChosenRef']
                        all_vehicles[no_of_vehicles,2] = data['targetsArray'][k]['absoluteSpeedX']
                        all_vehicles[no_of_vehicles,3] = data['targetsArray'][k]['absoluteSpeedY']
                        theta_heading = data['targetsArray'][k]['posHeadingInChosenRef']
                        # all_vehicles[no_of_vehicles,2] = opp_speed*cos(theta_heading)
                        # all_vehicles[no_of_vehicles,3] = opp_speed*sin(theta_heading)
                        opp_vehicle_detected[no_of_vehicles] = data['targetsArray'][k]['scanerId']
                        all_vehicles[no_of_vehicles,0], all_vehicles[no_of_vehicles,1] \
                            = utils.anchorPointToCenter(\
                                all_vehicles[no_of_vehicles,0], \
                                all_vehicles[no_of_vehicles,1], \
                                    math.atan2(all_vehicles[no_of_vehicles,3], all_vehicles[no_of_vehicles,2]),\
                                    data['targetsArray'][k]['anchorPoint']) 
                        all_vehicles[no_of_vehicles,4] = all_vehicles[no_of_vehicles,0] + 1.791102
                        all_vehicles[no_of_vehicles,5] = all_vehicles[no_of_vehicles,1]

                        print("Vehicle no ", no_of_vehicles)
                        print("Vehicle id : ",opp_vehicle_detected[no_of_vehicles])
                        print("X : ", all_vehicles[no_of_vehicles,0])
                        print("Y : ", all_vehicles[no_of_vehicles,1])
                        print("Ego Vehicle frame X : ", all_vehicles[no_of_vehicles,4])
                        print("Ego Vehicle frame Y : ", all_vehicles[no_of_vehicles,5])
                        print("Speed X : ", all_vehicles[no_of_vehicles,2])
                        print("Speed Y : ", all_vehicles[no_of_vehicles,3])
                        print("detectionStatus :", data['targetsArray'][k]['detectionStatus'])
                        print("type :", data['targetsArray'][k]['type_'])
                        #print("name :", data['targetsArray'][k]['name'])
                        print("beamIndex :", data['targetsArray'][k]['beamIndex'])
                        print("existenceTime :", data['targetsArray'][k]['existenceTime'])
                        print("anchorPoint :", data['targetsArray'][k]['anchorPoint'])
                        print("referenceFrame :", data['targetsArray'][k]['referenceFrame'])
                        no_of_vehicles += 1
                    break
                print("From left radar")
                for sample in input_radar_left.samples.valid_data_iter:
                    data = sample.get_dictionary()
                    for k in range(len(data['targetsArray'])):
                        if(data['targetsArray'][k]['posYInChosenRef']>-2 or data['targetsArray'][k]['posXInChosenRef']<0 or data['targetsArray'][k]['posXInChosenRef']>5) :
                            continue
                        all_vehicles[no_of_vehicles,0] = -data['targetsArray'][k]['posYInChosenRef']
                        all_vehicles[no_of_vehicles,1] = data['targetsArray'][k]['posXInChosenRef']
                        all_vehicles[no_of_vehicles,2] = -data['targetsArray'][k]['absoluteSpeedY']
                        all_vehicles[no_of_vehicles,3] = data['targetsArray'][k]['absoluteSpeedX']
                        opp_vehicle_detected[no_of_vehicles] = data['targetsArray'][k]['scanerId']
                        all_vehicles[no_of_vehicles,0], all_vehicles[no_of_vehicles,1] \
                            = utils.anchorPointToCenter(\
                                all_vehicles[no_of_vehicles,0], \
                                all_vehicles[no_of_vehicles,1], \
                                    math.atan2(all_vehicles[no_of_vehicles,3], all_vehicles[no_of_vehicles,2]),\
                                    data['targetsArray'][k]['anchorPoint']) 
                        all_vehicles[no_of_vehicles,4] = all_vehicles[no_of_vehicles,0] + 0.220795
                        all_vehicles[no_of_vehicles,5] = all_vehicles[no_of_vehicles,1] + 2.064369
                        print("Vehicle no ", no_of_vehicles)
                        print("Vehicle id : ",opp_vehicle_detected[no_of_vehicles])
                        print("X : ", all_vehicles[no_of_vehicles,0])
                        print("Y : ", all_vehicles[no_of_vehicles,1])
                        print("Ego Vehicle frame X : ", all_vehicles[no_of_vehicles,4])
                        print("Ego Vehicle frame Y : ", all_vehicles[no_of_vehicles,5])
                        print("Speed X : ", all_vehicles[no_of_vehicles,2])
                        print("Speed Y : ", all_vehicles[no_of_vehicles,3])
                        print("detectionStatus :", data['targetsArray'][k]['detectionStatus'])
                        print("type :", data['targetsArray'][k]['type_'])
                        #print("name :", data['targetsArray'][k]['name'])
                        print("beamIndex :", data['targetsArray'][k]['beamIndex'])
                        print("existenceTime :", data['targetsArray'][k]['existenceTime'])
                        print("anchorPoint :", data['targetsArray'][k]['anchorPoint'])
                        print("referenceFrame :", data['targetsArray'][k]['referenceFrame'])
                        no_of_vehicles +=1
                    break
                
                print("From right radar")
                for sample in input_radar_right.samples.valid_data_iter:
                    data = sample.get_dictionary()
                    for k in range(len(data['targetsArray'])):
                        if (data['targetsArray'][k]['posXInChosenRef']<0 or data['targetsArray'][k]['posXInChosenRef']>5) :
                            continue
                        all_vehicles[no_of_vehicles,0] = data['targetsArray'][k]['posYInChosenRef']
                        all_vehicles[no_of_vehicles,1] = -data['targetsArray'][k]['posXInChosenRef']
                        all_vehicles[no_of_vehicles,2] = data['targetsArray'][k]['absoluteSpeedY']
                        all_vehicles[no_of_vehicles,3] = -data['targetsArray'][k]['absoluteSpeedX']
                        opp_vehicle_detected[no_of_vehicles] = data['targetsArray'][k]['scanerId']
                        all_vehicles[no_of_vehicles,0], all_vehicles[no_of_vehicles,1] \
                            = utils.anchorPointToCenter(\
                                all_vehicles[no_of_vehicles,0], \
                                all_vehicles[no_of_vehicles,1], \
                                    math.atan2(all_vehicles[no_of_vehicles,3], all_vehicles[no_of_vehicles,2]),\
                                    data['targetsArray'][k]['anchorPoint']) 
                        all_vehicles[no_of_vehicles,4] = all_vehicles[no_of_vehicles,0] - 0.220795
                        all_vehicles[no_of_vehicles,5] = all_vehicles[no_of_vehicles,1] + 2.064369 
                        print("Vehicle no ", no_of_vehicles)
                        print("Vehicle id : ",opp_vehicle_detected[no_of_vehicles])
                        print("X : ", all_vehicles[no_of_vehicles,0])
                        print("Y : ", all_vehicles[no_of_vehicles,1])
                        print("Ego Vehicle frame X : ", all_vehicles[no_of_vehicles,4])
                        print("Ego Vehicle frame Y : ", all_vehicles[no_of_vehicles,5])
                        print("Speed X : ", all_vehicles[no_of_vehicles,2])
                        print("Speed Y : ", all_vehicles[no_of_vehicles,3])
                        print("detectionStatus :", data['targetsArray'][k]['detectionStatus'])
                        print("type :", data['targetsArray'][k]['type_'])
                        #print("name :", data['targetsArray'][k]['name'])
                        print("beamIndex :", data['targetsArray'][k]['beamIndex'])
                        print("existenceTime :", data['targetsArray'][k]['existenceTime'])
                        print("anchorPoint :", data['targetsArray'][k]['anchorPoint'])
                        print("referenceFrame :", data['targetsArray'][k]['referenceFrame'])
                        no_of_vehicles += 1
                    break
            
            
            px = 0
            py = 0
            angle_heading = 0
            vx = 0
            vy = 0
            vz = 0
            # Get the vehicle data
            if True :
                input_speed.take()
                for sample in input_speed.samples.valid_data_iter:
                    # t1 = time.time()
                    data = sample.get_dictionary()
                    vx = data['cdgSpeed_x']
                    vy = data['cdgSpeed_y']
                    vz = data['cdgSpeed_z']
                    px = data['cdgPos_x']  
                    steering_ = data['cdgSpeed_heading']
                    w = data['cdgSpeed_heading']
                    py = data['cdgPos_y']  
                    angle_heading = data['cdgPos_heading']
                    slip_angle = data['slipAngle']
                    curr_engine_speed = data['EngineSpeed']
                    forcex = data['tireForce_x']
                    forcey = data['tireForce_y']
                    dist_covered = data['TraveledDistance']
                    normalz = data['tireForce_z']
                    angle_heading = data['cdgPos_heading']
                    slip_angle = data['slipAngle']
                    omega_dot = data['cdgAccel_heading']
                    curr_pedal = data['gasPedal']
                    curr_gear = data['GearEngaged']
                    lsr = data['LSR']
                    # diff_torque = omega_dot*pars.moment_of_inertia - ((forcey[0]+forcey[1])*pars.Lf - (forcey[2]+forcey[3])*pars.Lr)
                    # curr_mu = pars.mu_max - (pars.mu_max-pars.mu_min)/(pars.degradation_dist-dist_covered)
                    gyl = utils.get_gyk(slip_angle[2],normalz[2],3114,lsr[2])
                    gyr = utils.get_gyk(slip_angle[3],normalz[3],3114,lsr[3])
                    print("GRL and GYR :",gyl,gyr)
                    lr_ratio = forcey[1]/forcey[0]
                    diff_f = forcey[1] - utils.calc_force_from_slip_ratio(slip_angle[1],normalz[1],3114,lsr[1])
                    diff_r = forcey[3] - utils.calc_force_from_slip_ratio(slip_angle[3],normalz[3],3114,lsr[3])
                    
                    curr_speed = vx#math.sqrt(vx*vx+vy*vy+vz*vz)
                    perp_speed = vy
                    # print("Current State :",[px,py,angle_heading,curr_speed])
                    # print("Predicted State :",[predicted_x,predicted_y,predicted_theta,predicted_v])
                    if itr>10 and save_path_after!=-1:
                        itr = 0
                        traj_followed.append([px,py,curr_speed])
                    # print("Current Speed : ", curr_speed)
                    # break
            
            t1 = time.time()
            
            x_obst = px + all_vehicles[0,0]*cos(angle_heading) - all_vehicles[0,1]*sin(angle_heading)
            y_obst = py + all_vehicles[0,0]*sin(angle_heading) + all_vehicles[0,1]*cos(angle_heading) 
            vx_obst = all_vehicles[0,2]*cos(angle_heading) - all_vehicles[0,3]*sin(angle_heading)
            vy_obst = all_vehicles[0,2]*sin(angle_heading) + all_vehicles[0,3]*cos(angle_heading) 
            
            yaw = angle_heading + math.atan2(all_vehicles[0,3],all_vehicles[0,2])
            rot_mat = np.array([[math.cos(yaw),math.sin(yaw)],[math.sin(-yaw),math.cos(yaw)]])
            c_obst = np.array([[x_obst,y_obst]])
            dims = np.array([[-1,-1],[-1,1],[3,1],[3,-1]])
            obstacle_points = c_obst + np.matmul(dims,rot_mat)
            
            # Calculate path using frenet planner
            if True : 
                global curr_x,curr_y,curr_speed_x,curr_speed_y, curr_vel, last_speed_y, last_time_t, not_turned
                curr_x = px
                curr_y = py
                curr_speed_x = vx
                curr_speed_y = vy
                # omega = odom_data1.twist.twist.angular.z
                # print("Got speed ego :", curr_x,curr_y,curr_speed_x, curr_speed_y)
                # initial state
                
                c_speed = vx #min((curr_speed_x**2 + curr_speed_y**2)**0.5+1,10)  # current speed [m/s]
                _,near_point_stats = get_path(px, py, angle_heading, trajectory_to_follow.tolist(), int(2*N), speed, T)
                # TO BE CHANGED
                a,b,c,s0 = near_point_stats[0],near_point_stats[1],near_point_stats[2],near_point_stats[3]
                yaw_diff = -math.atan2(-a,b)
                c_d = c/(a**2+b**2)**(1/2) # current lateral position [m]
                c_d_d = vx*sin(yaw_diff) + vy*cos(yaw_diff)  # current lateral speed [m/s]
                c_d_dd = 0#curr_speed_x*omega  # current lateral acceleration [m/s]
                # s0 = curr_x - tx[0]  # current course position
                ob = np.array([
                            [x_obst, y_obst],
                            ])
                print("##############################", curr_vel)
                ob_speed = np.array([
                            [vx_obst, vy_obst],
                            ])
                
                # print(s0, c_speed, c_d, c_d_d, c_d_dd, ob, ob_speed)
                path = frenet_optimal_planning(
                        csp, s0, c_speed, c_d, c_d_d, c_d_dd, ob, ob_speed)
                # print("Received data!",x_obs,y_obs)
                planner_path = np.array([path.x,path.y,path.s_d]).T  
                
            # MPC call back
            if True :
                global curr_time_est, curr_time_est_var
                global time_estimates
                global gt_steering
                global has_start
                global buff_con
                
                has_start=True
                x_bot_init = px
                y_bot_init = py
                
                yaw_car_init = angle_heading # yaw in radians
                v_bot_init = vx
                print("v_bot_init :", v_bot_init)
                gt_steering = 0
                current_pose=[x_bot_init,y_bot_init,yaw_car_init,v_bot_init,gt_steering]
                
                # Shift initial state according to the current upper bound on time estimate
                if wait_time < 0 :
                    current_pose = get_future_state_est(current_pose,buff_con,curr_time_est+curr_time_est_var*beta+Td)
                else :
                    current_pose = get_future_state_est(current_pose,buff_con,wait_time)
                # print("Pose after : ", np.array(current_pose))
                
                # Inflate the obstacles according to invariant set, Z
                print(inv_set,obstacle_points,current_pose[2])
                obstacle_points_updated = obstacle_points#get_new_X_constraint_fast(inv_set,obstacle_points,current_pose[2])
                print(obstacle_points_updated)
                # IRIS constraint line for each inflated obstacle
                a_eqn, b_eqn, c_eqn, i_eqn = get_safety_line_eqn(obstacle_points_updated,current_pose[0],current_pose[1])
                
                
                # Get path 
                print(planner_path)
                ref_poses,_ = get_path(current_pose[0], current_pose[1], current_pose[2], planner_path.tolist(), int(N), speed, T)    
                # get_path(px,py,angle_heading,trajectory_to_follow.tolist(),int(2*N),speed,T)
                # End point of path for plotting
                xs=[ref_poses[-1,0],ref_poses[-1,1],0]
                ref_poses = ref_poses[:,:2]
                print(list(current_pose),list(ref_poses.reshape(2*N)),[a_eqn,b_eqn,c_eqn])
                p=list(current_pose)+list(ref_poses.reshape(2*N))+[a_eqn,b_eqn,c_eqn]
                
                # Steering and acctuator limits
                lbx=np.zeros(2*N)
                ubx=np.zeros(2*N)
                for k in range (1,2*N,2): 
                    # print("Steering limits :", steering_limits[int(v_bot_init)])
                    lbx[k] = -0.2#steering_limits[int(v_bot_init)-40]
                    ubx[k] = 0.2#steering_limits[int(v_bot_init)-40]

                for k in range (0,2*N,2): 
                    # print("Acc limits :", acc_limits[int(v_bot_init)])
                    lbx[k] = -1#acc_limits[int(v_bot_init)]/4
                    ubx[k] = 1#acc_limits[int(v_bot_init)]/4

                # Solve optimization problem
                so=solver(x0=x0,p=p,lbx=lbx,ubx=ubx,lbg=([0]*(len(vehicle_footprint)*(N+1))+[-mu*g_const]*(N+1))) 
                x=so['x']
                u = reshape(x.T,2,N).T        
                buff_con = u[:,:]
                ff_value=ff(u.T,p).T
                
                
                # for i in range(0,N,1):
                #     # Pass nominal states and commands to communicate low level pre-compensator
                #     pose3=PoseStamped()
                #     pose3.header.stamp.secs = int((2+ff_value[i,4])*1000)
                #     pose3.pose.position.x=min(40, max(-40, float(buff_con[i,1]) * 180 / math.pi))
                #     pose3.pose.position.y = t1 + curr_time_est+curr_time_est_var*beta + T*i
                #     if wait_time > 0 :
                #         pose3.pose.position.y = t1 + wait_time+ T*i
                #     pose3.pose.position.z = float(buff_con[i,0])
                #     pose3.pose.orientation.x=ff_value[i,0]
                #     pose3.pose.orientation.y=ff_value[i,1]
                #     pose3.pose.orientation.z=ff_value[i,2]
                #     pose3.pose.orientation.w=ff_value[i,3]
                #     ctrlmsg.poses.append(pose3)

                # For plotting    
                print("Published ", buff_con)
                t2 = time.time()
                if wait_time < 0 :
                    time.sleep(extra_time_to_wait + 0.0*math.sin(t2/3.5))
                else :
                    if (t2-t1)<wait_time :
                        time.sleep(wait_time-(t2-t1))

                t3 = time.time()
                new_time_est = t3-t1
                
                # Wait for remaining time to maintain consistency i.e. (upper bound time - time observed)
                if wait_time < 0 :
                    if curr_time_est+curr_time_est_var*beta-new_time_est > 0 :
                        time.sleep(curr_time_est+curr_time_est_var*beta-new_time_est)
                print("Time taken : ", new_time_est)
                out_controls = {}
                S1 = np.zeros(N).astype(float)
                S2 = np.zeros(N).astype(float)
                for i in range(N):
                    S1[i] = buff_con[i,0]
                    S2[i] = buff_con[i,1]
                # print("Commands :", buff_con)
                out_controls['speedsArray'] = S1.tolist()
                out_controls['steeringArray'] = S2.tolist()
                controls.instance.set_dictionary(out_controls)
                controls.write()
            
                time_estimates.append([curr_time_est,new_time_est,curr_time_est+curr_time_est_var*beta,t2])
                
                # Get updated computation time estimates and variance
                curr_time_est, curr_time_est_var = update_delay_time_estimate(new_time_est)
                print("Updated time estimates : ",curr_time_est, curr_time_est_var)
                print('Published at ', time.time())


if __name__ == '__main__':    
    start()
