###############################################################################
# (c) Copyright, Real-Time Innovations, 2019.  All rights reserved.           #
# No duplications, whole or partial, manual or electronic, may be made        #
# without express written permission.  Any such copies, or revisions thereof, #
# must display this notice unaltered.                                         #
# This code contains trade secrets of Real-Time Innovations, Inc.             #
###############################################################################

"""Reads Squares, transforms them and writes them as Circles."""

from sys import path as sys_path
from os import path as os_path

file_path = os_path.dirname(os_path.realpath(__file__))
sys_path.append(file_path + "/../../../")
import rticonnextdds_connector as rti
import numpy as np
import math

curr_index = 0
last_index = -1
kp = 1
curr_steerings = np.zeros(20)
curr_speeds = np.zeros(20)
file_new_path = "./path_with_comp.txt" # File in which the new coordinates will be saved
save_path_after = -1
import matplotlib.pyplot as plt

with rti.open_connector(
        config_name="MyParticipantLibrary::SpeedParticipant",
        url=file_path + "/Sensors_ego1.xml") as connector:

    input = connector.get_input("StateSubscriber::stateReader")
    input_controls = connector.get_input("controlSubscriber::controlSub")
    output = connector.get_output("SpeedPublisher::speedPub")
    output_steering = connector.get_output("steeringPublisher::steeringPub")
    wait_topic = connector.get_input("simWaitSub::simWaitReader")
    done_topic = connector.get_output("simDonePub::simDoneWriter")
    
    traj_followed = []
    
    # Read data from the input, transform it and write it into the output
    print("Waiting for data...")
    input_controls.wait()
    input.wait()
    itr = 0
    while True:
        itr += 1
        print("Iteration ", itr)
        if itr > save_path_after and save_path_after!=-1:
            break
        wait_topic.wait()
        input_controls.take()
        wait_topic.take()
        wait_msg = []
        for sample in wait_topic.samples.valid_data_iter:
            data = sample.get_dictionary()
            wait_msg = data
        
        for sample in input_controls.samples.valid_data_iter:
            data = sample.get_dictionary()
            #print(data)
            curr_steerings = data['steeringArray']
            curr_speeds = data['speedsArray']
            curr_index = 0 #curr_index - last_index - 1
            last_index = curr_index
            break
        input.wait() # Wait for data in the input
        input.take()
        if curr_index > 15 :
            curr_index = 15
        for sample in input.samples.valid_data_iter:
    
            data = sample.get_dictionary()
            vx = data['cdgSpeed_x']
            vy = data['cdgSpeed_y']
            vz = data['cdgSpeed_z']
            curr_speed = math.sqrt(vx*vx+vy*vy+vz*vz)
            px = data['cdgPos_x']  
            steering_ = data['cdgSpeed_heading']
            py = data['cdgPos_y']  
            
            if itr>0 and save_path_after!=-1:
                traj_followed.append([px,py,curr_speed])
            
            out = {}
            out_steering = {}
            target_speed=curr_speeds[curr_index]
            #print(target_speed-curr_speed)
            out['AcceleratorAdditive'] = max(0,target_speed)
            out['AcceleratorMultiplicative'] = 0
            out['BrakeAdditive'] = -min(0,target_speed)*10*700
            out['BrakeMultiplicative'] = 0
            out['ClutchAdditive'] = 0
            out['ClutchMultiplicative'] = 0
            out['GearboxAutoMode'] = 1
            out['GearboxTakeOver'] = 0
            out['IsRatioLimit'] = 0
            out['MaxRatio'] = 1000
            out['MinRatio'] = 1
            out['ParkingBrakeAdditive'] = 0
            out['ParkingBrakeMultiplicative'] = 0
            out['ShiftDown'] = 0
            out['ShiftUp'] = 0
            out['WantedGear'] = 1
            
            out['TimeOfUpdate'] = data['TimeOfUpdate']
            # if curr_index==0:
            output.instance.set_dictionary(out)
            output.write()
            print("Current Index:", curr_index)
            print("Target Speed:", target_speed)
            curr_steering = curr_steerings[curr_index]
            out_steering['AdditiveSteeringWheelAngle'] = curr_steering*2.7/1.6         
            out_steering['AdditiveSteeringWheelAccel'] = 0
            out_steering['AdditiveSteeringWheelSpeed'] = 0
            out_steering['AdditiveSteeringWheelTorque'] = 0
            out_steering['MultiplicativeSteeringWheelAccel'] = 1
            out_steering['MultiplicativeSteeringWheelAngle'] = 0
            out_steering['MultiplicativeSteeringWheelSpeed'] = 1
            out_steering['MultiplicativeSteeringWheelTorque'] = 1
            out_steering['TimeOfUpdate'] = data['TimeOfUpdate']
            print("Steering Command : " , curr_steering)
            # if curr_index==0:
            output_steering.instance.set_dictionary(out_steering)
            output_steering.write()
            done_topic.instance.set_dictionary(wait_msg)
            done_topic.write()
            print("Time: ", data['TimeOfUpdate'])
            break
        curr_index += 1
    traj_followed = np.array(traj_followed)
    plt.plot(traj_followed[:,0],traj_followed[:,1],'k', lw=0.5, alpha=0.5)
    np.savetxt(file_new_path, traj_followed, delimiter=',')
    plt.show()
