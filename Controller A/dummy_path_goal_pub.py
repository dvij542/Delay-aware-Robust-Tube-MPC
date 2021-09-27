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

def pathCallback(pathdata):

    pathdata.header.stamp = rospy.Time.now()
    with open('dummypath.pkl', 'wb') as file:
        pickle.dump(pathdata,file)

def goalCallback(goaldata):
    
    goaldata.header.stamp = rospy.Time.now()
    with open('dummygoal.pkl', 'wb') as file:
        pickle.dump(goaldata,file)

def start():
    global pub1
    global pub2
    # print(f([1,2,3],[1,2]))
    rospy.init_node('dummy_path_publisher', anonymous=True)
    pub1 = rospy.Publisher('astroid_path', Path,queue_size=1000)
    pub2 = rospy.Publisher('hybrid_astar_goal', PoseStamped,queue_size=1000)
    # rospy.Subscriber("hybrid_astar_goal_dummy", PoseStamped, goalCallback,queue_size=1)
    rospy.Subscriber("hybrid_astar_goal_dummy", PoseStamped, goalCallback,queue_size=1)
    # rospy.spin()
    with open('Delay aware Robust Tube MPC/Controller A/dummypath.pkl', 'rb') as file:
        pathdata = pickle.load(file)
    with open('Delay aware Robust Tube MPC/Controller A/dummygoal.pkl', 'rb') as file:
        goaldata = pickle.load(file)
    r=rospy.Rate(20)
    first_time = True
    while(1) :
        pathdata.header.stamp = rospy.Time.now()
        goaldata.header.stamp.secs = 0
        goaldata.header.stamp.nsecs = 0
        pub1.publish(pathdata)
        # if first_time :
            # pub2.publish(goaldata)
            # first_time = False
        time.sleep(0.02)
        print("Published")
        r.sleep()

start()