# -*- coding: utf-8 -*-

#!/usr/bin/env python3

'''
This code publishes the dummy path in dummypath.pkl (obtained
from Hybrid A* motion planner) 
'''

# Imports
if True :
    from casadi import *

    import rospy
    import math
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import Path
    import pickle
    import time
    from nav_msgs.msg import Path
    # from ackermann_msgs.msg import AckermannDriveStamped

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
    rospy.init_node('dummy_path_publisher', anonymous=True)
    pub1 = rospy.Publisher('astroid_path', Path,queue_size=1000)
    pub2 = rospy.Publisher('hybrid_astar_goal', PoseStamped,queue_size=1000)
    # rospy.Subscriber("hybrid_astar_goal_dummy", PoseStamped, goalCallback,queue_size=1)
    # rospy.Subscriber("hybrid_astar_goal_dummy", PoseStamped, goalCallback,queue_size=1)
    # rospy.spin()
    with open('Delay aware Robust Tube MPC/Controller A/dummypath.pkl', 'rb') as file:
        pathdata = pickle.load(file)
    with open('Delay aware Robust Tube MPC/Controller A/dummygoal.pkl', 'rb') as file:
        goaldata = pickle.load(file)
    r=rospy.Rate(20)
    while(1) :
        pathdata.header.stamp = rospy.Time.now()
        goaldata.header.stamp.secs = 0
        goaldata.header.stamp.nsecs = 0
        pub1.publish(pathdata)
        time.sleep(0.02)
        print("Published")
        r.sleep()

if __name__ == '__main__':
    start()