import math
import numpy as np
import matplotlib.pyplot as plt

R1 = 300
R2 = 314

left_lane = []
right_lane = []
vehicle_pos = [[-10,3],[-7,3],[-7,1.5],[-10,1.5]]
opp_pos = [[7,-3],[10,-3],[10,-1.5],[7,-1.5]]

for theta in np.arange(-0.15,0.15,0.0001) :
    t = -math.pi/2 + theta
    x1,y1 = R1*math.cos(t), R1*math.sin(t)
    x2,y2 = R2*math.cos(t), R2*math.sin(t)
    left_lane.append([x1,y1])
    right_lane.append([x2,y2])

def get_coords(d,e) :
    t = -math.pi/2 + (d/(307))
    R = 307 - e
    return R*math.cos(t), R*math.sin(t)

def draw_line(x1,y1,x2,y2) :
    line_ = []
    for s in np.arange(0,1,0.01) :
        d,e = x1*(1-s) + x2*s, y1*(1-s) + y2*s
        x,y = get_coords(d,e)
        line_.append([x,y])
    line_ = np.array(line_)
    plt.plot(line_[:,0],line_[:,1])
    
def draw_rect(points) :
    for i in range(4) :
        j = (i+1)%4
        draw_line(points[i][0],points[i][1],points[j][0],points[j][1])

left_lane = np.array(left_lane)
right_lane = np.array(right_lane)
plt.plot(left_lane[:,0],left_lane[:,1])
plt.plot(right_lane[:,0],right_lane[:,1])
draw_rect(vehicle_pos)
draw_rect(opp_pos)
draw_line(-15,-7,40,7)

plt.axis('equal')
plt.show()