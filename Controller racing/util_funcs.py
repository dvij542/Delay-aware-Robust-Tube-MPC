import params as p
import math


SC_REAR_RIGHT_EDGE = 0
SC_REAR_LEFT_EDGE = 1
SC_FRONT_RIGHT_EDGE = 2
SC_FRONT_LEFT_EDGE = 3
SC_REAR_FACE_CENTER = 4
SC_FRONT_FACE_CENTER = 5
SC_RIGHT_FACE_CENTER = 6
SC_LEFT_FACE_CENTER = 7
SC_UNUSED = 8
SC_UNKNOWN_FRONT_FACE_RIGHT = 9
SC_UNKNOWN_FRONT_FACE_LEFT = 10
SC_UNKNOWN_REAR_FACE_RIGHT = 11
SC_UNKNOWN_REAR_FACE_LEFT = 12
SC_UNKNOWN_RIGHT_FACE_FRONT = 13
SC_UNKNOWN_LEFT_FACE_FRONT = 14
SC_UNKNOWN_RIGHT_FACE_REAR = 15
SC_UNKNOWN_LEFT_FACE_REAR = 16



def inside_region_2(px,py):
    if py < -1144 and px < -240 and py > -1470:
        return True
    return False

def anchorPointToCenter(x,y,t,no) :
    xd = 0
    yd = 0
    if(no==SC_REAR_RIGHT_EDGE) :
        xd = p.L/2
        yd = p.W/2
    if(no==SC_REAR_LEFT_EDGE) :
        xd = p.L/2
        yd = -p.W/2
    if(no==SC_FRONT_RIGHT_EDGE) :
        xd = -p.L/2
        yd = p.W/2
    if(no==SC_FRONT_LEFT_EDGE) :
        xd = -p.L/2
        yd = -p.W/2
    if(no==SC_REAR_FACE_CENTER) :
        xd = p.L/2
        yd = 0
    if(no==SC_FRONT_FACE_CENTER) :
        xd = -p.L/2
        yd = 0
    if(no==SC_RIGHT_FACE_CENTER) :
        xd = 0
        yd = p.W/2
    if(no==SC_LEFT_FACE_CENTER) :
        xd = 0
        yd = -p.W/2
    if(no==SC_UNKNOWN_FRONT_FACE_RIGHT) :
        xd = -p.L/2
        yd = p.W/4
    if(no==SC_UNKNOWN_FRONT_FACE_LEFT) :
        xd = -p.L/2
        yd = -p.W/4
    if(no==SC_UNKNOWN_REAR_FACE_RIGHT) :
        xd = p.L/2
        yd = p.W/4
    if(no==SC_UNKNOWN_REAR_FACE_LEFT) :
        xd = p.L/2
        yd = -p.W/4
    if(no==SC_UNKNOWN_RIGHT_FACE_FRONT) :
        xd = -p.L/4
        yd = p.W/2
    if(no==SC_UNKNOWN_LEFT_FACE_FRONT) :
        xd = -p.L/4
        yd = -p.W/2
    if(no==SC_UNKNOWN_RIGHT_FACE_REAR) :
        xd = p.L/4
        yd = p.W/2
    if(no==SC_UNKNOWN_LEFT_FACE_REAR) :
        xd = p.L/4
        yd = -p.W/2
    
    xdd = xd*math.cos(t) - yd*math.sin(t)
    ydd = xd*math.sin(t) + yd*math.cos(t)
    return x + xdd, y + ydd

def under_ll_turn(x,y):
    if y<-2120 and x<-258 :
        return True
    else :
        return False