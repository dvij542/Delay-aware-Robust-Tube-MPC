import math
import numpy

def get_line_through(x1,y1,x2,y2) :
    return (y2-y1), (x1-x2), (y1*x2-y2*x1)

def get_line_through_norm(x1,y1,x2,y2) :
    a,b,c = get_line_through(x1,y1,x2,y2)
    norm = math.sqrt(a**2+b**2)
    return a/norm, b/norm, c/norm

def get_line_perp(x1,y1,x2,y2) :
    return (x1-x2), (y1-y2), (y2**2+x2**2-y1*y2-x1*x2)

def get_line_perp_norm(x1,y1,x2,y2) :
    a,b,c = get_line_perp(x1,y1,x2,y2)
    norm = math.sqrt(a**2+b**2)
    return a/norm, b/norm, c/norm

def on_right_of(line,x,y) :
    return (line[0]*x+line[1]*y+line[2])>0

def get_region_side(points,x,y) :
    lines = []
    for i in range(4) :
        j = (i+1)%4
        line = get_line_through(points[i][0],points[i][1],points[j][0],points[j][1])
        lines.append(line)
    
    for i in range(4) :
        j = (i+1)%4
        k = (i+2)%4
        # print(i,on_right_of(lines[i],x,y))
        if on_right_of(lines[i],x,y) and on_right_of(lines[j],x,y) and on_right_of(lines[k],x,y) :
            return i
    
    for i in range(4) :
        j = (i+1)%4
        k = (i+2)%4
        # print(i,"hahahaha")
        if not on_right_of(lines[i],x,y) and on_right_of(lines[j],x,y) :
            return i+4

# Assumptions : Points are in clockwise direction
def get_safety_line_eqn(points,x,y) :
    reg_i = get_region_side(points,x,y)
    if reg_i<4 :
        i = reg_i
        j = (i+3)%4
        a,b,c = get_line_through_norm(points[i][0],points[i][1],points[j][0],points[j][1])
        return a,b,c,i
    else : 
        i = reg_i - 4
        # print("Returned value of i :", i)
        a,b,c = get_line_perp_norm(x,y,points[i][0],points[i][1])
        # print(a,b,c)
        return a,b,c,i
