import math
import numpy as np

def dist(a, x, y):
        return (((a.pose.position.x - x)**2) + ((a.pose.position.y - y)**2))**0.5

def path_length_distance(a,b):
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5

def calc_path_length(x_p):
    # global path_length
    path_length = []
    for i in range(len(x_p)):
        if i == 0:
            path_length.append(0)
        else:
            path_length.append(path_length[i-1] + path_length_distance(x_p[i], x_p[i-1]))
    return path_length

def convert_xyzw_to_rpy(x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
    
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
    
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
    
        return roll_x, pitch_y, yaw_z # in radians

# Preprocessing to get the trackable path by the vehicle (for MPC) at current speed, for N steps at T step length
def get_path(x_bot, y_bot, theta_bot, x_p, N, T) :
    path_lengths = calc_path_length(x_p)
    out_path = []
    distances = []   
    # print(len(x_p)) 
    for i in range(len(x_p)):
        a = x_p[i]
        distances += [path_length_distance([x_bot,y_bot],a)]
    # print(distances)
    
    ep = min(distances)
    total_index=len(x_p)
    cp = distances.index(ep)
    curr_dist = path_lengths[cp]
    s0 = curr_dist
    i = cp
    speed = x_p[cp][2]
    x_p += [x_p[-1][:]]
    path_lengths += [path_lengths[-1] + path_length_distance(x_p[-2],x_p[-1])]
    # In case of very sparsely present points, divide segments into multiple parts to get closest point
    new_dists = []
    if cp > 0 :
        available_length_l = path_lengths[cp] - path_lengths[cp-1]
    else :
        available_length_l = 0
    
    if cp < len(path_lengths) - 1 :
        available_length_r = path_lengths[cp+1] - path_lengths[cp]
    else :
        available_length_r = 0
    
    no_of_segs_l = int(available_length_l/(speed*T)) 
    no_of_segs_r = int(available_length_r/(speed*T)) 
    seg_len_l = available_length_l/max(no_of_segs_l,1)
    seg_len_r = available_length_r/max(no_of_segs_r,1)
    for s in range(no_of_segs_l) :
        x1,y1 = x_p[cp-1][0], x_p[cp-1][1]
        x2,y2 = x_p[cp][0], x_p[cp][1]
        xs,ys = x1 + (x2-x1)*(seg_len_l/available_length_l)*(s+1), y1 + (y2-y1)*(seg_len_l/available_length_l)*(s+1)
        new_dists += [((xs-x_bot)**2 + (ys-y_bot)**2)**0.5]
    new_dists.append(ep)
    for s in range(no_of_segs_r) :
        x1,y1 = x_p[cp][0], x_p[cp][1]
        x2,y2 = x_p[cp+1][0], x_p[cp+1][1]
        xs,ys = x1 + (x2-x1)*(seg_len_r/available_length_r)*(s+1), y1 + (y2-y1)*(seg_len_r/available_length_r)*(s+1)
        new_dists += [((xs-x_bot)**2 + (ys-y_bot)**2)**0.5]
    min_ni = new_dists.index(min(new_dists))
    if min_ni < no_of_segs_l :
        s = min_ni
        x1,y1 = x_p[cp-1][0], x_p[cp-1][1]
        x2,y2 = x_p[cp][0], x_p[cp][1]
        xs,ys = x1 + (x2-x1)*(seg_len_l/available_length_l)*(s+1), y1 + (y2-y1)*(seg_len_l/available_length_l)*(s+1)
        
        v1 = x_p[cp-1][2]
        v2 = x_p[cp][2]
        vs = v1 + (v2-v1)*(seg_len_l/available_length_l)*(s+1)
        
        c1 = x_p[cp-1][3]
        c2 = x_p[cp][3]
        cs = c1 + (c2-c1)*(seg_len_l/available_length_l)*(s+1)
        
        d1 = x_p[cp-1][4]
        d2 = x_p[cp][4]
        ds = d1 + (d2-d1)*(seg_len_l/available_length_l)*(s+1)
        
        # pose_temp=PoseStamped()
        # pose_temp.pose.position.x = xs
        # pose_temp.pose.position.y = ys
        x_p.insert(cp,[xs,ys,vs,cs,ds])
        s0 = path_lengths[cp-1] + seg_len_l*(s+1)
        path_lengths.insert(cp,path_lengths[cp-1] + seg_len_l*(s+1))
    if min_ni > no_of_segs_l :
        s = min_ni - no_of_segs_l - 1
        x1,y1 = x_p[cp][0], x_p[cp][1]
        x2,y2 = x_p[cp+1][0], x_p[cp+1][1]
        xs,ys = x1 + (x2-x1)*(seg_len_r/available_length_r)*(s+1), y1 + (y2-y1)*(seg_len_r/available_length_r)*(s+1)
        
        v1 = x_p[cp-1][2]
        v2 = x_p[cp][2]
        vs = v1 + (v2-v1)*(seg_len_r/available_length_r)*(s+1)
        
        c1 = x_p[cp-1][3]
        c2 = x_p[cp][3]
        cs = c1 + (c2-c1)*(seg_len_r/available_length_r)*(s+1)

        d1 = x_p[cp-1][4]
        d2 = x_p[cp][4]
        ds = d1 + (d2-d1)*(seg_len_r/available_length_r)*(s+1)

        x_p.insert(cp+1,[xs,ys,vs,cs,ds])
        s0 = path_lengths[cp] + seg_len_r*(s+1)
        path_lengths.insert(cp+1,path_lengths[cp] + seg_len_r*(s+1))
        cp = cp + 1
    i = cp
    
    req_dist = 0
    # Building the path
    total_length_track = path_lengths[-1]
    for j in range(N+1) :
        k = i
        no_of_rounds = 0
        while(no_of_rounds*total_length_track+path_lengths[k]-path_lengths[cp]<req_dist ) :
            k += 1
            no_of_rounds += k//len(path_lengths)
            k%=len(path_lengths)
        # print(i,k)

        if k>=len(path_lengths) :
            k = len(path_lengths) - 1
            out_path.append(np.array([x_p[k][0],x_p[k][1],x_p[k][3],x_p[k][4]]))
            req_dist += x_p[k][2]*T
            continue
        a = req_dist + path_lengths[cp] - (no_of_rounds*total_length_track+path_lengths[k-1])
        b = (no_of_rounds*total_length_track+path_lengths[k]) - req_dist - path_lengths[cp]
        X1 = np.array([x_p[k-1][0],x_p[k-1][1],x_p[k-1][2],x_p[k-1][3],x_p[k-1][4]])
        X2 = np.array([x_p[k][0],x_p[k][1],x_p[k][2],x_p[k][3],x_p[k][4]])
        X = X1*b/(a+b) + X2*a/(a+b)
        out_path.append(X)
        req_dist += X[2]*T
        if k > 0 :
            i = k-1
    # print("Path : ", out_path)
    X1 = out_path[0]
    X2 = out_path[1]
    x1_,y1_ = X1[0]-x_bot,X1[1]-y_bot
    x2_,y2_ = X2[0]-x_bot,X2[1]-y_bot
    x1 = x1_*math.cos(theta_bot) + y1_*math.sin(theta_bot)
    y1 = y1_*math.cos(theta_bot) - x1_*math.sin(theta_bot)
    x2 = x2_*math.cos(theta_bot) + y2_*math.sin(theta_bot)
    y2 = y2_*math.cos(theta_bot) - x2_*math.sin(theta_bot)
    dist = ((y1-y2)**2 + (x2-x1)**2)**(1/2)
    a = (y1-y2)/dist
    b = (x2-x1)/dist
    c = (y2*x1 - y1*x2)/dist
    return np.array(out_path[1:]),[a,b,c,s0,X1[3],(out_path[-1][3]-X1[3])/(N*speed*T)]

def sample_path(s_bot,path,speed,speed_d,N,T) :
    # print(s_bot,path.s)
    req_dist = (s_bot-path.s[0])+speed*T*path.s_d[0]/math.sqrt(path.s_d[0]**2+path.d_d[0]**2)
    i = 0
    out_path = []
    # Building the path
    for j in range(N) :
        k = i
        # print(req_dist)
        while(k<len(path.s) and path.s[k]-path.s[0]<req_dist ) :
            k += 1
        if k>=len(path.s) :
            k = len(path.s) - 1
            out_path.append(np.array([path.s[k],path.d[k],speed]))
            continue
        a = req_dist + path.s[0] - path.s[k-1]
        b = path.s[k] - req_dist - path.s[0]
        X1 = np.array([path.s[k-1]-s_bot,path.d[k-1],speed])
        X2 = np.array([path.s[k]-s_bot,path.d[k],speed])
        X = X1*b/(a+b) + X2*a/(a+b)
        # print(a,b,k,X)
        out_path.append(X)
        speed += speed_d*speed*T
        req_dist += speed*T*path.s_d[k]/math.sqrt(path.s_d[k]**2+path.d_d[k]**2)
        i = max(k-1,0)
    
    return np.array(out_path)

def get_sd_from_xy(x_p,x_bot,y_bot,T) :
    path_lengths = calc_path_length(x_p)
    distances = []   
    for i in range(len(x_p)):
        a = x_p[i]
        distances += [path_length_distance([x_bot,y_bot],a)]
    
    ep = min(distances)
    cp = distances.index(ep)
    i = cp
    speed = x_p[cp][2]
    
    # In case of very sparsely present points, divide segments into multiple parts to get closest point
    new_dists = []
    if cp > 0 :
        available_length_l = path_lengths[cp] - path_lengths[cp-1]
    else :
        available_length_l = 0
    
    if cp < len(path_lengths) - 1 :
        available_length_r = path_lengths[cp+1] - path_lengths[cp]
    else :
        available_length_r = 0
    
    no_of_segs_l = int(available_length_l/(speed*T)) 
    no_of_segs_r = int(available_length_r/(speed*T)) 
    seg_len_l = available_length_l/max(no_of_segs_l,1)
    seg_len_r = available_length_r/max(no_of_segs_r,1)
    for s in range(no_of_segs_l) :
        x1,y1 = x_p[cp-1][0], x_p[cp-1][1]
        x2,y2 = x_p[cp][0], x_p[cp][1]
        xs,ys = x1 + (x2-x1)*(seg_len_l/available_length_l)*(s+1), y1 + (y2-y1)*(seg_len_l/available_length_l)*(s+1)
        new_dists += [((xs-x_bot)**2 + (ys-y_bot)**2)**0.5]
    new_dists.append(ep)
    for s in range(no_of_segs_r) :
        x1,y1 = x_p[cp][0], x_p[cp][1]
        x2,y2 = x_p[cp+1][0], x_p[cp+1][1]
        xs,ys = x1 + (x2-x1)*(seg_len_r/available_length_r)*(s+1), y1 + (y2-y1)*(seg_len_r/available_length_r)*(s+1)
        new_dists += [((xs-x_bot)**2 + (ys-y_bot)**2)**0.5]
    min_ni = new_dists.index(min(new_dists))
    if min_ni < no_of_segs_l :
        s = min_ni
        x1,y1 = x_p[cp-1][0], x_p[cp-1][1]
        x2,y2 = x_p[cp][0], x_p[cp][1]
        xs,ys = x1 + (x2-x1)*(seg_len_l/available_length_l)*(s+1), y1 + (y2-y1)*(seg_len_l/available_length_l)*(s+1)
        
        v1 = x_p[cp-1][2]
        v2 = x_p[cp][2]
        vs = v1 + (v2-v1)*(seg_len_l/available_length_l)*(s+1)
        
        c1 = x_p[cp-1][3]
        c2 = x_p[cp][3]
        cs = c1 + (c2-c1)*(seg_len_l/available_length_l)*(s+1)
        
        d1 = x_p[cp-1][4]
        d2 = x_p[cp][4]
        ds = d1 + (d2-d1)*(seg_len_l/available_length_l)*(s+1)
        
        x_p.insert(cp,[xs,ys,vs,cs,ds])
        path_lengths.insert(cp,path_lengths[cp-1] + seg_len_l*(s+1))
    if min_ni > no_of_segs_l :
        s = min_ni - no_of_segs_l - 1
        x1,y1 = x_p[cp][0], x_p[cp][1]
        x2,y2 = x_p[cp+1][0], x_p[cp+1][1]
        xs,ys = x1 + (x2-x1)*(seg_len_r/available_length_r)*(s+1), y1 + (y2-y1)*(seg_len_r/available_length_r)*(s+1)
        
        v1 = x_p[cp-1][2]
        v2 = x_p[cp][2]
        vs = v1 + (v2-v1)*(seg_len_r/available_length_r)*(s+1)
        
        c1 = x_p[cp-1][3]
        c2 = x_p[cp][3]
        cs = c1 + (c2-c1)*(seg_len_r/available_length_r)*(s+1)

        d1 = x_p[cp-1][4]
        d2 = x_p[cp][4]
        ds = d1 + (d2-d1)*(seg_len_r/available_length_r)*(s+1)

        x_p.insert(cp+1,[xs,ys,vs,cs,ds])
        path_lengths.insert(cp+1,path_lengths[cp] + seg_len_r*(s+1))
        cp = cp + 1
    i = cp
    d = math.sqrt((x_p[i][0]-x_bot)**2+(x_p[i][1]-y_bot)**2)
    x1,y1,x2,y2 = x_p[i][0],x_p[i][1],x_p[i+1][0],x_p[i+1][1]
    math.copysign(d,float(y_bot*(x2-x1)-x_bot*(y2-y1)+y2*x1-y1*x2))
    s = path_lengths[i]
    return s,d

def get_xy_from_sd(x_p,s_bot,d_bot,T) :
    path_lengths = calc_path_length(x_p)
    i = 0
    while(path_lengths[i]<s_bot) :
        i += 1
    i -= 1
    d1 = s_bot - path_lengths[i]
    d2 = path_lengths[i+1] - s_bot
    r = d1/(d1+d2)
    x_close = x_p[i][0]*(1-r) + x_p[i+1][0]*r
    y_close = x_p[i][1]*(1-r) + x_p[i+1][1]*r
    direction = math.atan2(x_p[i+1][1]-x_p[i][1],x_p[i+1][0]-x_p[i][0]) + math.pi/2
    x_bot = x_close - d_bot*math.cos(direction)
    y_bot = y_close - d_bot*math.sin(direction)
    return x_bot,y_bot
