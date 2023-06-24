import params as p
import math
from casadi import *

# Consts
pcy1 = 1.603
pdy1 = 1.654
pdy2 = -0.1783
pdy3 = 0
pey1 = -1.409
pey2 = -1.6617
pey3 = 0.26886
pey4 = -13.61
pky1 = -53.05
pky2 = 4.1265
pky3 = 1.5016
phy1 = 0.0039
phy2 = -0.00253
pvy1 = -0.01038
pvy2 = -0.1
pvy3 = 0.4498
pvy4 = -1.5

rby1 = 35.304
rby2 = 15.666
rby3 = -0.01179

rcy1 = 1.018
rey1 = 0.35475
rey2 = 0.01966
rhy1 = 0.00667
rhy2 = 0.00207

rvy1 = 0.0426
rvy2 = 0.03077
rvy3 = 0.37305
rvy4 = 100
rvy5 = 2.2
rvy6 = 25

ssz1 = -0.09916
ssz2 = 0.025876
ssz3 = 2.2703
ssz4 = -1.8657

qhz1 = 0.015524
qhz2 = -0.009173
qhz3 = -0.07129
qhz4 = 0.034226

pkx1 = 63.75
pkx2 = -15
pkx3 = 0.2891

qdz1 = 0.11496
qdz2 = 0.005418
qdz3 = 1.5
qdz4 = 0
qdz6 = -0.00023155
qdz7 = -0.02192
qdz8 = -1.3554
qdz9 = 0.119

qbz1 = 12.457
qbz2 = -0.04661
qbz3 = 0
qbz4 = 5.096
qbz5 = 4.664
qbz9 = 6.924
qbz10 = 0

qcz1 = 1.6

qez1 = -10
qez2 = 5.742
qez3 = 0
qez4 = 1.3084
qez5 = 1.2514


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

def get_gyk(slip_angle,fz,fz0,slip_ratio):
    epsilon = 0.000
    dfz = (fz-fz0)/fz0
    shy = phy1 + phy2*dfz
    ky = slip_angle + shy
    Cy = pcy1
    muy = pdy1 + pdy2*dfz
    # print("muy :", muy)
    Dy = muy*fz
    Ey = (pey1 + pey2*dfz)*(1+pey3*(2*(ky<0)-1))
    K = fz0*pky1*sin(2*atan(fz/(pky2*fz0)))#+pky2*dfz)#*exp(pky3*dfz)
    By=K/(Cy*Dy+epsilon)
    svy = fz*(pvy1+pvy2*dfz)
    fy0 = (Dy*sin(Cy*atan(By*ky- Ey*(By*ky-atan(By*ky)))) + svy)
    # return Fy

    # const req rvy1, rvy2, rvy3,  rvy4,  rvy5, rvy6, lvyka, lyka
    #variable req gamma(maybe 0), alpha(slip_angle), k(slip_ratio),
    gamma = 0
    Dvyk = muy*fz*(rvy1+rvy2*dfz+rvy3*gamma)*cos(atan(rvy4*slip_angle))
    Svyk = Dvyk*sin(rvy5*atan(rvy6*slip_ratio))
    Shyk = rhy1+rhy2*dfz
    Eyk = rey1+rey2*dfz
    Cyk = rcy1
    Byk = rby1*cos(atan(rby2*(slip_angle-rby3)))
    ks = slip_ratio+Shyk
    Gyk0 = cos(Cyk*atan(Byk*Shyk-Eyk*(Byk*Shyk-atan(Byk*Shyk))))
    Gyk = cos(Cyk*atan(Byk*ks-Eyk*(Byk*ks-atan(Byk*ks))))/Gyk0
    # print(Gyk,Svyk)
    Fy = Gyk*fy0+Svyk
    return Gyk

def calc_force_from_slip_ratio(slip_angle,fz,fz0,slip_ratio):
    epsilon = 0.000
    dfz = (fz-fz0)/fz0
    shy = phy1 + phy2*dfz
    ky = slip_angle + shy
    Cy = pcy1
    muy = pdy1 + pdy2*dfz
    # print("muy :", muy)
    Dy = muy*fz
    Ey = (pey1 + pey2*dfz)*(1+pey3*(2*(ky<0)-1))
    K = fz0*pky1*sin(2*atan(fz/(pky2*fz0)))#+pky2*dfz)#*exp(pky3*dfz)
    By=K/(Cy*Dy+epsilon)
    svy = fz*(pvy1+pvy2*dfz)
    fy0 = (Dy*sin(Cy*atan(By*ky- Ey*(By*ky-atan(By*ky)))) + svy)
    # return Fy

    # const req rvy1, rvy2, rvy3,  rvy4,  rvy5, rvy6, lvyka, lyka
    #variable req gamma(maybe 0), alpha(slip_angle), k(slip_ratio),
    gamma = 0
    Dvyk = muy*fz*(rvy1+rvy2*dfz+rvy3*gamma)*cos(atan(rvy4*slip_angle))
    Svyk = Dvyk*sin(rvy5*atan(rvy6*slip_ratio))
    Shyk = rhy1+rhy2*dfz
    Eyk = rey1+rey2*dfz
    Cyk = rcy1
    Byk = rby1*cos(atan(rby2*(slip_angle-rby3)))
    ks = slip_ratio+Shyk
    Gyk0 = cos(Cyk*atan(Byk*Shyk-Eyk*(Byk*Shyk-atan(Byk*Shyk))))
    Gyk = cos(Cyk*atan(Byk*ks-Eyk*(Byk*ks-atan(Byk*ks))))/Gyk0
    # print(Gyk,Svyk)
    Fy = Gyk*fy0+Svyk
    return p.road_coeff*Fy

def calc_force_from_slip_angle(slip_angle,fz,fz0,curr_road_coeff):
    epsilon = 0.000
    dfz = (fz-fz0)/fz0
    # dfz=0
    shy = phy1 + phy2*dfz
    ky = slip_angle + shy
    Cy = pcy1
    muy = pdy1 + pdy2*dfz
    # print("muy :", muy)
    Dy = muy*fz
    Ey = (pey1 + pey2*dfz)*(1+pey3)
    K = fz0*pky1*sin(2*atan(fz/(pky2*fz0)))#+pky2*dfz)#*exp(pky3*dfz)
    By=K/(Cy*Dy+epsilon)
    svy = fz*(pvy1+pvy2*dfz)
    # print(slip_angle,fz,dfz,By*ky)
    Fy = Dy*sin(Cy*atan(By*ky - Ey*(By*ky-atan(By*ky)))) + svy
    return curr_road_coeff*Fy

# def calc_force_from_slip(slip,speed) :
#     fz = p.fz0 + p.lift_coeff*speed**2
#     dfz = (fz-p.fz0)/p.fz0
#     shx = phx1 + phx2*dfz
#     kx = slip + shx
#     Cx = pcx1
#     mux = pdx1 + pdx2*dfz
#     Dx = mux*fz
#     Ex = (pex1 + pex2*dfz + pex3*dfz**2)*(1-pex4)
#     K = (fz+dfz*fz)*(pkx1+pkx2*dfz)*np.exp(pkx3*dfz)
#     Bx=K/(Cx*Dx+epsilon)
#     svx = fz*(pvx1+pvx2*dfz)
#     Fx = Dx*np.sin(Cx*np.arctan(Bx*kx - Ex*(Bx*kx-np.arctan(Bx*kx)))) + svx
#     return Fx


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