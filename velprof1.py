import numpy as np
import matplotlib.pyplot as plt
import math
import scipy

lengthRear = 1.5

B = 10
C = 1.9
D = 1
load = 4000

def f(x): return load*D*math.sin(C*math.atan(B*x))
maxFunc = scipy.optimize.fmin(lambda x: -f(x), 0)
print(maxFunc.shape)
maxS = maxFunc[0]
maxF = f(maxS)

v_i = 15
beta = math.pi/2
theta_dot = 0.01

S_ry = -0.05 #((v_i*math.sin(beta))-(theta_dot*lengthRear))/(v_i*math.cos(beta))


angle = np.arange(0,2*math.pi,0.01)
rear = np.zeros((angle.shape[0]))
F_rx = np.zeros((angle.shape[0]))
F_ry = np.zeros((angle.shape[0]))
F_fx = np.zeros((angle.shape[0]))
F_fy = np.zeros((angle.shape[0]))
F_gx = np.zeros((angle.shape[0]))
F_gy = np.zeros((angle.shape[0]))

for i in range(angle.shape[0]):
    F_rx[i] = 0
    F_ry[i] = 0
    if (math.sin(angle[i] != 0 or math.tan(angle[i]) != 0)):
        S_r = S_ry/math.sin(angle[i])
        S_rx = S_ry/math.tan(angle[i])
        if(abs(S_r) <= 1):
            F_r = f(S_r)
            F_rx[i] = -(S_rx/S_r)*F_r
            F_ry[i] = -(S_ry/S_r)*F_r
    F_fx[i] = maxF*math.cos(angle[i])
    F_fy[i] = maxF*math.sin(angle[i])

for i in range(0,int(angle.shape[0]/2)):
    F_gx[i] = F_fx[i] + F_rx[i]
    F_gy[i] = F_fy[i] + F_ry[i]
for i in range(int(angle.shape[0]/2),int(angle.shape[0])):
    F_gx[i] = F_fx[i] + F_rx[i-int(angle.shape[0]/2)]
    F_gy[i] = F_fy[i] + F_ry[i-int(angle.shape[0]/2)]
plt.figure()
plt.grid()    
plt.plot(F_rx,F_ry,":")
plt.plot(F_fx,F_fy,":")
plt.plot(F_gx,F_gy)
ax = plt.gca()
ax.set_aspect("equal", "datalim")
plt.show()