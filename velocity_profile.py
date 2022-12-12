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
S_rx = np.arange(-1,1,0.01)
F_rx = np.zeros((S_rx.shape[0]))
F_ry = np.zeros((S_rx.shape[0]))

angle = np.arange(0,2*math.pi,0.01)
F_fx = np.zeros((angle.shape[0]))
F_fy = np.zeros((angle.shape[0]))
for i in range(S_rx.shape[0]):
    S_r = math.sqrt(S_ry**2 + S_rx[i]**2)
    F_r = f(S_r)
    print (F_r)
    F_rx[i] = -(S_rx[i]/S_r)*F_r
    F_ry[i] = -(S_ry/S_r)*F_r

for i in range(angle.shape[0]):
    F_fx[i] = maxF*math.cos(angle[i])
    F_fy[i] = maxF*math.sin(angle[i])
    

plt.figure()
plt.grid()    
plt.plot(F_rx,F_ry)
plt.plot(F_fx,F_fy)
ax = plt.gca()
ax.set_aspect("equal", "datalim")
plt.show()