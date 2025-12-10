#%%
import numpy as np
import matplotlib.pyplot as plt


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data_robot = np.load("Data/robot_throwing2.npz")
time = data_robot['ts']
robot_q = data_robot['qs']
robot_dq = data_robot['qs_t']
robot_ddq = data_robot['qs_tt']
robot_u = data_robot['us']

print (time.shape,robot_q.shape , robot_ddq.shape , robot_u.shape)

data_ball = np.load("Data/ball_throwing2.npz")
ball_q = data_ball['ball_x']
ball_dq = data_ball['ball_xt']
ball_f = data_ball['f']
ball_c = data_ball['c']


print (ball_q.shape , ball_dq.shape , ball_f.shape)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def find_throwing(ball_f , rest_time):
    idx_throw = np.where(ball_f[rest_time:] < 1e-04)[0]
    flag = False
    i=0
    while flag == False:
        if idx_throw[i+10] - idx_throw[i] == 10:
            idx_throw = int(rest_time + idx_throw[i])
            flag = True
        i=i+1
    return idx_throw


def hitting_ground(ball_q):
    idx_floor = np.where(ball_q[:, 2] - 0.038 < 1e-4)[0]
    return idx_floor[0]

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
idx1 = 1
idx2 = 7
DT =0.002
idx_throw1 = find_throwing(ball_f[idx1] , rest_time = 50)
idx_throw2 = find_throwing(ball_f[idx2] , rest_time = 50)

idx_hit1 = hitting_ground(ball_q[idx1])
idx_hit2 = hitting_ground(ball_q[idx2])

print ("throwing time for trajectory" , idx1 , "is" , idx_throw1*DT)
print ("throwing time for trajectory" , idx2 , "is" , idx_throw2*DT)

print ("hitting ground time for trajectory" , idx1 , "is" , idx_hit1*DT)
print ("throwing time for trajectory" , idx2 , "is" , idx_hit2*DT)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i in range (3):
    plt.figure()
    plt.plot(time[idx1,:idx_hit1] , ball_q[idx1, :idx_hit1,i] , lw=1.6 , label = "ball position ${idx1}$" , color = 'b')
    plt.plot(time[idx2,:idx_hit2] , ball_q[idx2, :idx_hit2,i] , lw=1.6 , label = "ball position ${idx2}$" , color ='r')
    plt.axvline(time[idx1,idx_throw1], linestyle='--', linewidth=1.0 , color='b')
    plt.axvline(time[idx2,idx_throw2], linestyle='--', linewidth=1.0 , color='r')
    plt.axvline(time[idx1,idx_hit1], linestyle='--', linewidth=1.0 , color='b')
    plt.axvline(time[idx2,idx_hit2], linestyle='--', linewidth=1.0 , color='r')
    labels = ["x [m]", "y [m]", "z [m]"]
    plt.xlabel("t [s]")
    plt.ylabel(labels[i])
    plt.title("ball position")
    plt.legend()
    plt.grid(True, alpha=0.3)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

plt.figure()
plt.plot(time[idx1,:1000] , ball_f[idx1, :1000] , lw=1.6 , label = "ball force ${idx1}$" , color = 'b')
plt.plot(time[idx2,:1000] , ball_f[idx2, :1000] , lw=1.6 , label = "ball force ${idx2}$" , color = 'r')
plt.axvline(time[idx1,idx_throw1], linestyle='--', linewidth=1.0 , color='b')
plt.axvline(time[idx2,idx_throw2], linestyle='--', linewidth=1.0 , color='r')
plt.xlabel("t [s]")
plt.ylabel(rf"$fn$")
plt.title("ball force")
plt.legend()
plt.grid(True, alpha=0.3)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i in range (4):
    plt.figure()
    plt.plot(time[idx1,:idx_throw1] , robot_q[idx1, :idx_throw1,i] , lw=1.6 , label = "robot position ${idx1}$" , color = 'b' )
    plt.plot(time[idx2,:idx_throw2] , robot_q[idx2, :idx_throw2,i] , lw=1.6 , label = "robot position ${idx2}$" , color = 'r')
    plt.axvline(time[idx1,idx_throw1], linestyle='--', linewidth=1.0 , color='b')
    plt.axvline(time[idx2,idx_throw2], linestyle='--', linewidth=1.0 , color='r')
    plt.xlabel("t [s]")
    plt.ylabel(rf"$q_{i+1}$")
    plt.title("robot position")
    plt.legend()
    plt.grid(True, alpha=0.3)


# %%

