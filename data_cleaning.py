#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import numpy as np
from helping_function import find_throwing

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data_robot = np.load("Data/Initial_data/second dataset/robot_throwing.npz")
time = data_robot['ts']
robot_q = data_robot['qs']
robot_dq = data_robot['qs_t']
robot_ddq = data_robot['qs_tt']
robot_u = data_robot['us']

print (time.shape,robot_q.shape , robot_ddq.shape , robot_u.shape)

data_ball = np.load("Data/Initial_data/second dataset/ball_throwing.npz")
ball_q = data_ball['ball_x']
ball_dq = data_ball['ball_xt']
ball_f = data_ball['f']
ball_c = data_ball['c']


print (ball_q.shape , ball_dq.shape , ball_f.shape)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N_total, T, _ = robot_q.shape
rest_time = 50
idx_delete = []
count_non = 0
count_leave_ball = 0
count_noise = 0
for k in range(N_total):
    idx = find_throwing(ball_c[k] , rest_time)
    if idx is None:
        count_non += 1
        idx_delete.append(k)
    else:
        if ball_dq[k , idx ,2] <= 0.1:
            count_leave_ball +=1
            idx_delete.append(k)
        else:
            if np.mean(ball_c[k , rest_time:idx] )< 0.90:
                count_noise +=1
                idx_delete.append(k)


idx_delete =np.array(idx_delete)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

print ("total number of initial samples is :" , N_total)
print ("number of Non throwing samples is :" , count_non)
print ("number of leaving ball instead of throwing  is :" , count_leave_ball)
print ("number of throw happend but ball was too noisy in cup  is :" , count_noise)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
time = np.delete(time , idx_delete , axis=0)
robot_q = np.delete(robot_q , idx_delete , axis=0)
robot_dq = np.delete(robot_dq , idx_delete , axis=0)
robot_ddq = np.delete(robot_ddq , idx_delete , axis=0)
robot_u = np.delete(robot_u , idx_delete , axis=0)

print (time.shape,robot_q.shape , robot_ddq.shape , robot_u.shape)

ball_q = np.delete(ball_q , idx_delete , axis=0)
ball_dq = np.delete(ball_dq , idx_delete , axis=0)
ball_f = np.delete(ball_f , idx_delete , axis=0)
ball_c = np.delete(ball_c , idx_delete , axis=0)

print (ball_q.shape , ball_dq.shape , ball_f.shape)
# %%

np.savez('Data/clean_data/second dataset/robot_throwing.npz', time =time , robot_q =robot_q , 
                              robot_dq = robot_dq , robot_ddq = robot_ddq , robot_u = robot_u)

np.savez('Data/clean_data/second dataset/ball_throwing.npz' , ball_q =ball_q , 
                              ball_dq = ball_dq , ball_f = ball_f , ball_c = ball_c)
# %%
