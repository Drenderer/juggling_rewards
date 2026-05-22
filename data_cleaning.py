#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import numpy as np
from helping_function import find_throwing , hitting

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data_robot = np.load("Data/Initial_data/third dataset/robot_throwing.npz")
time = data_robot['ts']
robot_q = data_robot['qs']
robot_dq = data_robot['qs_t']
robot_ddq = data_robot['qs_tt']
robot_u = data_robot['us']

print (time.shape,robot_q.shape , robot_ddq.shape , robot_u.shape)

data_ball = np.load("Data/Initial_data/third dataset/ball_throwing.npz")
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
count_not_hitting = 0
for k in range(N_total):
    idx = find_throwing(ball_c[k] , rest_time)
    idx_hit = hitting(ball_c[k] , idx)
    if idx is None:
        count_non += 1
        idx_delete.append(k)
    else:
        if ball_dq[k , idx ,2] <= 0.1:
            count_leave_ball +=1
            idx_delete.append(k)
        else:
            if np.mean(ball_c[k , rest_time:idx] )< 0.92:
                count_noise +=1
                idx_delete.append(k)
            else:
                if idx_hit is None:
                    count_not_hitting +=1
                    idx_delete.append(k)

idx_delete =np.array(idx_delete)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

print ("total number of initial samples is :" , N_total)
print ("number of Non throwing samples is :" , count_non)
print ("number of leaving ball instead of throwing  is :" , count_leave_ball)
print ("number of throw happend but ball was too noisy in cup  is :" , count_noise)
print ("number of sample which not hittig somewhere is :" , count_not_hitting)
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


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
train_ratio = 0.8
N = robot_q.shape[0]
rng = np.random.default_rng(seed=42)   
perm = rng.permutation(N)
N_train = int(train_ratio * N)

train_idx = perm[:N_train]
test_idx  = perm[N_train:]

time_train = time[train_idx]
robot_train_q = robot_q[train_idx]
robot_train_dq = robot_dq[train_idx]
robot_train_ddq = robot_ddq[train_idx]
robot_train_u = robot_u[train_idx]
ball_train_q = ball_q[train_idx]
ball_train_dq = ball_dq[train_idx]
ball_train_f = ball_f[train_idx]
ball_train_c = ball_c[train_idx]


time_test = time[test_idx]
robot_test_q = robot_q[test_idx]
robot_test_dq = robot_dq[test_idx]
robot_test_ddq = robot_ddq[test_idx]
robot_test_u = robot_u[test_idx]
ball_test_q = ball_q[test_idx]
ball_test_dq = ball_dq[test_idx]
ball_test_f = ball_f[test_idx]
ball_test_c = ball_c[test_idx]

print (robot_train_q.shape , robot_test_q.shape)



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

np.savez('Data/clean_data/third dataset/robot_throwing_train.npz', time =time_train , robot_q =robot_train_q , 
                              robot_dq = robot_train_dq , robot_ddq = robot_train_ddq , robot_u = robot_train_u)

np.savez('Data/clean_data/third dataset/ball_throwing_train.npz' , ball_q =ball_train_q , 
                              ball_dq = ball_train_dq , ball_f = ball_train_f , ball_c = ball_train_c)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

np.savez('Data/clean_data/third dataset/robot_throwing_test.npz', time =time_test , robot_q =robot_test_q , 
                              robot_dq = robot_test_dq , robot_ddq = robot_test_ddq , robot_u = robot_test_u)

np.savez('Data/clean_data/third dataset/ball_throwing_test.npz' , ball_q =ball_test_q , 
                              ball_dq = ball_test_dq , ball_f = ball_test_f , ball_c = ball_test_c)
# %%
