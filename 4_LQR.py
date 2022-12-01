import sys
if sys.version_info < (3,7,0):
    print("Please use python 3.7.0 or higher")
    sys.exit(1)
    
import gym
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

l = 0.5
mp = 0.1
mc = 1.0
g = 9.8
dt = 0.02

# get environment
env = gym.make('CartPole-v1', render_mode="human").unwrapped
#env.env.seed(1)     # seed for reproducibility
obs, info = env.reset(seed=1)
reward_threshold = 475
reward_total = 0

# ADD SOMETHING HERE
A = np.array([[0, 1, 0, 0],
              [0, 0, -mp*(mp * (g-l) + mc*g)/((mc+mp)*((4/3) * mc + (1/3) * mp)), 0],
              [0, 0, 0, 1],
              [0, 0, (mp*(g-l) + mc * g)/(l*((4/3) * mc + (1/3) * mp)), 0]])

B = np.array([[0],
              [(1/(mc + mp) - mp/((mc + mp) * ((4/3) * mc + (1/3) * mp)))],
              [0],
              [(-1/(l * ((4/3) * mc + (1/3) * mp)))]])

C = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]])

Q = np.array([[200, 0, 0, 0],
              [0, 200, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 100]])

R = np.array([[0.1]])

P = linalg.solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R)@np.transpose(B)@P

# compute statenum
statenum = A.shape[0]

def apply_state_controller(K, x):
    # feedback controller
    # MODIFY THIS PARTS
    u = -np.dot(K, x)   # u = -Kx
    print("u: ", u)
    if u > 0:
        action = 1
    else:
        action = 0
    return action, u

u_array = []
theta_array = []
t_array = []
x_array = []

for i in range(1000):
    # time logging
    t = i*dt
    t_array.append(t)

    env.render()

    # print and log current state
    print("obs: ", obs)
    x_array.append(obs[0])
    theta_array.append(obs[2])

    # get force direction (action) and force value (force)

    # MODIFY THIS PART
    action, force = apply_state_controller(K, obs)
    u_array.append(force)

    # absolute value, since 'action' determines the sign, F_min = -10N, F_max = 10N
    abs_force = abs(float(np.clip(force, -10, 10)))
    
    # change magnitute of the applied force in CartPole
    env.force_mag = abs_force

    # apply action
    obs, reward, done, truncated, info = env.step(action)

    reward_total = reward_total+reward
    print()
    if done or truncated or reward_total == reward_threshold:
        print(f'Terminated after {i+1} iterations.')
        print("reward: ", reward_total)

        u_array_abs = []
        for i in range(len(u_array)):
            u_array_abs.append(np.abs(u_array[i]))
        u_avg = np.around(np.mean(u_array_abs),3)
        print("u_avg: ", u_avg, "N")

        # plot 
        subplots = []
        for i in range(statenum-2):
            fig, ax = plt.subplots()
            subplots.append(ax)

        subplots[0].plot(t_array, x_array)
        subplots[0].set_title(f"x")
        subplots[0].set_xlabel("time (s)")
        subplots[0].set_ylabel("x")

        subplots[1].plot(t_array, theta_array)
        subplots[1].set_title(f"theta")
        subplots[1].set_xlabel("time (s)")
        subplots[1].set_ylabel("radians")

        obs, info = env.reset()
        break

env.close()
plt.show(block=True)
