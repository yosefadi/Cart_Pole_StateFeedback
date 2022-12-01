import sys
if sys.version_info < (3,7,0):
    print("Please use python 3.7.0 or higher")
    sys.exit(1)

import gym
import numpy as np
import control
import matplotlib.pyplot as plt

l = 0.5
mp = 0.1
mc = 1.0
g = 9.8
dt = 0.02  # from openai gym docs

# get environment
env = gym.make('CartPole-v1', render_mode="human").unwrapped
#env.env.seed(1)     # seed for reproducibility # only works for old version of openai gym
obs, info = env.reset(seed=1) # specify seed for latest version of openai gym
reward_threshold = 475
reward_total = 0

# System State Space Equation
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

# create matrices for state estimator computation
# using duality principle
At = np.transpose(A)
Bt = np.transpose(C)
Ct = np.transpose(B)

# compute statenum
statenum = A.shape[0]

# desired pole
P = np.array([-0.25+0.5j, -0.25-0.5j, -20+0.25j, -20-0.25j])
Pt = 4*P

# compute regulator and observer gain
K = control.place(A, B, P)
L = control.place(At, Bt, Pt)
L = np.transpose(L)

def compute_state_estimator(x_hat, x, u):
    y = C@x
    x_hat_dot = A@x_hat + B@u + L@(y - C@x_hat)
    x_hat = x_hat_dot * dt + x_hat
    return x_hat

def apply_state_controller(x):
    u = -K@x   # u = -Kx
    if u > 0:
        action = 1
    else:
        action = 0 
    return action, u

obs_hat = np.zeros(statenum,)
print(obs_hat)
u_array = []
x_array = []
theta_array = []
theta_deg_array = []
t_array = []

for i in range(1000):
    # time logging
    t = i*dt
    t_array.append(t)

    env.render()

    # states data logging
    print("obs: ", obs)
    print("obs_hat: ", obs_hat)
    x_array.append(obs[0])
    theta_array.append(obs[2])

    # MODIFY THIS PART
    action, force = apply_state_controller(obs_hat)
    print("u: ", force)
    
    # absolute value, since 'action' determines the sign, F_min = -10N, F_max = 10N
    clip_force = np.clip(force, -10, 10)
    abs_force = np.abs(float(clip_force))

    # log absolute force for plotting
    u_array.append(abs_force)

    # change magnitute of the applied force in CartPole
    env.force_mag = abs_force

    # apply action
    obs, reward, done, truncated, info = env.step(action)

    # compute state estimator
    obs_hat = compute_state_estimator(obs_hat, obs, clip_force)

    print()
    reward_total = reward_total+reward
    if done or truncated or reward_total == reward_threshold:
        print(f'Terminated after {i+1} iterations.')
        print("reward: ", reward_total)

        u_avg = np.around(np.mean(u_array),3)
        print("force_avg: ", u_avg, "N")

        for i in range(len(theta_array)):
            theta_deg_array.append(np.rad2deg(theta_array[i]))

        # plot 
        subplots = []
        for i in range(statenum-2):
            fig, ax = plt.subplots()
            subplots.append(ax)

        subplots[0].plot(t_array, x_array)
        subplots[0].set_title(f"x")
        subplots[0].set_xlabel("time (s)")
        subplots[0].set_ylabel("x")

        subplots[1].plot(t_array, theta_deg_array)
        subplots[1].set_title(f"theta")
        subplots[1].set_xlabel("time (s)")
        subplots[1].set_ylabel("deg")

        obs, info = env.reset()
        break

env.close()
plt.show(block=True)
