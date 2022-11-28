"""
References:   https://pages.jh.edu/piglesi1/Courses/454/Notes6.pdf
Implemented in python with numpy
"""

import gym
import numpy as np
import control

l = 0.5
mp = 0.1
mc = 1.0
g = 9.8
dt = 0.02  # from openai gym docs

# get environment
env = gym.make('CartPole-v1', render_mode="human").unwrapped
#env.env.seed(1)     # seed for reproducibility
obs, info = env.reset(seed=1)
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

# reorganize matrices
Ar = np.array(A, copy=True)
Ar[[0, 1, 2, 3]] = Ar[[0, 2, 1, 3]]
Ar[:, [1, 2]] = Ar[:, [2, 1]]

Br = np.array(B, copy=True)
Br[[0, 1, 2, 3]] = Br[[0, 2, 1, 3]]

Cr = np.array(C, copy=True)
Cr[:,[1,2]] = Cr[:,[2,1]]

# partitioned matrices
# a = available states
# u = unavailable states
Aaa = Ar[:2,:2]
Aau = Ar[:2,2:]
Aua = Ar[2:,:2]
Auu = Ar[2:,2:]
Ba = Br[:2]
Bu = Br[2:]

# desired poles
P = np.array([-0.25+0.25j, -0.25-0.25j, -20, -30])
Pt = 6 * P[:2]

# compute regulator and observer gains
K = control.place(A, B, P)
L = control.place(np.transpose(Auu), np.transpose(Aau), Pt)
L = np.transpose(L)

def compute_reduced_observer(x, x_hat, y, xcc, u):
    xcopy = np.array(x, copy=True)
    xa = np.empty([2,])
    xa[[0]] = xcopy[[0]]
    xa[[1]] = xcopy[[2]]
    print("xa: ", xa)

    x_hat_copy = np.array(x_hat, copy=True)
    xu_hat = np.empty([2,])
    xu_hat[[0]] = x_hat_copy[[1]]
    xu_hat[[1]] = x_hat_copy[[3]]
    print("xu_hat: ", xu_hat)

    xcc_dot = (Auu - L@Aau)@xu_hat + (Aua - L@Aaa)@y + (Bu - L@Ba)@u
    xcc = xcc + xcc_dot*dt
    xu_hat = xcc + L@y
    
    x_hat_new = np.concatenate((xa, xu_hat))
    x_hat_new[[2,1]] = x_hat_new[[1,2]]
    return xcc,x_hat_new
    
def apply_state_controller(x):
    u = -K@x   # u = -Kx
    if u > 0:
        action = 1
    else:
        action = 0 
    return action, u

obs_hat = np.zeros(4,)
xcc = np.zeros(2,)
u_array = []
theta_array = []
t_array = []

for i in range(1000):
    # time logging
    t = i*dt
    t_array.append(t)

    env.render()

    # states data logging
    print("obs_hat: ", obs_hat)
    print("obs: ", obs)
    theta_array.append(obs[2])

    # MODIFY THIS PART
    action, force = apply_state_controller(obs_hat)
    print("u:", force)
    u_array.append(force)

    # absolute value, since 'action' determines the sign, F_min = -10N, F_max = 10N
    clip_force = np.clip(force, -10, 10)
    abs_force = np.abs(float(clip_force))

    # change magnitute of the applied force in CartPole
    env.force_mag = abs_force

    # apply action
    obs, reward, done, truncated, info = env.step(action)
    y = C@obs

    # compute observer state
    xcc,obs_hat = compute_reduced_observer(obs, obs_hat, y, xcc, clip_force)
    print("obs_hat: ", obs_hat)

    print()
    reward_total = reward_total+reward
    if done or truncated or reward_total == reward_threshold:
        print(f'Terminated after {i+1} iterations.')
        print("reward: ", reward_total)

        u_array_abs = []
        for i in range(len(u_array)):
            u_array_abs.append(np.abs(u_array[i]))
        
        u_avg = np.around(np.mean(u_array_abs),3)
        print("force_avg: ", u_avg, "N")

        theta_max = np.amax(theta_array)
        theta_min = np.amin(theta_array)
        if theta_max > np.abs(theta_min):
            theta_abs = theta_max
            search_theta = theta_max
        else:
            theta_abs = np.abs(theta_min)
            search_theta = theta_min
        
        overshoot_rad = np.around(theta_abs, 3)
        overshoot_deg = np.around(np.rad2deg(theta_abs),3)
        print("overshoot: ", overshoot_deg, "degree")

        for i in range(len(theta_array)):
            if np.abs(theta_array[i]) < 1e-3:
                converge = 0
                for j in range(10):
                    if np.abs(theta_array[i+j]) < 1e-3:
                        converge = converge + 1
                if converge == 10:
                    peak_time = np.around(i * dt,3)/2
                    print("peak_time: ", peak_time, "s")
                    break

        obs, info = env.reset()
        break

env.close()