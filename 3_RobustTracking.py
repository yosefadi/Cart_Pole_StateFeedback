"""
    PSKM - State Compensator
    Author:
        - Yosef Adi Sulistyo
        - Andreas Ryan C.K.
        - Bonaventura Riko K.D.
        - Rafli Priyo Utomo
        - Muhammad Bagus H.
"""

# Check Python version
# Requires Python 3.6 or later
import sys
if sys.version_info < (3,6,0):
    print("Please use python 3.6.0 or higher")
    sys.exit(1)

# Version Checking Function
def versiontuple(v):
    return tuple(map(int, (v.split("."))))

# Try using Gymnasium (updated fork of OpenAI Gym) for the environment
# if not installed, fallback to OpenAI Gym Standard Module
try:
    import gymnasium as gym
    print("Using Gymnasium API.")
    legacy_api = False
except ImportError:
    import gym
    print("Gymnasium not installed. Using OpenAI Gym API instead.")
    if versiontuple(gym.__version__) < (0,23,0):
        print("Enabling compat API for OpenAI Gym version < 0.23.0")
        legacy_api = True
    else:
        legacy_api = False
    pass

# Import necessary module
import numpy as np
import control
import matplotlib.pyplot as plt

# Environment Variable
l = 0.5
mp = 0.1
mc = 1.0
g = 9.8
dt = 0.02

# get environment
if legacy_api == True:
    env = gym.make('CartPole-v0').unwrapped
    env.seed(1)
    obs = env.reset()
else:
    env = gym.make('CartPole-v0', render_mode="human").unwrapped
    obs, info = env.reset(seed=1)

reward_threshold = 200
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

C = np.array([[1, 0, 0, 0]])

# Augmented SS Equation for Robust Tracking
A_aug = np.block([[np.zeros([C.shape[0],C.shape[0]]), C],
                  [np.zeros([A.shape[0],C.shape[0]]), A]])
print("A_aug: ", A_aug)

B_aug = np.block([[np.zeros([C.shape[0],1])],
                  [B]])
print("B_aug: ", B_aug)

B_L = np.array(B_aug, copy=True)

# noise/disturbance 
w = np.array([0.5])
w = np.reshape(w,1)

# desired pole
P = np.array([-0.25+0.5j, -0.25-0.5j, -10, -20])
P_aug = np.array([-2+0.5j,-2-0.5j,-1.75+0.25j,-1.75-0.25j,-100])

# compute regulator gain
K = control.place(A,B,P)
K_aug = control.place(A_aug, B_aug, P_aug)

# compute statenum
statenum = A.shape[0]

def f_aug_linear(x, u):
    x_aug_dot = A_aug@x + B_aug@u + B_L@w
    return x_aug_dot

def apply_state_controller(x):
    # feedback controller
    # MODIFY THIS PARTS
    if(x.shape[0] == A_aug.shape[0]):
        K_cont = K_aug
    else:
        K_cont = K
    
    u = -K_cont @ x
    print("u: ", u)
    return u

obs_aug = np.block([[np.zeros([C.shape[0],1])],
                          [obs.reshape([4,1])]])
obs_aug = np.reshape(obs_aug, obs_aug.shape[0])
force = np.zeros([1,])

u_array = []
x_array = []
x_dot_array = []
theta_array = []
theta_deg_array = []
theta_dot_array = []
theta_dot_deg_array = []
t_array = []

for i in range(1000):
    # time logging
    t = i*dt
    t_array.append(t)

    env.render()
    
    # log state
    x_array.append(obs[0])
    x_dot_array.append(obs[1])
    theta_array.append(obs[2])
    theta_dot_array.append(obs[3])
    print("obs: ", obs)
    print("obs_aug: ", obs_aug)

    # MODIFY THIS PART
    force = apply_state_controller(obs_aug)

    # input noise
    force = force + w

    # determine action
    if force > 0:
        action = 1
    else:
        action = 0

    # absolute value, since 'action' determines the sign, F_min = -10N, F_max = 10N
    abs_force = abs(float(np.clip(force, -10, 10)))
    
    # log force
    u_array.append(abs_force)

    # change magnitute of the applied force in CartPole
    env.force_mag = abs_force

    # apply action
    if legacy_api == True:
        obs, reward, done, info = env.step(action)
    else:
        obs, reward, done, truncated, info = env.step(action)

    obs_aug_dot = f_aug_linear(obs_aug, force)
    obs_aug = obs_aug + obs_aug_dot * dt

    for n in range(obs.shape[0]):
        obs_aug[n+C.shape[0]] = obs[n]

    reward_total = reward_total+reward
    print()
    if done or reward_total == reward_threshold:
        print(f'Terminated after {i+1} iterations.')
        print("reward: ", reward_total)

        x_array_abs = []
        for i in range(len(x_array)):
            x_array_abs.append(abs(x_array[i]))
        
        ess_avg = np.around(np.mean(x_array_abs),3)
        print("ess: ", ess_avg)

        u_array_abs = []
        for i in range(len(u_array)):
            u_array_abs.append(abs(u_array[i]))
        u_avg = np.around(np.mean(u_array_abs),3)
        print("u_avg: ", u_avg, "N")

        for i in range(len(theta_array)):
            theta_deg_array.append(np.rad2deg(theta_array[i]))
            theta_dot_deg_array.append(np.rad2deg(theta_dot_array[i]))

        # plot 
        subplots = []
        for i in range(statenum):
            fig, ax = plt.subplots()
            subplots.append(ax)

        subplots[0].plot(t_array, x_array)
        subplots[0].set_title("x")
        subplots[0].set_xlabel("time (s)")
        subplots[0].set_ylabel("x")

        subplots[1].plot(t_array, x_dot_array)
        subplots[1].set_title("x dot")
        subplots[1].set_xlabel("time (s)")
        subplots[1].set_ylabel("dx/dt")
        
        subplots[2].plot(t_array, theta_deg_array)
        subplots[2].set_title("theta")
        subplots[2].set_xlabel("time (s)")
        subplots[2].set_ylabel("degree")

        subplots[3].plot(t_array, theta_dot_deg_array)
        subplots[3].set_title("theta dot")
        subplots[3].set_xlabel("time (s)")
        subplots[3].set_ylabel("dtheta/dt (deg/s)")

        if legacy_api == True:
            obs = env.reset()
        else:
            obs, info = env.reset()
        break

env.close()
plt.show(block=True)
