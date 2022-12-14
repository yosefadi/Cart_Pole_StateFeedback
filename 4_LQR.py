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
from scipy import linalg
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

# State Space Equation
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

# LQR Controller
Q = np.array([[2000, 0, 0, 0],
              [0, 200, 0, 0],
              [0, 0, 100, 0],
              [0, 0, 0, 100]])

R = np.array([[0.1]])

# solve Riccati equation
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

    # print and log current state
    print("obs: ", obs)
    x_array.append(obs[0])
    x_dot_array.append(obs[1])
    theta_array.append(obs[2])
    theta_dot_array.append(obs[3])

    # get force direction (action) and force value (force)

    # MODIFY THIS PART
    action, force = apply_state_controller(K, obs)

    # absolute value, since 'action' determines the sign, F_min = -10N, F_max = 10N
    abs_force = abs(float(np.clip(force, -10, 10)))
    u_array.append(abs_force)
    
    # change magnitute of the applied force in CartPole
    env.force_mag = abs_force

    # apply action
    if legacy_api == True:
        obs, reward, done, info = env.step(action)
    else:
        obs, reward, done, truncated, info = env.step(action)

    reward_total = reward_total+reward
    print()
    if done or reward_total == reward_threshold:
        print(f'Terminated after {i+1} iterations.')
        print("reward: ", reward_total)

        u_avg = np.around(np.mean(u_array),3)
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
