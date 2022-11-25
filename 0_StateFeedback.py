import gym
import numpy as np
import math

lp = 0.5
mp = 0.1
mk = 1.0
mt = mp+mk
g = 9.8

# get environment
env = gym.make('CartPole-v1', render_mode="human")
#env.env.seed(1)     # seed for reproducibility
obs, info = env.reset(seed=1)
reward_total = 0

# System State Equation
A = np.array([[0, 1, 0, 0],
              [0, 0, -0.72, 0],
              [0, 0, 0, 1],
              [0, 0, 15.77, 0]])
B = np.array([[0],
              [0.98],
              [0],
              [-1.46]])

C = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]])

# place the regulator pole to -10, -10+j5, -10-j5, -20
K = 10**3 * np.array([[-1.7357,-0.5381,-1.8094,-0.3954]])

def apply_state_controller(K, x):
    # feedback controller
    # MODIFY THIS PARTS
    u = -K@x   # u = -Kx
    print(u)
    return u

for i in range(1000):
    env.render()
    
    # get force direction (action) and force value (force)

    # MODIFY THIS PART
    force = apply_state_controller(K, obs)
    
    force = apply_state_controller(K, obs)
    if force > 0:
        action = 1
    else:
        action = 0

    # absolute value, since 'action' determines the sign, F_min = -10N, F_max = 10N
    abs_force = abs(float(np.clip(force, -10, 10)))
    
    # change magnitute of the applied force in CartPole
    env.force_mag = abs_force

    # apply action
    obs, reward, done, truncated, info = env.step(action)

    reward_total = reward_total+reward
    if done or truncated:
        print(f'Terminated after {i+1} iterations.')
        print(reward_total)
        obs, info = env.reset()
        break

env.close()