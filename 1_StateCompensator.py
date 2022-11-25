import gym
import numpy as np
import math

lp = 0.5
mp = 0.1
mk = 1.0
mt = mp+mk
g = 9.8

# get environment
env = gym.make('CartPole-v0', render_mode="human")
#env.env.seed(1)     # seed for reproducibility
obs, info = env.reset(seed=1)
reward_total = 0

# System State Equation
A = np.array([[0, 1, 0, 0],
              [0, 0, -0.7137, 0],
              [0, 0, 0, 1],
              [0, 0, 15.7024, 0]])

B = np.array([[0],
              [0.8426],
              [0],
              [-1.4634]])

C = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]])

L = 10**3 * np.array([[0.1285, -0.0654],
                      [3.7010, -2.8048],
                      [0.0063, 0.0715],
                      [0.3737, 1.1376]])

# place the regulator pole to -10, -10+j5, -10-j5, -20
K = 10**3 * np.array([[-2.0515,-0.6360,-1.8240,-0.4003]])

dt = 0.02

def compute_state_estimator(A, B, C, L, x_hat, y, u, dt):
    x_hat_dot = A@x_hat + B@u + L@(y-C@x_hat)
    x_hat_new = x_hat + x_hat_dot*dt
    return x_hat_new

def apply_state_controller(K, x):
    # feedback controller
    # MODIFY THIS PARTS
    u = -K@x   # u = -Kx
    #print(u)
    return u

obs_hat = obs
for i in range(1000):
    env.render()

    # get force direction (action) and force value (force)

    # MODIFY THIS PART
    force = apply_state_controller(K, obs_hat)
    if force > 0:
        action = 1
    else:
        action = 0 

    # absolute value, since 'action' determines the sign, F_min = -10N, F_max = 10N
    clip_force = np.clip(force, -10, 10)
    abs_force = abs(float(clip_force))

    y = C@obs
    obs_hat = compute_state_estimator(A, B, C, L, obs_hat, y, clip_force, dt)
    y_hat = C@obs_hat
    error = y - y_hat
    print(error)

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