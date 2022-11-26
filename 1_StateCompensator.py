import gym
import numpy as np
import math

l = 0.5
mp = 0.1
mc = 1.0
#mt = mp+mk
g = 9.8

# get environment
env = gym.make('CartPole-v0', render_mode="human")
#env.env.seed(1)     # seed for reproducibility
obs, info = env.reset(seed=1)
reward_total = 0

# System State Equation
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

L = 10**0 * np.array([[3, 0],
                      [2, -0.7137],
                      [0, 2],
                      [0, 17.7024]])

# place the regulator pole to -1, -1+j5, -1-j5, -2
K = 10**0 * np.array([[-4.2672,-6.7291,-36.4202,-7.2910]])

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