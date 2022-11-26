import gym
import numpy as np
import math

l = 0.5
mp = 0.1
mc = 1.0
g = 9.8
dt = 0.02  # from openai gym docs

# get environment
env = gym.make('CartPole-v1', render_mode="human")
#env.env.seed(1)     # seed for reproducibility
obs, info = env.reset(seed=1)
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


# place the regulator pole to -1, -0.5+i, -0.5-i, -7
K = 10**0 * np.array([[-0.7180,-1.3951,-22.2476,-6.9532]])

# place estimator pole to -6,-0.5+i,-0.5-i,-42
# 6 times faster than regulator pole
L = 10**0 * np.array([[6.0296, -4.1877],
                      [1.9812, -47.4969],
                      [-1.6359, 42.9704],
                      [4.7417, 62.7275]])

def compute_state_estimator(A, B, C, L, x_hat, y, u, dt):
    x_hat_dot = A@x_hat + B@u + L@(y - C@x_hat)
    return x_hat_dot

def apply_state_controller(K, x):
    # feedback controller
    # MODIFY THIS PARTS
    #print(x)
    u = -K@x   # u = -Kx
    #print(u)
    return u

obs_hat = np.zeros(4)
print(obs_hat)
iter = 0
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
    abs_force = np.abs(float(clip_force))

    # change magnitute of the applied force in CartPole
    env.force_mag = abs_force

    # apply action
    obs, reward, done, truncated, info = env.step(action)

    y = C@obs
    obs_hat_dot = compute_state_estimator(A, B, C, L, obs_hat, y, clip_force, dt)
    obs_hat = obs_hat + obs_hat_dot*dt
    y_hat = C@obs_hat
    error = obs - obs_hat
    #print(error)

    iter = iter+1

    reward_total = reward_total+reward
    if done or truncated:
        print(f'Terminated after {i+1} iterations.')
        print(reward_total)
        obs, info = env.reset()
        break

env.close()