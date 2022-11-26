import gym
import numpy as np
import control

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

At = np.transpose(A)
Bt = np.transpose(C)
Ct = np.transpose(B)

K = control.place(A, B, [-4, -0.5+1j, -0.5-1j, -11])
L = control.place(At,Bt, [-24, -3+6j, -3-6j, -66])
L = np.transpose(L)

"""
# place the regulator pole to -2, -0.5+i, -0.5-i, -9
K = 10**0 * np.array([[-1.8464,-2.6055,-32.4639,-9.7001]])

# place estimator pole to -12,-0.5+i,-0.5-i,-54
# 6 times faster than regulator pole
L = 10**0 * np.array([[11.9567, -2.8508],
                      [0.8236, -58.2869],
                      [-1.0877, 55.0433],
                      [13.2195, 75.0915]])
"""

def compute_state_estimator(A, B, C, L, x_hat, x, u, dt):
    y = C@x
    x_hat_dot = A@x_hat + B@u + L@(y - C@x_hat)
    return x_hat_dot

def apply_state_controller(K, x):
    u = -K@x   # u = -Kx
    if u > 0:
        action = 1
    else:
        action = 0 
    return action, u

obs_hat = np.zeros(4)
print(obs_hat)
iter = 0
for i in range(1000):
    env.render()

    # MODIFY THIS PART
    action, force = apply_state_controller(K, obs_hat)
    print("u:", force)

    # absolute value, since 'action' determines the sign, F_min = -10N, F_max = 10N
    clip_force = np.clip(force, -10, 10)
    abs_force = np.abs(float(clip_force))

    # change magnitute of the applied force in CartPole
    env.force_mag = abs_force

    # apply action
    obs, reward, done, truncated, info = env.step(action)
    print("obs: ", obs)

    # compute state estimator
    obs_hat_dot = compute_state_estimator(A, B, C, L, obs_hat, obs, clip_force, dt)
    obs_hat = obs_hat + obs_hat_dot*dt
    print("obs_hat: ", obs_hat)
    error = np.abs(obs - obs_hat)
    rae = error/abs(obs)
    print("estimator relative error: ", rae, "%")

    reward_total = reward_total+reward
    if done or truncated:
        print(f'Terminated after {i+1} iterations.')
        print(reward_total)
        obs, info = env.reset()
        break

env.close()