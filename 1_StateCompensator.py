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

# desired pole
P = np.array([-10, -0.5+1j, -0.5-1j, -20])
Pt = 4*P

# compute regulator and observer gain
K = control.place(A, B, P)
L = control.place(At,Bt, Pt)
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

obs_hat = np.zeros(4)
print(obs_hat)
u_array = []
u_total = 0
theta_array = []
theta_max = 0

for i in range(1000):
    env.render()

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
    theta_array.append(obs[2])
    print("obs: ", obs)

    # compute state estimator
    obs_hat = compute_state_estimator(obs_hat, obs, clip_force)
    print("obs_hat: ", obs_hat)
    error = np.abs(obs - obs_hat)
    rae = error/abs(obs)
    print("estimator relative error: ", rae, "%")

    reward_total = reward_total+reward
    if done or truncated:
        for i in range(len(u_array)):
            u_total += np.abs(u_array[i])

        print(f'Terminated after {i+1} iterations.')
        print("reward: ", reward_total)

        u_avg = u_total/len(u_array)
        print("u_avg: ", u_avg)

        theta_max = np.amax(theta_array)
        print("theta_max: ", theta_max * 180/np.pi, "degree")
        obs, info = env.reset()
        break

env.close()