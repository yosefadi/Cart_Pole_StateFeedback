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

At = np.transpose(A)
Bt = np.transpose(C)
Ct = np.transpose(B)

# desired pole
P = np.array([-10, -0.25+0.25j, -0.25-0.25j, -20])
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

obs_hat = np.zeros(4)
print(obs_hat)
u_array = []
theta_array = []
t_array = []

for i in range(1000):
    # time logging
    t = i*dt
    t_array.append(t)

    env.render()

    # states data logging
    print("obs: ", obs)
    print("obs_hat: ", obs_hat)
    theta_array.append(obs[2])

    # MODIFY THIS PART
    action, force = apply_state_controller(obs_hat)
    print("u: ", force)
    u_array.append(force)

    # absolute value, since 'action' determines the sign, F_min = -10N, F_max = 10N
    clip_force = np.clip(force, -10, 10)
    abs_force = np.abs(float(clip_force))

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

        u_array_abs = []
        for i in range(len(u_array)):
            u_array_abs.append(np.abs(u_array[i]))

        u_avg = np.around(np.mean(u_array_abs),3)
        print("force_avg: ", u_avg, "N")

        theta_min = np.amin(theta_array)
        theta_max = np.amax(theta_array)
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
                peak_time = np.around(i * dt,3)
                print("peak_time: ", peak_time, "s")
                break

        obs, info = env.reset()
        break

env.close()