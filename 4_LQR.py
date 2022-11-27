import gym
import numpy as np
from scipy import linalg

l = 0.5
mp = 0.1
mc = 1.0
g = 9.8
dt = 0.02

# get environment
env = gym.make('CartPole-v1', render_mode="human").unwrapped
#env.env.seed(1)     # seed for reproducibility
obs, info = env.reset(seed=1)
reward_threshold = 475
reward_total = 0

# ADD SOMETHING HERE
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

Q = np.array([[1, 0, 0, 0],
              [0, 10, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 100]])

R = np.array([[0.1]])

P = linalg.solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R)@np.transpose(B)@P


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
theta_array = []
t_array = []

for i in range(1000):
    # time logging
    t = i*dt
    t_array.append(t)

    env.render()

    # print current state
    print("obs: ", obs)
    theta_array.append(obs[2])

    # get force direction (action) and force value (force)

    # MODIFY THIS PART
    action, force = apply_state_controller(K, obs)
    u_array.append(force)

    # absolute value, since 'action' determines the sign, F_min = -10N, F_max = 10N
    abs_force = abs(float(np.clip(force, -10, 10)))
    
    # change magnitute of the applied force in CartPole
    env.force_mag = abs_force

    # apply action
    obs, reward, done, truncated, info = env.step(action)

    reward_total = reward_total+reward
    print()
    if done or truncated or reward_total == reward_threshold:
        print(f'Terminated after {i+1} iterations.')
        print("reward: ", reward_total)

        u_array_abs = []
        for i in range(len(u_array)):
            u_array_abs.append(np.abs(u_array[i]))
        u_avg = np.around(np.mean(u_array_abs),3)
        print("u_avg: ", u_avg, "N")

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
            if theta_array[i] == search_theta:
                peak_time = np.around(i * dt,3)
                print("peak_time: ", peak_time, "s")
                break

        obs, info = env.reset()
        break

env.close()