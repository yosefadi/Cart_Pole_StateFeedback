import gym
import numpy as np
import control

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
P_aug = np.array([-0.5+1j,-0.5-1j,-0.5+0.5j,-0.5-0.5j,-15])

# compute regulator gain
K = control.place(A,B,P)
K_aug = control.place(A_aug, B_aug, P_aug)

print("K_aug: ", K_aug)

def f_aug_linear(x, u):
    x_aug_dot = A_aug@x + B_aug@u + B_L@u
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
theta_array = []
t_array = []

for i in range(1000):
    # time logging
    t = i*dt
    t_array.append(t)

    env.render()
    
    # log state
    x_array.append(obs[0])
    theta_array.append(obs[2])
    print("obs: ", obs)
    print("obs_aug: ", obs_aug)

    # MODIFY THIS PART
    force = apply_state_controller(obs_aug)
    u_array.append(force)

    # input noise
    force = force + w
    
    # determine action
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

    obs_aug_dot = f_aug_linear(obs_aug, force)
    obs_aug = obs_aug + obs_aug_dot * dt

    for n in range(obs.shape[0]):
        obs_aug[n+C.shape[0]] = obs[n]

    reward_total = reward_total+reward
    print()
    if done or truncated or reward_total == reward_threshold:
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
