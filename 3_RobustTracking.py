import gym
import numpy as np
import control

l = 0.5
mp = 0.1
mc = 1.0
#mt = mp+mk
g = 9.8

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

# Augmented SS Equation for Robust Tracking
A_aug = np.block([[np.zeros([C.shape[0],C.shape[0]]), C],
                  [np.zeros([A.shape[0],C.shape[0]]), A]])

B_aug = np.block([[np.zeros([C.shape[0],1])],
                  [B]])

# noise/disturbance 
w = 0.5

# place the regulator pole to -10, -0.5+i, -0.5-i, -20
P = np.array([-2, -0.5+1j, -0.5-1j, -9, -10, -7])
K = control.place(A_aug, B_aug, P)

def apply_state_controller(x):
    # feedback controller
    # MODIFY THIS PARTS
    u = -K@x   # u = -Kx
    #print(u)
    if u > 0:
        action = 1
    else:
        action = 0
    return action, u

print(obs)
for i in range(1000):
    env.render()
    
    obs_tilda = np.block([[np.zeros([C.shape[0],1])],
                          [obs.reshape([4,1])]])

    # MODIFY THIS PART
    action, force = apply_state_controller(obs_tilda)
    
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
