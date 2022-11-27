import gym
import numpy as np
import control

l = 0.5
mp = 0.1
mc = 1.0
g = 9.8
dt = 0.02

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

C = np.array([[1, 0, 0, 0]])

# Augmented SS Equation for Robust Tracking
A_aug = np.block([[np.zeros([C.shape[0],C.shape[0]]), C],
                  [np.zeros([A.shape[0],C.shape[0]]), A]])
print("A_aug: ", A_aug)

B_aug = np.block([[np.zeros([C.shape[0],1])],
                  [B]])
print("B_aug: ", B_aug)

B_L = np.array(B_aug, copy=True)
for i in range(B_L.shape[0]):
    if B_L[i] != 0:
        B_L[i] = 1
    else:
        B_L[i] = 0


# noise/disturbance 
w = np.array([0.5])
w = np.reshape(w,1)

# desired pole
P = np.array([-0.25+0.5j, -0.25-0.5j, -10, -20])
P_aug = np.array([-0.1+0.1j,-0.1-0.1j,-0.2-1j,-0.2+1j,-10])

# compute regulator gain
K = control.place(A,B,P)
K_aug = control.place(A_aug, B_aug, P_aug)

print("K_aug: ", K_aug)

def f_aug_linear(x, u):
    x_aug_dot = A_aug@x + B_aug@(u+w)
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
    if u > 0:
        action = 1
    else:
        action = 0
    return action, u

obs_aug = np.block([[np.zeros([C.shape[0],1])],
                          [obs.reshape([4,1])]])
obs_aug = np.reshape(obs_aug, obs_aug.shape[0])
force = np.zeros([1,])

for i in range(1000):
    env.render()
    
    print("obs: ", obs)
    print("obs_aug: ", obs_aug)

    # MODIFY THIS PART
    action, force = apply_state_controller(obs_aug)

    force = force + w
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
    if done or truncated:
        print(f'Terminated after {i+1} iterations.')
        print(reward_total)
        obs, info = env.reset()
        break

env.close()
