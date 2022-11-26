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
Ar = np.array([[0, 0, 1, 0],
               [0, 0, 0, 1],
               [0, -mp*(mp * (g-l) + mc*g)/((mc+mp)*((4/3) * mc + (1/3) * mp)), 0, 0],
               [0, (mp*(g-l) + mc * g)/(l*((4/3) * mc + (1/3) * mp)), 0, 0]])

Br = np.array([[0],
              [0],
              [(1/(mc + mp) - mp/((mc + mp) * ((4/3) * mc + (1/3) * mp)))],
              [(-1/(l * ((4/3) * mc + (1/3) * mp)))],])

Cr = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])

Aaa = Ar[:2,:2]
Aau = Ar[:2,2:]
Aua = Ar[2:,:2]
Auu = Ar[2:,2:]
Ba = Br[:2]
Bu = Br[2:]

A = np.empty([4,4])
A[[0, 1, 2, 3]] = Ar[[0, 2, 1, 3]]
A[:, [1, 2]] = A[:, [2, 1]]

B = np.empty([4,1])
B[[0, 1, 2, 3]] = Br[[0, 2, 1, 3]]

K = control.place(A, B, [-4, -0.5+1j, -0.5-1j, -11])

def compute_reduced_observer(Aaa, Aau, Aua, Auu, Bu, Cr, Lr, x_hat, x, u, dt):
    x[[2,1]] = x[[1,2]]
    y = Cr@x

    x_hat[[2,1]] = x_hat[[2,1]]
    xu_hat = x_hat[2:]

    xc_dot = (Auu - Lr@Aau)@xu_hat + (Aua - Lr@Aaa)@y + (Bu - Lr@Ba)@u
    xc = xc + xc_dot*dt
    xb = xc + Lr@y
    return xb

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
    action, force = apply_state_controller(K, obs)
    print("u:", force)

    # absolute value, since 'action' determines the sign, F_min = -10N, F_max = 10N
    clip_force = np.clip(force, -10, 10)
    abs_force = np.abs(float(clip_force))

    # change magnitute of the applied force in CartPole
    env.force_mag = abs_force

    # apply action
    obs, reward, done, truncated, info = env.step(action)
    print("obs: ", obs)

    reward_total = reward_total+reward
    if done or truncated:
        print(f'Terminated after {i+1} iterations.')
        print(reward_total)
        obs, info = env.reset()
        break

env.close()