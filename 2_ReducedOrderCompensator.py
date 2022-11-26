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

Ar = np.empty([4,4])
Ar[[0, 1, 2, 3]] = A[[0, 2, 1, 3]]
Ar[:, [1, 2]] = Ar[:, [2, 1]]

Br = np.empty([4,1])
Br[[0, 1, 2, 3]] = B[[0, 2, 1, 3]]

Aaa = Ar[:2,:2]
Aau = Ar[:2,2:]
Aua = Ar[2:,:2]
Auu = Ar[2:,2:]
Ba = Br[:2]
Bu = Br[2:]

Cr = C
Cr[:,[1,2]] = Cr[:, [2,1]]

K = control.place(A, B, [-3, -0.25+0.5j, -0.25-0.5j, -11])
L = control.place(np.transpose(Auu), np.transpose(Aau), [-2.5+7j, -2.5-7j])
L = np.transpose(L)

def compute_reduced_observer(Aaa, Aau, Aua, Auu, Bu, Cr, Lr, x, x_hat, y, xcc, u, dt):
    x_hat_loc = x_hat
    x_hat_loc[[1,2]] = x_hat_loc[[1,2]]
    xu_hat = x_hat_loc[2:]
    print("xu_hat: ", xu_hat)

    xa = np.empty([2,])
    xa[[0]] = x[[0]]
    xa[[1]] = x[[3]]
    print("xa: ", xa)

    xcc_dot = (Auu - Lr@Aau)@xu_hat + (Aua - Lr@Aaa)@y + (Bu - Lr@Ba)@u
    xcc = xcc + xcc_dot*dt
    xu_hat = xcc + Lr@y
    
    x_hat_new = np.concatenate((xa, xu_hat))
    x_hat_new[[2,1]] = x_hat_new[[1,2]]
    return xcc,x_hat_new
    
def apply_state_controller(K, x):
    u = -K@x   # u = -Kx
    if u > 0:
        action = 1
    else:
        action = 0 
    return action, u

obs_hat = np.zeros(4)
xcc = np.zeros(2)
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

    y = C@obs
    print("y: ", y)
    print("obs_hat: ", obs_hat)
    xcc,obs_hat = compute_reduced_observer(Aaa, Aau, Aua, Auu, Bu, Cr, L, obs, obs_hat, y, xcc, clip_force, dt)

    reward_total = reward_total+reward
    if done or truncated:
        print(f'Terminated after {i+1} iterations.')
        print(reward_total)
        obs, info = env.reset()
        break

env.close()