import gymnasium as gym
import numpy as np

lp = 0.5
mp = 0.1
mk = 1.0
mt = mp+mk
g = 9.8

# get environment
env = gym.make('CartPole-v1', render_mode="human")
#env.env.seed(1)     # seed for reproducibility
obs, info = env.reset(seed=42)
reward_total = 0

# ADD SOMETHING HERE
K = 10**12 * np.array([[-1.0041,-0.024,-0.5781,-0.0014]])
#K = np.array([[ 1,1,1,1]])
def apply_state_controller(K, x):
    # feedback controller
    # MODIFY THIS PARTS
    u = -K@x   # u = -Kx
    print(u)
    if u > 0:
        return 1, u     # if force_dem > 0 -> move cart right
    else:
        return 0, u     # if force_dem <= 0 -> move cart left

for i in range(1000):
    env.render()
    
    # get force direction (action) and force value (force)

    # MODIFY THIS PART
    action, force = apply_state_controller(K, obs)
    
    # absolute value, since 'action' determines the sign, F_min = -10N, F_max = 10N
    abs_force = abs(float(np.clip(force, -10, 10)))
    
    # change magnitute of the applied force in CartPole
    env.env.force_mag = abs_force

    # apply action
    obs, reward, done, truncated, info = env.step(action)

    reward_total = reward_total+reward
    #if done:
    #    print(f'Terminated after {i+1} iterations.')
    #    print(reward_total)
    #    obs, info = env.reset()
    #    break

env.close()