from env import *
import wandb
import random
import numpy as np
max_episode = 100000
anneal_eps = 5000

env = ModifiedCleanupEnv()

step_time = 0
a2c_time = 0
offline_time = 0
test_time = 0
# loss_time = 0
# forward_time = 0
for ep in range(max_episode):
    print("eps", ep, "=====================================")
    obs = env.reset()
    all_done = False
    total_reward = np.zeros(4)
    total_collect_waste_num = np.zeros(env.player_num)
    total_collect_apple_num = 0
    step = 0
    values = []
    rewards = []

    while not all_done:
        actions = []
        for i in range(4):
            if i == 0 or i == 1: # clean waste
                avail_action = env.__actionmask__(i)
                avail_action_index = np.nonzero(avail_action)[0]
                player_i_pos = np.argwhere(obs[i][i] == 1)[0]
                player_i_real_pos = env.player_pos[i]
                if np.sum(obs[i][-2]) == 0:
                    if player_i_real_pos[0] >= 2:
                        action = 0
                    else:
                        action = random.choice([2,3])
                else:
                    waste_pos = np.argwhere(obs[i][-2] == 1)
                    distances = np.sum(np.abs(waste_pos - player_i_pos), axis=1)
                    nearest_waste_pos = waste_pos[np.argmin(distances)]
                    if nearest_waste_pos[0] < player_i_pos[0]:
                        action = 0  # 上
                    elif nearest_waste_pos[0] > player_i_pos[0]:
                        action = 1  # 下
                    elif nearest_waste_pos[1] < player_i_pos[1]:
                        action = 2  # 左
                    elif nearest_waste_pos[1] > player_i_pos[1]:
                        action = 3  # 右
                    else:
                        action = 5  # clean waste
                if avail_action[action] == 0: # not avail, just stay
                    action = 4
                actions.append(action)
            
            else: # collect apple
                avail_action = env.__actionmask__(i)
                avail_action_index = np.nonzero(avail_action)[0]
                player_i_pos = np.argwhere(obs[i][i] == 1)[0]
                player_i_real_pos = env.player_pos[i]
                if np.sum(obs[i][-1]) == 0:
                    if player_i_real_pos[0] < 6:
                        action = 1
                    else:
                        action = random.choice([2,3])
                else:
                    waste_pos = np.argwhere(obs[i][-1] == 1)
                    distances = np.sum(np.abs(waste_pos - player_i_pos), axis=1)
                    nearest_waste_pos = waste_pos[np.argmin(distances)]
                    if nearest_waste_pos[0] < player_i_pos[0]:
                        action = 0  # 上
                    elif nearest_waste_pos[0] > player_i_pos[0]:
                        action = 1  # 下
                    elif nearest_waste_pos[1] < player_i_pos[1]:
                        action = 2  # 左
                    elif nearest_waste_pos[1] > player_i_pos[1]:
                        action = 3  # 右
                    else:
                        action = 6  # collect apple
                if avail_action[action] == 0: # not avail, just stay
                    action = 4
                actions.append(action)

        next_obs, reward, done, info = env.step(actions)
        total_collect_waste_num += info[0]
        total_collect_apple_num += info[1]

        obs = next_obs
        total_reward += reward

    print('reward', total_reward)
    print('clean num', total_collect_waste_num)