import numpy as np
import copy
import time

CLEANUP_MAP = [
    "@@@@@@@@@@@@@@@@@@",
    "@RRRRRR     BBBBB@",
    "@HHHHHH      BBBB@",
    "@RRRRRR     BBBBB@",
    "@RRRRR  P    BBBB@",
    "@RRRRR    P BBBBB@",
    "@HHHHH       BBBB@",
    "@RRRRR      BBBBB@",
    "@HHHHHHSSSSSSBBBB@",
    "@HHHHHHSSSSSSBBBB@",
    "@RRRRR   P P BBBB@",
    "@HHHHH   P  BBBBB@",
    "@RRRRRR    P BBBB@",
    "@HHHHHH P   BBBBB@",
    "@RRRRR       BBBB@",
    "@HHHH    P  BBBBB@",
    "@RRRRR       BBBB@",
    "@HHHHH  P P BBBBB@",
    "@RRRRR       BBBB@",
    "@HHHH       BBBBB@",
    "@RRRRR       BBBB@",
    "@HHHHH      BBBBB@",
    "@RRRRR       BBBB@",
    "@HHHH       BBBBB@",
    "@@@@@@@@@@@@@@@@@@",
]

ORIENTATIONS = {"LEFT": [0, -1], "RIGHT": [0, 1], "UP": [-1, 0], "DOWN": [1, 0]}

class CleanupEnv():
    def __init__(self):
        self.player_num = 4
        self.height = 16
        self.width = 23
        self.clean_beam_len = 5
        self.fire_beam_len = 5
        self.waste_spawn_prob = 0.5
        self.apple_respawn_prob = 0.05
        self.threshold_restoration = 0.0
        self.threshold_depletion = 0.4
        self.window_size = 7
        self.punishment = 50
        self.action_num = 7

        self.waste_num_origin = 50
        self.waste_cost = 1
        self.apple_reward = 1
        self.beam_cost = 1
        self.waste_regeneration_rate = 0.4
        self.final_time = 100
        self.channel = self.player_num + 3
        # self.max_apple_regeneration_rate = config["max_apple_regeneration_rate"]

        self.state = None
        # self.observation_space = Dict(dict(observation=Box(low=0, high=1, shape=(self.player_num+3, self.height, self.width), dtype=np.int8),))
        # self.action_space = Discrete(6)
        self.players = [f'player_{i+1}' for i in range(self.player_num)]

        self.order = np.array(list(range(self.player_num)))

        # self.render_count = 0
        # self.render_frame = []
        self.dir_name = int(time.time())

    def reset(self):
        self.state = np.zeros(
            (self.player_num + 3, self.height, self.width), dtype=np.int8)
        self.player_pos = np.zeros((self.player_num, 2), dtype=np.int8)
        self.player_orientation = [ORIENTATIONS["LEFT"] for i in range(self.player_num)]
        self.waste_pos = np.zeros((self.waste_num_origin, 2), dtype=np.int8)

        player_pos= np.random.choice((self.height-11)*self.width, (self.player_num,), replace=False)
        for i in range(self.player_num):
            self.state[i, int(player_pos[i]//self.width+6), int(player_pos[i]%self.width)] = 1
            self.player_pos[i] = [int(player_pos[i]//self.width+6), int(player_pos[i]%self.width)]
        
        waste_pos= np.random.choice(6*self.width, (self.waste_num_origin,), replace=False)
        for i in range(self.waste_num_origin):
            self.state[self.player_num + 1, int(waste_pos[i]//self.width), int(waste_pos[i]%self.width)] = 1

        self.time = 0
        obs = []
        for i in range(4):
            o = self.state[:, :, max(0, self.player_pos[i][1]-self.window_size):min(self.player_pos[i][1]+self.window_size+1, self.width)]
            if self.player_pos[i][1] - self.window_size < 0:
                o = np.pad(o, ((0, 0), (0, 0), (self.window_size-self.player_pos[i][1], 0)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))
            elif self.player_pos[i][1] + self.window_size + 1 > self.width:
                o = np.pad(o, ((0, 0), (0, 0), (0, self.player_pos[i][1]-(self.width-self.window_size-1))), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))
            obs.append(o)
        return np.array(obs)

    def step(self, action_dict):
        self.time += 1
        collect_waste_num = 0
        collect_apple_num = 0
        punish_num = 0
        rewards = [0 for _ in range(4)]
        # np.random.shuffle(self.order)
        for id in range(4):
            action = action_dict[id]
            if self.__actionmask__(id)[action] == 0:
                action = 4

            x = self.player_pos[id, 0]
            y = self.player_pos[id, 1]                 

            if action == 0:
                self.state[id, x, y] = 0
                self.state[id, x-1, y] = 1
                self.player_pos[id, 0] -= 1
                self.player_orientation[id] = ORIENTATIONS["UP"]
            elif action == 1:
                self.state[id, x, y] = 0
                self.state[id, x+1, y] = 1
                self.player_pos[id, 0] += 1
                self.player_orientation[id] = ORIENTATIONS["DOWN"]
            elif action == 2:
                self.state[id, x, y] = 0
                self.state[id, x, y-1] = 1
                self.player_pos[id, 1] -= 1
                self.player_orientation[id] = ORIENTATIONS["LEFT"]
            elif action == 3:
                self.state[id, x, y] = 0
                self.state[id, x, y+1] = 1
                self.player_pos[id, 1] += 1
                self.player_orientation[id] = ORIENTATIONS["RIGHT"]

            if self.state[-1, self.player_pos[id, 0], self.player_pos[id, 1]] == 1:
                collect_apple_num += 1
                rewards[id] += self.apple_reward

            elif action == 5: # clean beam
                rewards[id] -= self.beam_cost
                target_pos = copy.deepcopy(self.player_pos[id])
                for _ in range(self.clean_beam_len):
                    target_pos[0] += self.player_orientation[id][0]
                    target_pos[1] += self.player_orientation[id][1]
                    if target_pos[0] < 0 or target_pos[0] >= self.height or target_pos[1] < 0 or target_pos[1] >= self.width:
                        break
                    if self.state[-2, target_pos[0], target_pos[1]] == 1:
                        self.state[-2, target_pos[0], target_pos[1]] = 0 # success clean
                        collect_waste_num += 1
                        break
            
            elif action == 6:
                rewards[id] -= self.beam_cost
                target_pos = copy.deepcopy(self.player_pos[id])
                beam_blocked = False
                for _ in range(self.clean_beam_len):
                    target_pos[0] += self.player_orientation[id][0]
                    target_pos[1] += self.player_orientation[id][1]
                    if target_pos[0] < 0 or target_pos[0] >= self.height or target_pos[1] < 0 or target_pos[1] >= self.width:
                        break
                    for j in range(self.player_num):
                        if j == id:
                            continue
                        if np.array_equal(target_pos, self.player_pos[j]): # punish somebody
                            punish_num += 1
                            rewards[j] -= self.punishment
                            beam_blocked = True
                            break
                    if beam_blocked:
                        break
                
            
        dones = [False for _ in range(4)]
        waste_num = np.sum(self.state[-2, 0 : 6, :])
        apples_num = np.sum(self.state[-1, self.height - 5 : self.height, :])
        apples_regeneration_rate = 1 - waste_num / self.waste_num_origin
        apples_regeneration_distribution = [1 - apples_regeneration_rate, apples_regeneration_rate]
        waste_regeneration_distribution = [1 - self.waste_regeneration_rate, self.waste_regeneration_rate]
        is_new_waste = np.random.choice(2, size=(1,), p=waste_regeneration_distribution)
        is_new_apple = np.random.choice(2, size=(1,), p=apples_regeneration_distribution)
        if is_new_waste[0] == 1 and waste_num < self.waste_num_origin:
            empty_waste_grid = np.where(
                self.state[self.player_num + 1, 0 : 6, :] == 0
            )
            empty_waste_grid_num = len(empty_waste_grid[0])
            new_waste_pos = np.random.choice(empty_waste_grid_num, 1, replace=False)
            self.state[self.player_num + 1, empty_waste_grid[0][new_waste_pos], empty_waste_grid[1][new_waste_pos]] = 1
        
        if is_new_apple[0]==1 and apples_num < 5 * self.width:
            empty_apple_grid = np.where(
                self.state[self.player_num + 2, self.height - 5 : self.height, :] == 0
            )
            empty_apple_grid_num = len(empty_apple_grid[0])
            new_apple_pos = np.random.choice(empty_apple_grid_num, 1, replace=False)
            self.state[self.player_num + 2, empty_apple_grid[0][new_apple_pos]+self.height-5, empty_apple_grid[1][new_apple_pos]] = 1

        if self.time >= self.final_time:
            for id in range(4):
                dones[id] = True

        return self.__obs__(), np.array(rewards), np.array(dones), [collect_waste_num, collect_apple_num, punish_num]

    def __obs__(self):
        obs = []
        for i in range(4):
            o = self.state[:, :, max(0, self.player_pos[i][1]-self.window_size):min(self.player_pos[i][1]+self.window_size+1, self.width)]
            if self.player_pos[i][1] - self.window_size < 0:
                o = np.pad(o, ((0, 0), (0, 0), (self.window_size-self.player_pos[i][1], 0)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))
            elif self.player_pos[i][1] + self.window_size + 1 > self.width:
                o = np.pad(o, ((0, 0), (0, 0), (0, self.player_pos[i][1]-(self.width-self.window_size-1))), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))
            obs.append(o)
        return np.array(obs)

    def __actionmask__(self, id):
        actions = np.zeros((7,))
        # find current position
        x, y = self.player_pos[id, 0], self.player_pos[id, 1]
        if x == 0 or np.sum(self.state[:self.player_num, x-1, y]) > 0:
            actions[0] = 1
        if x == self.height-1 or np.sum(self.state[:self.player_num, x+1, y]) > 0:
            actions[1] = 1
        if y == 0 or np.sum(self.state[:self.player_num, x, y-1]) > 0:
            actions[2] = 1
        if y == self.width-1 or np.sum(self.state[:self.player_num, x, y+1]) > 0:
            actions[3] = 1

        return 1 - actions  # available actions---output 1

    
class SnowDriftEnv():
    def __init__(self):
        self.player_num = 4
        self.drift_num = 6
        self.height = 8
        self.width = 8
        self.cost = 4
        self.final_time = 50
        self.drift_return = 6
        self.channel = self.player_num + 1
        self.action_space = 6
        
    def reset(self):
        self.state = np.zeros(
            (self.player_num + 1, self.height, self.width), dtype=np.int8)
        self.player_pos = np.zeros((self.player_num, 2), dtype=np.int8)
        self.waste_pos = np.zeros((self.drift_num, 2), dtype=np.int8)

        # player_pos = np.random.choice(self.height * self.width, (self.player_num,), replace=False)
        empty_grid = np.where(self.state[-1,:,:]==0)
        empty_grid_num = len(empty_grid[0])
        for i in range(self.player_num):
            grid_index=np.random.randint(0,empty_grid_num)
            self.player_pos[i]=[empty_grid[0][grid_index],empty_grid[1][grid_index]]
            self.state[i,empty_grid[0][grid_index],empty_grid[1][grid_index]]=1
        drift_pos=np.random.choice(empty_grid_num,size=(self.drift_num,),replace=False)
        for i in range(self.drift_num):
            self.state[-1,empty_grid[0][drift_pos[i]],empty_grid[1][drift_pos[i]]]=1
        
        self.time = 0
        obs = []
        for i in range(4):
            o = self.state[:, :, max(0, self.player_pos[i][1]-2):min(self.player_pos[i][1]+3, 8)]
            if self.player_pos[i][1] - 2 < 0:
                o = np.pad(o, ((0, 0), (0, 0), (2-self.player_pos[i][1], 0)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))
            elif self.player_pos[i][1] + 3 > 8:
                o = np.pad(o, ((0, 0), (0, 0), (0, self.player_pos[i][1]-5)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))
            obs.append(o)
        return np.array(obs)

    def step(self, action_dict):
        self.time += 1
        picked_sd = 0
        rewards = [0 for _ in range(4)]
        # remove_drift_players = []
        for id in range(4):
            action = action_dict[id]
            if self.__actionmask__(id)[action] == 0:
                action = 4

            x = self.player_pos[id, 0]
            y = self.player_pos[id, 1]                 

            if action == 0:
                self.state[id, x, y] = 0
                self.state[id, x-1, y] = 1
                self.player_pos[id, 0] -= 1
            elif action == 1:
                self.state[id, x, y] = 0
                self.state[id, x+1, y] = 1
                self.player_pos[id, 0] += 1
            elif action == 2:
                self.state[id, x, y] = 0
                self.state[id, x, y-1] = 1
                self.player_pos[id, 1] -= 1
            elif action == 3:
                self.state[id, x, y] = 0
                self.state[id, x, y+1] = 1
                self.player_pos[id, 1] += 1
            elif action == 5:
                picked_sd += 1
                self.state[-1, x, y] = 0
                for j in range(4):
                    if j == id:
                        rewards[j] += self.drift_return - self.cost
                    else:
                        rewards[j] += self.drift_return

        dones = [False for _ in range(4)]
        if self.time >= self.final_time or np.sum(self.state[-1, :, :]) == 0: # time limit, and exist snowdrift
            for id in range(4):
                dones[id] = True
        
        return self.__obs__(), np.array(rewards), np.array(dones), picked_sd
    
    def __obs__(self):
        obs = []
        for i in range(4):
            o = self.state[:, :, max(0, self.player_pos[i][1]-2):min(self.player_pos[i][1]+3, 8)]
            if self.player_pos[i][1] - 2 < 0:
                o = np.pad(o, ((0, 0), (0, 0), (2-self.player_pos[i][1], 0)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))
            elif self.player_pos[i][1] + 3 > 8:
                o = np.pad(o, ((0, 0), (0, 0), (0, self.player_pos[i][1]-5)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))
            obs.append(o)
        return np.array(obs)

    def __actionmask__(self, id):
        actions = np.zeros((6,))
        # find current position
        x, y = self.player_pos[id, 0], self.player_pos[id, 1]
        if x == 0 or np.sum(self.state[:self.player_num, x-1, y]) > 0:
            actions[0] = 1
        if x == self.height-1 or np.sum(self.state[:self.player_num, x+1, y]) > 0:
            actions[1] = 1
        if y == 0 or np.sum(self.state[:self.player_num, x, y-1]) > 0:
            actions[2] = 1
        if y == self.width-1 or np.sum(self.state[:self.player_num, x, y+1]) > 0:
            actions[3] = 1
        if self.state[-1, x, y] == 0:  # not allowed to link/unlink
            actions[5] = 1

        return 1 - actions  # available actions---output 1

# class HarvestEnv():
#     def __init__(self):
#         self.player_num = 4
#         self.apple_num = 9
#         self.height = 8
#         self.width = 8
#         self.apple_reward = 1
#         self.final_time = 100
#         self.spawn_prob = [0, 0.01, 0.05, 0.25]
#         self.max_apple_num = 64

#     def reset(self):
#         self.state = np.zeros((self.player_num+1, self.height, self.width), dtype=np.int8)
#         self.player_pos = np.zeros((self.player_num, 2), dtype=np.int8)
#         empty_grid = np.where(self.state[-1, :, :] == 0)
#         empty_grid_num = len(empty_grid[0])

#         for i in range(self.player_num):
#             grid_index = np.random.randint(0, empty_grid_num)
#             self.player_pos[i] = [empty_grid[0][grid_index],empty_grid[1][grid_index]]
#             self.state[i,empty_grid[0][grid_index], empty_grid[1][grid_index]] = 1

#         # add apple
#         apple_pos = np.random.choice(empty_grid_num,size=(self.apple_num,), replace=False)
#         for i in range(self.apple_num):
#             self.state[-1, empty_grid[0][apple_pos[i]], empty_grid[1][apple_pos[i]]] = 1

#         self.time = 0
#         obs = []
#         for i in range(self.player_num):
#             o = self.state[:, :, max(0, self.player_pos[i][1]-2):min(self.player_pos[i][1]+3, 8)]
#             if self.player_pos[i][1] - 2 < 0:
#                 o = np.pad(o, ((0, 0), (0, 0), (2-self.player_pos[i][1], 0)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))
#             elif self.player_pos[i][1] + 3 > 8:
#                 o = np.pad(o, ((0, 0), (0, 0), (0, self.player_pos[i][1]-5)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))
#             obs.append(o)
#         return np.array(obs)

#     def step(self, action_dict):
#         self.time += 1
#         rewards = [0 for _ in range(4)]
#         for id in range(self.player_num):
#             action = action_dict[id]
#             if self.__actionmask__(id)[action] == 0:
#                 action = 4
#             x = self.player_pos[id, 0]
#             y = self.player_pos[id, 1] 

#             if action == 0:
#                 self.state[id, x, y] = 0
#                 self.state[id, x-1, y] = 1
#                 self.player_pos[id, 0] -= 1
#             elif action == 1:
#                 self.state[id, x, y] = 0
#                 self.state[id, x+1, y] = 1
#                 self.player_pos[id, 0] += 1
#             elif action == 2:
#                 self.state[id, x, y] = 0
#                 self.state[id, x, y-1] = 1
#                 self.player_pos[id, 1] -= 1
#             elif action == 3:
#                 self.state[id, x, y] = 0
#                 self.state[id, x, y+1] = 1
#                 self.player_pos[id, 1] += 1

#             elif 

class StagHuntEnv():
    def __init__(self):
        self.player_num = 4
        self.stag_num = 2
        self.hare_num = 4
        self.height = 8
        self.width = 8
        self.stag_reward = 10
        self.hare_reward = 1
        self.final_time = 50
        self.first_hunt_time = 0
        self.channel = self.player_num + 2
        self.prey_moving_mode = 'random'
        self.action_space = 7

    def reset(self):
        self.state = np.zeros(
            (self.channel, self.height, self.width), dtype=np.int8)
        self.player_pos = np.zeros((self.player_num, 2), dtype=np.int8)
        self.stag_pos = np.zeros((self.stag_num, 2), dtype=np.int8)
        self.hare_pos = np.zeros((self.hare_num, 2), dtype=np.int8)
        player_pos= np.random.choice(self.height * self.width, (self.player_num,), replace=False)
        for i in range(self.player_num):
            self.state[i, int(player_pos[i]//self.width), int(player_pos[i]%self.width)] = 1
            self.player_pos[i] = [int(player_pos[i]//self.width), int(player_pos[i]%self.width)]

        stag_hare_pos = np.random.choice(self.height*self.width, (self.stag_num+self.hare_num,), replace=False)
        for i in range(self.stag_num):
            self.state[-2, int(stag_hare_pos[i]//self.width), int(stag_hare_pos[i]%self.width)] = 1
            self.stag_pos[i] = [int(stag_hare_pos[i]//self.width), int(stag_hare_pos[i]%self.width)]

        for i in range(self.stag_num, self.stag_num + self.hare_num):
            self.state[-1, int(stag_hare_pos[i]//self.width), int(stag_hare_pos[i]%self.width)] = 1
            self.hare_pos[i-self.stag_num] = [int(stag_hare_pos[i]//self.width), int(stag_hare_pos[i]%self.width)]

        self.time = 0
        self.first_hunt_time = 0
        self.terminal = np.zeros(self.player_num, dtype=np.int8)
        obs = []
        for i in range(4):
            o = self.state[:, :, max(0, self.player_pos[i][1]-2):min(self.player_pos[i][1]+3, 8)]
            if self.player_pos[i][1] - 2 < 0:
                o = np.pad(o, ((0, 0), (0, 0), (2-self.player_pos[i][1], 0)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))
            elif self.player_pos[i][1] + 3 > 8:
                o = np.pad(o, ((0, 0), (0, 0), (0, self.player_pos[i][1]-5)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))
            obs.append(o)
        return np.array(obs)
    
    def step(self, action_dict):
        self.time += 1
        hunt_stag_num = 0
        hunt_hare_num = 0
        rewards = [0 for _ in range(4)]
        dones = [self.terminal[i] for i in range(4)]
        hunt_hare_players = []
        hunt_stag_players = []
        for id in range(self.player_num):
            action = action_dict[id]
            if self.__actionmask__(id)[action] == 0:
                action = 4
            
            x = self.player_pos[id, 0]
            y = self.player_pos[id, 1]

            if action == 0:
                self.state[id, x, y] = 0
                self.state[id, x-1, y] = 1
                self.player_pos[id, 0] -= 1
            elif action == 1:
                self.state[id, x, y] = 0
                self.state[id, x+1, y] = 1
                self.player_pos[id, 0] += 1
            elif action == 2:
                self.state[id, x, y] = 0
                self.state[id, x, y-1] = 1
                self.player_pos[id, 1] -= 1
            elif action == 3:
                self.state[id, x, y] = 0
                self.state[id, x, y+1] = 1
                self.player_pos[id, 1] += 1

            elif action == 5:
                hunt_hare_players.append(id)
            elif action == 6:
                hunt_stag_players.append(id)

        while len(hunt_hare_players) != 0:
            same_hare_with_zero = [hunt_hare_players[0]]
            zero_hare_pos = tuple(self.player_pos[hunt_hare_players[0]])
            for i in range(len(hunt_hare_players)-1, 0, -1):
                if tuple(self.player_pos[hunt_hare_players[i]]) == zero_hare_pos:
                    same_hare_with_zero.append(hunt_hare_players[i])
                    hunt_hare_players.pop(i)
            hunt_hare_players.pop(0)
            self.state[id, zero_hare_pos[0], zero_hare_pos[1]] = 0
            for id in same_hare_with_zero:
                self.state[id, zero_hare_pos[0], zero_hare_pos[1]] = 0
                self.terminal[id] = 1

                rewards[id]+=(self.hare_reward*1.0/len(same_hare_with_zero))
                # if total_strength == 0:
                #     rewards[self.players[id]
                #             ] += (self.hare_reward*1.0/len(same_hare_with_zero))
                # else:
                #     rewards[self.players[id]
                #             ] += (self.hare_reward*self.player_strength[id]/total_strength)
                dones[id] = True
            self.state[-1, zero_hare_pos[0], zero_hare_pos[1]] = 0
            index = np.where((self.hare_pos == (zero_hare_pos[0], zero_hare_pos[1])).all(axis=1))[0]
            self.hare_pos[index[0]][0] = -1
            self.hare_pos[index[0]][1] = -1
            hunt_hare_num += 1

        while len(hunt_stag_players) != 0:
            same_stag_with_zero = [hunt_stag_players[0]]
            zero_stag_pos = tuple(self.player_pos[hunt_stag_players[0]])
            for i in range(len(hunt_stag_players)-1, 0, -1):
                if tuple(self.player_pos[hunt_stag_players[i]]) == zero_stag_pos:
                    same_stag_with_zero.append(hunt_stag_players[i])
                    hunt_stag_players.pop(i)
            hunt_stag_players.pop(0)
            if len(same_stag_with_zero) != 1:
                for id in same_stag_with_zero:
                    self.state[id, zero_stag_pos[0], zero_stag_pos[1]] = 0
                    self.terminal[id] = 1
                    rewards[id]+=(self.stag_reward*1.0/len(same_stag_with_zero))
                    dones[id] = True
                self.state[-2, zero_stag_pos[0], zero_stag_pos[1]] = 0
                index = np.where((self.stag_pos == (zero_stag_pos[0], zero_stag_pos[1])).all(axis=1))[0]
                self.stag_pos[index[0]][0] = -1
                self.stag_pos[index[0]][1] = -1
                hunt_stag_num += 1

        if self.time >= self.final_time:
            for id in range(4):
                dones[id] = True

        all_done = True if sum(dones) == self.player_num else False
        dones.append(all_done)

        return self.__obs__(), np.array(rewards), np.array(dones), [hunt_hare_num, hunt_stag_num]

    def __obs__(self):
        obs = []
        for i in range(4):
            o = self.state[:, :, max(0, self.player_pos[i][1]-2):min(self.player_pos[i][1]+3, 8)]
            if self.player_pos[i][1] - 2 < 0:
                o = np.pad(o, ((0, 0), (0, 0), (2-self.player_pos[i][1], 0)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))
            elif self.player_pos[i][1] + 3 > 8:
                o = np.pad(o, ((0, 0), (0, 0), (0, self.player_pos[i][1]-5)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))
            obs.append(o)
        return np.array(obs)
    
    def __actionmask__(self, id):
        actions = np.zeros((7,))
        x, y = self.player_pos[id, 0], self.player_pos[id, 1]
        if x == 0:
            actions[0] = 1
        if x == self.height - 1:
            actions[1] = 1
        if y == 0:
            actions[2] = 1
        if y == self.width - 1:
            actions[3] = 1
        if self.state[-1, x, y] == 0:
            actions[5] = 1
        if self.state[-2, x, y] == 0 or np.sum(self.state[:self.player_num, x, y]) <= 1:
            actions[6] = 1
        return 1 - actions

class ModifiedCleanupEnv():
    def __init__(self):
        self.player_num = 4
        self.height = 8
        self.width = 8
        self.waste_spawn_prob = 0.5
        # self.apple_respawn_prob = 0.05
        self.window_size = 2
        self.action_num = 6

        self.waste_num_origin = 8
        self.waste_cost = 1
        self.apple_reward = 10
        self.waste_regeneration_rate = 0.4
        self.final_time = 1000
        self.channel = self.player_num + 3
        # self.max_apple_regeneration_rate = config["max_apple_regeneration_rate"]

        self.state = None

    def reset(self):
        self.state = np.zeros(
            (self.player_num + 3, self.height, self.width), dtype=np.int8)
        self.player_pos = np.zeros((self.player_num, 2), dtype=np.int8)
        self.waste_pos = np.zeros((self.waste_num_origin, 2), dtype=np.int8)

        player_pos= np.random.choice((self.height-4)*self.width, (self.player_num,), replace=False)
        for i in range(self.player_num):
            self.state[i, int(player_pos[i]//self.width+2), int(player_pos[i]%self.width)] = 1
            self.player_pos[i] = [int(player_pos[i]//self.width+2), int(player_pos[i]%self.width)]
        
        waste_pos= np.random.choice(2*self.width, (self.waste_num_origin,), replace=False)
        for i in range(self.waste_num_origin):
            self.state[self.player_num + 1, int(waste_pos[i]//self.width), int(waste_pos[i]%self.width)] = 1

        self.time = 0
        obs = []
        for i in range(4):
            o = self.state[:, :, max(0, self.player_pos[i][1]-self.window_size):min(self.player_pos[i][1]+self.window_size+1, self.width)]
            if self.player_pos[i][1] - self.window_size < 0:
                o = np.pad(o, ((0, 0), (0, 0), (self.window_size-self.player_pos[i][1], 0)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))
            elif self.player_pos[i][1] + self.window_size + 1 > self.width:
                o = np.pad(o, ((0, 0), (0, 0), (0, self.player_pos[i][1]-(self.width-self.window_size-1))), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))
            obs.append(o)
        return np.array(obs)

    def step(self, action_dict):
        self.time += 1
        collect_waste_num = 0
        # collect_apple_num = 0
        # punish_num = 0
        rewards = [0 for _ in range(4)]
        # np.random.shuffle(self.order)
        for id in range(4):
            action = action_dict[id]
            if self.__actionmask__(id)[action] == 0:
                action = 4

            x = self.player_pos[id, 0]
            y = self.player_pos[id, 1]                 

            if action == 0:
                self.state[id, x, y] = 0
                self.state[id, x-1, y] = 1
                self.player_pos[id, 0] -= 1
            elif action == 1:
                self.state[id, x, y] = 0
                self.state[id, x+1, y] = 1
                self.player_pos[id, 0] += 1
            elif action == 2:
                self.state[id, x, y] = 0
                self.state[id, x, y-1] = 1
                self.player_pos[id, 1] -= 1
            elif action == 3:
                self.state[id, x, y] = 0
                self.state[id, x, y+1] = 1
                self.player_pos[id, 1] += 1

            elif action == 5:
                assert self.state[self.player_num + 1, x, y] + self.state[self.player_num + 2, x, y] > 0, 'link ERROR'
                if x < 2:
                    self.state[self.player_num + 1, x, y] = 0
                    rewards[id] -= self.waste_cost
                    collect_waste_num += 1
                elif x >= self.height - 2:
                    self.state[self.player_num + 2, x, y] = 0
                    rewards[id] += self.apple_reward
                
        dones = [False for _ in range(4)]
        waste_num = np.sum(self.state[-2, 0 : 2, :])
        apples_num = np.sum(self.state[-1, self.height - 2 : self.height, :])
        apples_regeneration_rate = 1 - waste_num / self.waste_num_origin
        apples_regeneration_distribution = [1 - apples_regeneration_rate, apples_regeneration_rate]
        waste_regeneration_distribution = [1 - self.waste_regeneration_rate, self.waste_regeneration_rate]
        is_new_waste = np.random.choice(2, size=(1,), p=waste_regeneration_distribution)
        is_new_apple = np.random.choice(2, size=(1,), p=apples_regeneration_distribution)
        if is_new_waste[0] == 1 and waste_num < self.waste_num_origin:
            empty_waste_grid = np.where(
                self.state[self.player_num + 1, 0 : 2, :] == 0
            )
            empty_waste_grid_num = len(empty_waste_grid[0])
            new_waste_pos = np.random.choice(empty_waste_grid_num, 1, replace=False)
            self.state[self.player_num + 1, empty_waste_grid[0][new_waste_pos], empty_waste_grid[1][new_waste_pos]] = 1
        
        if is_new_apple[0]==1 and apples_num < 2 * self.width:
            empty_apple_grid = np.where(
                self.state[self.player_num + 2, self.height - 2 : self.height, :] == 0
            )
            empty_apple_grid_num = len(empty_apple_grid[0])
            new_apple_pos = np.random.choice(empty_apple_grid_num, 1, replace=False)
            self.state[self.player_num + 2, empty_apple_grid[0][new_apple_pos]+self.height-2, empty_apple_grid[1][new_apple_pos]] = 1

        if self.time >= self.final_time:
            for id in range(4):
                dones[id] = True

        return self.__obs__(), np.array(rewards), np.array(dones), collect_waste_num

    def __obs__(self):
        obs = []
        for i in range(4):
            o = self.state[:, :, max(0, self.player_pos[i][1]-2):min(self.player_pos[i][1]+3, 8)]
            if self.player_pos[i][1] - 2 < 0:
                o = np.pad(o, ((0, 0), (0, 0), (2-self.player_pos[i][1], 0)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))
            elif self.player_pos[i][1] + 3 > 8:
                o = np.pad(o, ((0, 0), (0, 0), (0, self.player_pos[i][1]-5)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))
            obs.append(o)
        return np.array(obs)

    def __actionmask__(self, id):
        actions = np.zeros((6,))
        # find current position
        x, y = self.player_pos[id, 0], self.player_pos[id, 1]
        if x == 0 or np.sum(self.state[:self.player_num, x-1, y]) > 0:
            actions[0] = 1
        if x == self.height-1 or np.sum(self.state[:self.player_num, x+1, y]) > 0:
            actions[1] = 1
        if y == 0 or np.sum(self.state[:self.player_num, x, y-1]) > 0:
            actions[2] = 1
        if y == self.width-1 or np.sum(self.state[:self.player_num, x, y+1]) > 0:
            actions[3] = 1
        if np.sum(self.state[self.player_num+1:self.player_num+3, x, y]) == 0:  # not allowed to link/unlink
            actions[5] = 1

        return 1 - actions  # available actions---output 1
