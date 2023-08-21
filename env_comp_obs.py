import numpy as np
import copy
import time

class CleanupEnv():
    def __init__(self):
        self.player_num = 4
        self.height = 8
        self.width = 8

        self.rubbish_num_origin = 8
        self.rubbish_cost = 1
        self.apple_reward = 10
        self.rubbish_regeneration_rate = 0
        self.final_time = 100
        self.channel = self.player_num + 2
        # self.max_apple_regeneration_rate = config["max_apple_regeneration_rate"]

        self.state = None
        self.obs_mode = 'partial'
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
        self.rubbish_pos = np.zeros((self.rubbish_num_origin, 2), dtype=np.int8)
        self.product_attr = np.zeros((6, 4), dtype=np.int8)
        self.link = -np.ones((self.player_num,), dtype=np.int8)

        player_pos= np.random.choice((self.height-4)*self.width, (self.player_num,), replace=False)
        for i in range(self.player_num):
            self.state[i, int(player_pos[i]//self.width+2), int(player_pos[i]%self.width)] = 1
            self.player_pos[i] = [int(player_pos[i]//self.width+2), int(player_pos[i]%self.width)]
        
        rubbish_pos= np.random.choice(2*self.width, (self.rubbish_num_origin,), replace=False)
        for i in range(self.rubbish_num_origin):
            self.state[self.player_num + 1, int(rubbish_pos[i]//self.width), int(rubbish_pos[i]%self.width)] = 1

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
        pick_rubbish_num = 0
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
                    rewards[id] -= self.rubbish_cost
                    pick_rubbish_num += 1
                elif x >= self.height - 2:
                    self.state[self.player_num + 2, x, y] = 0
                    rewards[id] += self.apple_reward
            
        dones = [False for _ in range(4)]
        rubbish_num = np.sum(self.state[self.player_num + 1, 0 : 2, :])
        apples_num = np.sum(self.state[self.player_num + 2, self.height - 2 : self.height, :])
        apples_regeneration_rate = 1 - rubbish_num / (10)
        apples_regeneration_distribution = [1 - apples_regeneration_rate, apples_regeneration_rate]
        rubbish_regeneration_distribution = [1 - self.rubbish_regeneration_rate, self.rubbish_regeneration_rate]
        is_new_rubbish = np.random.choice(2, size=(1,), p=rubbish_regeneration_distribution)
        is_new_apple = np.random.choice(2, size=(1,), p=apples_regeneration_distribution)
        if is_new_rubbish[0] == 1 and rubbish_num < self.rubbish_num_origin:
            empty_rubbish_grid = np.where(
                self.state[self.player_num + 1, 0 : 2, :] == 0
            )
            empty_rubbish_grid_num = len(empty_rubbish_grid[0])
            new_rubbish_pos = np.random.choice(empty_rubbish_grid_num, 1, replace=False)
            self.state[self.player_num + 1, empty_rubbish_grid[0][new_rubbish_pos], empty_rubbish_grid[1][new_rubbish_pos]] = 1
        
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

        # if self.prosocial:
        #     avg_reward = sum(rewards.values())/self.player_num
        #     for i in rewards.keys():
        #         rewards[i] = avg_reward

        return self.__obs__(), np.array(rewards), np.array(dones), pick_rubbish_num

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

    def set_state(self, state_dict):
        state_dict_copy = copy.deepcopy(state_dict)
        self.state = state_dict_copy['state']
        self.time = state_dict_copy['time']
        self.player_pos = state_dict_copy['player_pos']
        return None

    def get_state(self):
        state_dict = {}
        state_dict['state'] = self.state.copy()
        state_dict['time'] = self.time
        state_dict['player_pos'] = self.player_pos.copy()
        return state_dict

    def check_connectivity(self):
        check_list = self.state[-1, :, :].copy()
        empty_grid = np.where(self.state[-1, :, :] == 0)
        empty_grid_num = len(empty_grid[0])
        start_index = np.random.randint(0, empty_grid_num)
        start = (empty_grid[0][start_index], empty_grid[1][start_index])
        grid_queue = [start]
        while len(grid_queue) != 0:
            x = grid_queue[0][0]
            y = grid_queue[0][1]
            if x != 0 and check_list[x-1, y] != 1:
                grid_queue.append((x-1, y))
            if y != 0 and check_list[x, y-1] != 1:
                grid_queue.append((x, y-1))
            if x != self.height-1 and check_list[x+1, y] != 1:
                grid_queue.append((x+1, y))
            if y != self.width-1 and check_list[x, y+1] != 1:
                grid_queue.append((x, y+1))
            check_list[x, y] = 1
            grid_queue.pop(0)
        if len(np.where(check_list == 0)[0]) != 0:
            return False
        return True
    
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
        
    def reset(self):
        self.state = np.zeros(
            (self.player_num + 1, self.height, self.width), dtype=np.int8)
        self.player_pos = np.zeros((self.player_num, 2), dtype=np.int8)
        self.rubbish_pos = np.zeros((self.drift_num, 2), dtype=np.int8)

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