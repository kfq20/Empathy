import collections
import numpy as np
import random
import time

class ReplayBuffer:
    def __init__(self, capacity, id):
        self.buffer = collections.deque(maxlen=capacity)
        self.id = id

    def add(self, state, action, other_action, reward, next_state, done):
        self.buffer.append((state, action, other_action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, other_action, reward, next_state, done = zip(*transitions)
        return np.array(state), np.array(action), np.array(other_action), np.array(reward), np.array(next_state), np.array(done)
    
    def size(self):
        return len(self.buffer)
    
    def find_most_similar_vector(self, obs):
    # 计算向量之间的相似度
        a = obs
        total_obs, _, total_action, _, _, _, _ = self.sample(self.size())
        b = total_obs
        similarity_scores = np.sum(a == b, axis=(1, 2, 3))

        # 找到与a最相似的向量
        most_similar_index = np.argmax(similarity_scores)
        c = b[most_similar_index]

        # 判断相等元素的个数是否大于总数的一半
        equal_count = np.sum(a == c)
        total_count = a.size
        if equal_count > total_count * 0.3:
            return c, total_action[most_similar_index]
        return None, None

    def similarity_compute(self, id, obs, other_action, ttt):
        similarity = np.zeros(4)
        total_condition = []
        total_other_obs = []
        for i in range(4):
            i_obs = []
            if i == id:
                total_condition.append([])
                total_other_obs.append([])
                continue
            else:
                condition = other_action[:, i] > -1
                condition = condition.astype(float)
                total_condition.append(condition)
                for j in range(obs.shape[0]):
                    if condition[j]:
                        pos = np.where(obs[j][i] > 0)
                        x, y = pos[0][0], pos[1][0]
                        other_obs = obs[j][:, :, max(0, y-2):min(y+3, 5)]
                        if y-2 < 0:
                            other_obs = np.pad(other_obs, ((0, 0), (0, 0), (2-y, 0)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))
                        elif y+3 > 5:
                            other_obs = np.pad(other_obs, ((0, 0), (0, 0), (0, y-2)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))
                        i_obs.append(other_obs)
                        ss_t = time.time()
                        similar_obs, similar_action = self.find_most_similar_vector(other_obs)
                        ee_t = time.time()
                        ttt += ee_t - ss_t
                        if similar_action == other_action[j][i]:
                            similarity[i] += 1
                if condition.sum() > 0:
                    similarity[i] /= condition.sum()
                total_other_obs.append(np.array(i_obs))
        return similarity, np.array(total_other_obs), total_condition, ttt
    
# class ReplayBuffer:
#     def __init__(self, capacity, id):
#         self.buffer = collections.deque(maxlen=capacity)
#         self.id = id

#     def add(self, state, action, avail_action, avail_next_action, other_action, reward, next_state, done, pad, hidden_state):
#         self.buffer.append((state, action, avail_action, avail_next_action, other_action, reward, next_state, done, pad, hidden_state))

#     def sample(self, batch_size):
#         transitions = random.sample(self.buffer, batch_size)
#         state, action, avail_action, avail_next_action, other_action, reward, next_state, done, pad, hidden_state = zip(*transitions)
#         return np.array(state), np.array(action), np.array(avail_action), np.array(avail_next_action), np.array(other_action), np.array(reward), np.array(next_state), np.array(done), np.array(pad), np.array(hidden_state)
    
#     def size(self):
#         return len(self.buffer)
    