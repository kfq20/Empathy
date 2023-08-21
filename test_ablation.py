import torch
from env import *
from model import *
from replay import *
from torch.distributions import Categorical
import torch.nn.functional as F
import wandb

device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

agents = [QMix().to(device) for _ in range(4)]
agents_target = [QMix().to(device) for _ in range(4)]
selfish_agents = [Qnet().to(device) for _ in range(4)]
selfish_agents_target = [Qnet().to(device) for _ in range(4)]
parameters = [list(agents[i].parameters()) + list(selfish_agents[i].parameters()) for i in range(4)]
# agents_clone = [Qnet().to(device) for _ in range(4)]
optim = [torch.optim.Adam(parameters[i], lr=1e-4) for i in range(4)]
agents_imagine = [Imagine().to(device) for _ in range(4)]
# optim = [torch.optim.Adam(agents[i].parameters(), lr=1e-4) for i in range(4)]
optim_imagine = [torch.optim.Adam(agents_imagine[i].parameters(), lr=1e-4) for i in range(4)]
optim_selfish = [torch.optim.Adam(selfish_agents[i].parameters(), lr=1e-4) for i in range(4)]
buffer = [ReplayBuffer(capacity=100000, id=i) for i in range(4)]
gamma = 0.99
max_episode = 100000
batch_size = 32
similar_factor = 1
alpha = 0.5
target_update = 10
minimal_size = 500
total_time = 0
epsilon = 0.02
delta = 0.1
beta = 1
count = 0

env = SnowDriftEnv()

wandb.init(project='Empathy', entity='kfq20', name='snowdrift_ablation')

def get_other_obs(obs, id):
    batch_size = obs.shape[0]
    total_other_obs = []
    total_condition = []
    observable = np.zeros((batch_size, 4))
    for i in range(4):
        if i == id:
            continue
        else:
            for j in range(batch_size):
                if np.sum(obs[j, i]) > 0:
                    observable[j, i] = 1
    other_obs = np.zeros((batch_size, env.player_num, env.channel, env.height, env.width-3))
    for i in range(4):
        i_obs = []
        if i == id:
            total_other_obs.append(np.zeros((batch_size, env.channel, env.height, env.width-3)))
            total_condition.append(np.zeros(batch_size))
            continue
        else:
            condition = observable[:, i] == 1
            condition = condition.astype(float)
            total_condition.append(condition)
            for j in range(batch_size):
                if condition[j]:
                    pos = np.where(obs[j][i] > 0)
                    x, y = pos[0][0], pos[1][0]
                    other_obs = obs[j][:, :, max(0, y-2):min(y+3, 5)]
                    if y-2 < 0:
                        other_obs = np.pad(other_obs, ((0, 0), (0, 0), (2-y, 0)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))
                    elif y+3 > 5:
                        other_obs = np.pad(other_obs, ((0, 0), (0, 0), (0, y-2)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))
                    i_obs.append(other_obs)
                else:
                    i_obs.append(np.zeros((env.channel, env.height, env.width-3)))
            total_other_obs.append(np.array(i_obs))
    return np.array(total_other_obs), np.array(total_condition)

# def get_other_obs(obs, id):
#     batch_size = obs.shape[0]
#     total_other_obs = []
#     total_condition = []
#     observable = np.sum(obs[:, :, id], axis=(2, 3)) > 0

#     for i in range(4):
#         if i == id:
#             total_other_obs.append([])
#             total_condition.append([])
#             continue

#         condition = observable[:, i].astype(float)
#         total_condition.append(condition)

#         i_obs = []
#         for j in range(batch_size):
#             if condition[j]:
#                 pos = np.where(obs[j, i] > 0)
#                 x, y = pos[0][0], pos[1][0]

#                 other_obs = obs[j, :, max(0, y - 2):min(y + 3, 5)]
#                 other_obs = np.pad(other_obs, ((0, 0), (0, 0), (-min(0, y-2), max(0, y+3-5))), 'constant')

#                 i_obs.append(other_obs)
#         total_other_obs.append(np.array(i_obs))

#     return np.array(total_other_obs), np.array(total_condition)

for ep in range(max_episode):
    obs = env.reset()
    if ep == 100:
        print(1)
    all_done = False
    total_reward = np.zeros(4)
    total_pick_num = 0
    # print('total_time', total_time)
    while not all_done:
        actions = []
        avail_actions = []
        avail_next_actions = []
        probs = []
        # obs = torch.tensor(obs, dtype=torch.float).to(device)
        for i in range(4):
            avail_action = env.__actionmask__(i)
            avail_action_index = np.nonzero(avail_action)[0]
            if np.random.random() < epsilon:
                action = np.random.choice(avail_action_index)
            else:
                state = torch.tensor(obs[i], dtype=torch.float).to(device)
                q_value = selfish_agents[i](state.unsqueeze(0))
                q_value[:, avail_action == 0] = -float("inf")
                action = q_value.argmax().item()
            actions.append(action)
            avail_actions.append(avail_action)
        next_obs, reward, done, pick_num = env.step(actions)
        if pick_num is not None:
            total_pick_num += pick_num
        for i in range(4):
            avail_next_action = env.__actionmask__(i)
            avail_next_actions.append(avail_next_action)
        # obs = next_obs
        total_reward += reward
        for i in range(4):
            other_action = np.ones(4) * -1
            for j in range(4):
                if j == i:
                    continue
                else:
                    if np.sum(obs[i][j]) > 0:
                        other_action[j] = actions[j]
            buffer[i].add(obs[i], actions[i], avail_actions[i], avail_next_actions[i], other_action, reward[i], next_obs[i], done[i])
        obs = next_obs
        all_done = done[0]
        if buffer[0].size() > minimal_size and done[0]:
            print(total_reward)
            total_loss = 0
            total_imagine_loss = 0
            wandb.log({'reward_1':total_reward[0], 'reward_2':total_reward[1], 'reward_3':total_reward[2], 'reward_4':total_reward[3], 'total_reward':sum(total_reward), 'pick_rubbish_num':total_pick_num})
            for i in range(4):
                obs, action, avail_action, avail_next_action, other_action, reward, next_obs, done = buffer[i].sample(min(buffer[i].size(), batch_size))
                other_obs, observable = get_other_obs(obs, i)
                next_other_obs, next_observable = get_other_obs(next_obs, i)
                obs = torch.tensor(obs, dtype=torch.float).to(device)
                other_action = torch.tensor(other_action, dtype=torch.int64).to(device)
                avail_action = torch.tensor(avail_action, dtype=torch.int64).to(device)
                action = torch.tensor(action).view(-1, 1).to(device)
                reward = torch.tensor(reward, dtype=torch.float).to(device).view(-1, 1)
                next_obs = torch.tensor(next_obs, dtype=torch.float).to(device)
                done = torch.tensor(done, dtype=torch.float).to(device).view(-1, 1)
                other_obs = torch.tensor(other_obs, dtype=torch.float).to(device) #(4, 32, 7, 8, 5)
                next_other_obs = torch.tensor(next_other_obs, dtype=torch.float).to(device)
                observable = torch.tensor(observable, dtype=torch.float).to(device)
                next_observable = torch.tensor(next_observable, dtype=torch.float).to(device)
                # q_value = agents[i](obs).gather(1, action)
                q_value_selfish = selfish_agents[i](obs).gather(1, action)

                # max_next_q_value = agents_target[i](next_obs).max(1)[0].view(-1, 1)
                q_target_selfish = selfish_agents_target[i](next_obs)
                q_target_selfish[avail_next_action == 0] = -99999
                q_target_selfish = q_target_selfish.max(1)[0].view(-1, 1)
                # factor = [0 for _ in range(4)]
                # max_other_next_q_value = []
                # total_other_next_q_value = 0
                # imagine_loss_1 = [0 for _ in range(4)]
                # imagine_loss_2 = [0 for _ in range(4)]
                # j_imagine_obs = [0 for _ in range(4)]
                # j_next_imagine_obs = [0 for _ in range(4)]
                # j_action = [0 for _ in range(4)]
                # j_q = [0 for _ in range(4)]
                # other_q_value = [0 for _ in range(4)]
                # other_q_target = [0 for _ in range(4)]
                # agents_clone = [Qnet().to(device) for _ in range(4)]
                # agents_target_clone = [Qnet().to(device) for _ in range(4)]
                # for j in range(4):
                #     if j == i:
                #         continue
                #     else:
                #         j_imagine_obs[j] = agents_imagine[i](other_obs[j])
                #         j_next_imagine_obs[j] = agents_imagine[i](next_other_obs[j])
                #         imagine_loss_2[j] = torch.mean(torch.abs(j_imagine_obs[j]-other_obs[j]))
                #         j_action[j] = other_action[:, j]
                #         aaa = (j_action[j] * observable[j]).to(torch.int64)
                #         j_action_onehot = F.one_hot(aaa, num_classes=6).squeeze()
                #         agents_clone[j].load_state_dict(selfish_agents[i].state_dict())
                #         agents_target_clone[j].load_state_dict(selfish_agents_target[i].state_dict())
                #         j_q_alone = agents_clone[j](j_imagine_obs[j])
                #         j_pi = F.softmax(j_q_alone, dim=1)
                #         j_q[j] = selfish_agents[i](j_imagine_obs[j].clone())
                #         m = j_pi[observable[j] == 1]
                #         n = j_action_onehot.to(torch.float)[observable[j] == 1]
                #         if m.shape[0] > 0:
                #             imagine_loss_1[j] = F.cross_entropy(m, n)
                #         p = j_q[j].gather(1, (j_action[j] * observable[j]).to(torch.int64).view(-1, 1))
                #         q = observable[j].view(-1, 1)
                #         other_q_value[j] = p * q
                #         next_j_q_value = agents_target_clone[j](j_next_imagine_obs[j])
                #         other_q_target[j] = next_j_q_value.max(1)[0].view(-1, 1) * next_observable[j].view(-1, 1)
                #         factor[j] = torch.tanh((q_value_selfish - other_q_value[j]) * observable[j].view(-1, 1)).detach()
                        # q_value = q_value + factor * j_q_value
                        # total_other_next_q_value = total_other_next_q_value + factor * max_next_j_q_value * next_observable[j].view(-1, 1)
                        # max_other_next_q_value.append(max_next_j_q_value)
                
                
                # total_q_value = agents[i](q_value_selfish, other_q_value, factor)
                # total_q_target = agents_target[i](q_target_selfish, other_q_target, factor)

                # q_target_selfish = reward + gamma * (max_next_selfish_q_value) * (1 - done)
                target = reward + gamma * (q_target_selfish) * (1 - done)
                loss = torch.mean(F.mse_loss(q_value_selfish, target))

                # selfish_dqn_loss = torch.mean(F.mse_loss(q_value_selfish, q_target_selfish))
                optim[i].zero_grad()
                loss.backward(retain_graph=True)
                optim[i].step()
                total_loss += loss.item()

                # imagine_loss = (1-delta) * sum(imagine_loss_1) + delta * sum(imagine_loss_2)
                # total_imagine_loss += imagine_loss
                # optim_imagine[i].zero_grad()
                # imagine_loss.backward()
                # optim_imagine[i].step()

            count += 1
            if count % target_update == 0:
                for i in range(4):
                    selfish_agents_target[i].load_state_dict(selfish_agents[i].state_dict())
                    # agents_target[i].load_state_dict(agents[i].state_dict())
            wandb.log({'dqn_loss': total_loss})

            