import torch
from env import *
from model import *
from replay import *
from torch.distributions import Categorical
import torch.nn.functional as F
import wandb

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

symp_agents = [Qnet().to(device) for _ in range(4)]
target_symp_agents = [Qnet().to(device) for _ in range(4)]
selfish_agents = [RNN().to(device) for _ in range(4)]
target_selfish_agents = [RNN().to(device) for _ in range(4)]
for i in range(4):
    target_symp_agents[i].load_state_dict(symp_agents[i].state_dict())
    target_selfish_agents[i].load_state_dict(selfish_agents[i].state_dict())

agents_imagine = [Imagine().to(device) for _ in range(4)]

optim_symp = [torch.optim.Adam(symp_agents[i].parameters(), lr=1e-4) for i in range(4)]
optim_imagine = [torch.optim.Adam(agents_imagine[i].parameters(), lr=1e-4) for i in range(4)]
optim_selfish = [torch.optim.Adam(selfish_agents[i].parameters(), lr=1e-4) for i in range(4)]
buffer = [ReplayBuffer(capacity=100000, id=i) for i in range(4)]

gamma = 0.99
max_episode = 300000
anneal_eps = 50000
batch_size = 64
similar_factor = 1
alpha = 0.5
target_update = 10
minimal_size = 500
total_time = 0
epsilon = 0.95
min_epsilon = 0.05
anneal_epsilon = (epsilon - min_epsilon) / anneal_eps
delta = 0.5
beta = 1
count = 0
env_obs_mode = 'complete'
window_height = 8
window_width = 5
cnt = 0

env = SnowDriftEnv()

wandb.init(project='Empathy', entity='kfq20', name='cleanup')

for ep in range(max_episode):
    obs = env.reset()
    all_done = False
    total_reward = np.zeros(4)
    total_pick_num = 0
    last_action = np.zeros((4, 6))
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
                state = torch.flatten(torch.tensor(obs[i], dtype=torch.float)).to(device)
                q_value = symp_agents[i](state.unsqueeze(0))
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
                    other_action[j] = actions[j]
            buffer[i].add(obs, actions[i], last_action, avail_actions[i], avail_next_actions[i], other_action, reward[i], next_obs, done[i])
        actions = np.array(actions)
        one_hot = np.zeros((len(actions), 6))
        one_hot[np.arange(len(actions)), actions] = 1
        last_action = one_hot
        obs = next_obs
        all_done = done[0]
        if buffer[0].size() > minimal_size and done[0]:
            cnt += 1
            # print(total_reward)
            total_self_loss = 0
            total_symp_loss = 0
            total_imagine_loss = 0
            wandb.log({'reward_1':total_reward[0], 'reward_2':total_reward[1], 'reward_3':total_reward[2], 'reward_4':total_reward[3], 'total_reward':sum(total_reward), 'pick_num':total_pick_num})
            for i in range(4):
                obs, action, avail_action, avail_next_action, other_action, reward, next_obs, done = buffer[i].sample(min(buffer[i].size(), batch_size))

                obs_i = torch.tensor(obs[:, i, :, :, :], dtype=torch.float).to(device)
                next_obs_i = torch.tensor(next_obs[:, i, :, :, :], dtype=torch.float).to(device)

                obs = torch.tensor(obs, dtype=torch.float).permute(1,0,2,3,4).to(device)
                other_action = torch.tensor(other_action, dtype=torch.int64).to(device)
                avail_action = torch.tensor(avail_action, dtype=torch.int64).to(device)
                action = torch.tensor(action).view(-1, 1).to(device)
                reward = torch.tensor(reward, dtype=torch.float).to(device).view(-1, 1)
                next_obs = torch.tensor(next_obs, dtype=torch.float).permute(1,0,2,3,4).to(device)
                done = torch.tensor(done, dtype=torch.float).to(device).view(-1, 1)

                selfish_q_value = selfish_agents[i](obs_i).gather(1, action)
                selfish_q_target = target_selfish_agents[i](next_obs_i)
                selfish_q_target[avail_next_action == 0] = -99999
                selfish_q_target = selfish_q_target.max(1)[0].view(-1, 1)

                symp_q_value = symp_agents[i](obs_i).gather(1, action)
                symp_q_target = target_symp_agents[i](next_obs_i)
                symp_q_target[avail_next_actions == 0] = -99999
                symp_q_target = symp_q_target.max(1)[0].view(-1, 1)

                factor = [0 for _ in range(4)]
                max_other_next_q_value = []
                total_other_next_q_value = 0
                imagine_loss_1 = [0 for _ in range(4)]
                imagine_loss_2 = [0 for _ in range(4)]
                j_imagine_obs = [0 for _ in range(4)]
                j_next_imagine_obs = [0 for _ in range(4)]
                j_action = [0 for _ in range(4)]
                j_q = [0 for _ in range(4)]
                other_q_value = [0 for _ in range(4)]
                other_q_target = [0 for _ in range(4)]
                other_reward = [0 for _ in range(4)]
                agents_clone = [Qnet().to(device) for _ in range(4)]
                # agents_target_clone = [Qnet().to(device) for _ in range(4)]
                for j in range(4):
                    if j == i:
                        continue
                    else:
                        j_imagine_obs[j] = agents_imagine[i](obs[j])
                        j_next_imagine_obs[j] = agents_imagine[i](next_obs[j])
                        imagine_loss_2[j] = torch.mean(torch.abs(j_imagine_obs[j]-obs[j]))
                        j_action[j] = other_action[:, j]
                        aaa = (j_action[j]).to(torch.int64)
                        j_action_onehot = F.one_hot(aaa, num_classes=6).squeeze()
                        agents_clone[j].load_state_dict(selfish_agents[i].state_dict())
                        # agents_target_clone[j].load_state_dict(selfish_agents_target[i].state_dict())
                        j_q[j] = agents_clone[j](j_imagine_obs[j])
                        j_pi = F.softmax(j_q[j], dim=1)
                        # j_q[j] = selfish_agents[i](j_imagine_obs[j].clone())
                        # m = j_pi[observable[j] == 1]
                        # n = j_action_onehot.to(torch.float)[observable[j] == 1]
                        imagine_loss_1[j] = F.cross_entropy(j_pi, j_action_onehot.to(torch.float))
                        # p = j_q[j].gather(1, (j_action[j] * observable[j]).to(torch.int64).view(-1, 1))
                        # q = observable[j].view(-1, 1)
                        other_q_value[j] = j_q[j].gather(1, (j_action[j]).to(torch.int64).view(-1, 1)).view(-1, 1).detach()
                        next_j_q_value = agents_clone[j](j_next_imagine_obs[j])
                        other_q_target[j] = next_j_q_value.max(1)[0].view(-1, 1).detach()
                        q_max = other_q_target[j]
                        other_reward[j] = (other_q_value[j] - gamma * q_max).detach()
                        l1_score = env.apple_reward / torch.sum(abs(other_reward[j]))
                        x = torch.mean(abs(selfish_q_value))
                        factor[j] = torch.tanh(((selfish_q_value - other_q_value[j]) / torch.mean(abs(selfish_q_value))).view(-1, 1)).detach()
                        # q_value = q_value + factor * j_q_value
                        # total_other_next_q_value = total_other_next_q_value + factor * max_next_j_q_value * next_observable[j].view(-1, 1)
                        # max_other_next_q_value.append(max_next_j_q_value)
                
                # print(other_reward)
                selfish_target = reward + gamma * selfish_q_target * (1 - done)
                selfish_loss = torch.mean(F.mse_loss(selfish_q_value, selfish_target))

                target_reward = 0
                for k in range(4):
                    if isinstance(factor[k], torch.Tensor):
                        target_reward += (1 - factor[k]) * reward + factor[k] * other_reward[k]
                symp_target = target_reward + gamma * symp_q_target * (1-done)
                symp_loss = torch.mean(F.mse_loss(symp_q_value, symp_target))

                optim_selfish[i].zero_grad()
                selfish_loss.backward(retain_graph=True)
                optim_selfish[i].step()
                total_self_loss += selfish_loss.item()

                optim_symp[i].zero_grad()
                symp_loss.backward(retain_graph=True)
                optim_symp[i].step()
                total_symp_loss += symp_loss.item()

                imagine_loss = (1-delta) * sum(imagine_loss_1) + delta * sum(imagine_loss_2)
                total_imagine_loss += imagine_loss
                optim_imagine[i].zero_grad()
                imagine_loss.backward()
                optim_imagine[i].step()

            if cnt % 100 == 0:
                print("eps: ", ep, "==========================")
                print("my reward:", reward.reshape(reward.shape[0]))
                print("other reward: ", other_reward)
                print("my action: ", action)
                print("other action", other_action)
            count += 1
            if count % target_update == 0:
                for i in range(4):
                    target_selfish_agents[i].load_state_dict(selfish_agents[i].state_dict())
                    target_symp_agents[i].load_state_dict(symp_agents[i].state_dict())
            wandb.log({'self_loss': total_self_loss, 'symp_loss': total_symp_loss, 'imagine_loss':imagine_loss})

    epsilon = epsilon - anneal_epsilon if epsilon > min_epsilon else epsilon