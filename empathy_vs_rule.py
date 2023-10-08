import torch
from env import *
from model_sd import *
from replay import *
import torch.nn.functional as F
import wandb
from torch.distributions import Categorical
import time

# device = torch.device("cpu")
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

MAX_REWARD = 1
MIN_REWARD = -1

gamma = 0.99
max_episode = 100000
anneal_eps = 5000
batch_size = 32
# similar_factor = 1
alpha = 0.5
target_update = 10
minimal_size = 50
total_time = 0
epsilon = 0.01
# min_epsilon = 0.01
# anneal_epsilon = (epsilon - min_epsilon) / anneal_eps
delta = 0
beta = 1
count = 0
env_obs_mode = 'complete'
window_height = 8
window_width = 5

env = ModifiedCleanupEnv()
config = {}
config["channel"] = env.channel
config["height"] = env.height
config["width"] = env.width - 3
# config["width"] = 15 # cleanup 16*15
config["player_num"] = env.player_num
config["action_space"] = env.action_space

symp_agents = [ActorCritic(config).to(device) for _ in range(4)]
# target_symp_agents = [ActorCritic(config).to(device) for _ in range(4)]
selfish_agents = [Qself(config).to(device) for _ in range(4)]
target_selfish_agents = [Qself(config).to(device) for _ in range(4)]
# target_symp_agents[i].load_state_dict(symp_agents[i].state_dict())
for i in range(4):
    # target_symp_agents[i].load_state_dict(symp_agents[i].state_dict())
    target_selfish_agents[i].load_state_dict(selfish_agents[i].state_dict())

agents_imagine = [Imagine().to(device) for _ in range(4)]

optim_symp = [torch.optim.Adam(symp_agents[i].parameters(), lr=1e-4) for i in range(4)]
optim_imagine = [torch.optim.Adam(agents_imagine[i].parameters(), lr=1e-4) for i in range(4)]
optim_selfish = [torch.optim.Adam(selfish_agents[i].parameters(), lr=1e-4) for i in range(4)]
buffer = [ReplayBuffer(capacity=5000, id=i) for i in range(4)]

def counterfactual_factor(my_id, other_id, agents, state, last_action, my_action, real_q_value, observable, avail_action):
    counterfactual_result = []
    log = []
    counterfactual_result.append(real_q_value)
    log.append(real_q_value.item())
    other_action = copy.deepcopy(last_action[:, env.action_space*other_id:env.action_space*(other_id+1)])
    virtual_action = torch.nonzero(other_action == 0).view(1, env.action_space-1, 2)
    other_action.fill_(0)
    for i in range(env.action_space-1):
        cur_try_action_pos = virtual_action[:, i, :] # 32*2
        other_action[cur_try_action_pos[:, 0], cur_try_action_pos[:, 1]] = 1 # try another action
        last_action[:, env.action_space*other_id:env.action_space*(other_id+1)] = other_action
        # cf_input = torch.cat((state, last_action), dim=1)
        cf_q_value = agents[my_id](state, last_action).gather(1, my_action)
        counterfactual_result.append(cf_q_value)
        log.append(cf_q_value.item())
        other_action.fill_(0)
    cf_result = torch.transpose(torch.stack(counterfactual_result), 0, 1)
    min_value, _ = cf_result.min(dim=1)
    max_value, _ = cf_result.max(dim=1)
    factor = 2 * (real_q_value - min_value) / (max_value - min_value) - 1
    if_log = random.random()
    if if_log < 0.001:
        print("other action", virtual_action)
        print(log)
    return factor.view(-1, 1).to(device).detach() * torch.tensor(observable).to(device).view(-1, 1)

def get_loss(batch, id, eps):
    obs, action, avail_action, avail_next_action, other_action, reward, next_obs, done, pad = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6], batch[7], batch[8]
    episode_num = obs.shape[0]
    self_loss = 0
    symp_loss = 0
    imagine_loss = 0
    # random_start = random.randint(0, env.final_time-51)
    for transition_idx in range(env.final_time):
        inputs, inputs_next = [], []
        obs_transition, next_obs_transition, action_transition, avail_action_transition = obs[:, transition_idx, :, :, :, :], next_obs[:, transition_idx, :, :, :, :], action[:, transition_idx, :], avail_action[:, transition_idx]
        avail_next_action_transition = avail_next_action[:, transition_idx]
        reward_transition = torch.tensor(reward[:, transition_idx], dtype=torch.float).view(-1, 1).to(device)
        done_transition = torch.tensor(done[:, transition_idx], dtype=torch.float).view(-1, 1).to(device)
        pad_transition = torch.tensor(pad[:, transition_idx], dtype=torch.float).view(-1, 1).to(device)
        other_action_transition = torch.tensor(other_action[:, transition_idx, :], dtype=torch.float).to(device)
        obs_i = obs_transition[:, id, :, :, :]
        done_player = np.zeros((batch_size, env.player_num))
        for j in range(4):
            if j == id:
                continue
            else:
                done_player[:, j] = np.sum(obs_i[:, j, :, :], axis=(1, 2))
        next_obs_i = next_obs_transition[:, id, :, :, :]
        action_i = torch.tensor(action_transition[:, id]).view(-1, 1).to(device)
        # inputs_next.append(next_obs_transition)
        if transition_idx == 0:
            last_action = np.zeros((episode_num, 4, env.action_space))
        else:
            last_action = action[:, transition_idx - 1]
            last_action = np.eye(env.action_space)[last_action] # one-hot
        
        # if transition_idx == env.final_time - 1:
        #     next_action = np.zeros((episode_num, 4, 6))
        # else:
        #     next_action = action[:, transition_idx + 1]
        #     next_action = np.eye(6)[next_action]

        this_action = action[:, transition_idx]
        this_action = np.eye(env.action_space)[this_action]

        state = torch.tensor(obs_i, dtype=torch.float).to(device)
        next_state = torch.tensor(next_obs_i, dtype=torch.float).to(device)
        last_action = torch.tensor(last_action, dtype=torch.float).to(device).view(episode_num, -1)
        # next_action = torch.tensor(next_action, dtype=torch.float).to(device).view(episode_num, 4*6)
        this_action = torch.tensor(this_action, dtype=torch.float).to(device).view(episode_num, -1)

        # selfish_inputs = torch.cat((state, last_action), dim=1)
        # selfish_next_inputs = torch.cat((next_state, this_action), dim=1)
        self_q_value = selfish_agents[id](state, last_action).gather(1, action_i)
        self_q_target = target_selfish_agents[id](next_state, this_action)
        self_q_value_next = selfish_agents[id](next_state, this_action)
        self_q_value_next[avail_next_action_transition == 0] = -99999
        max_action = torch.argmax(self_q_value_next, dim=1).view(-1, 1)

        # self_q_target[avail_next_action_transition == 0] = -99999
        # self_q_target = self_q_target.max(1)[0].view(-1, 1)
        self_q_target = self_q_target.gather(1, max_action)

        # factor = [0 for _ in range(4)]
        imagine_loss_1 = [0 for _ in range(4)]
        imagine_loss_2 = [0 for _ in range(4)]
        j_imagine_obs = [0 for _ in range(4)]
        j_next_imagine_obs = [0 for _ in range(4)]
        j_action = [0 for _ in range(4)]
        j_q = [0 for _ in range(4)]
        agents_clone = [Qself(config).to(device) for _ in range(4)]

        for j in range(4):
            if j == id:
                continue
            else:
                obs_j = torch.tensor(obs_transition[:, j, :, :, :], dtype=torch.float).to(device)
                next_obs_j = torch.tensor(next_obs_transition[:, j, :, :, :], dtype=torch.float).to(device)
                j_imagine_obs[j] = agents_imagine[id](obs_j).view(batch_size, env.channel, window_height, window_width)
                j_next_imagine_obs[j] = agents_imagine[id](next_obs_j).view(batch_size, env.channel, window_height, window_width)
                imagine_loss_2[j] = torch.mean(torch.abs(j_imagine_obs[j]-obs_j))
                j_action[j] = other_action_transition[:, j].to(torch.int64)
                j_action_onehot = F.one_hot(j_action[j], num_classes=env.action_space).squeeze()
                agents_clone[j].load_state_dict(selfish_agents[id].state_dict())
                # j_imagine_obs[j] = j_imagine_obs[j].view(episode_num, -1)
                # j_inputs = torch.cat((j_imagine_obs[j], this_action), dim=1)
                j_q[j] = agents_clone[j](j_imagine_obs[j], last_action)
                j_pi = F.softmax(j_q[j], dim=1)
                imagine_loss_1[j] = F.cross_entropy(j_pi, j_action_onehot.to(torch.float))

        self_target = reward_transition + gamma * self_q_target * (1 - done_transition)
        self_td_error = (self_q_value - self_target)
        mask_self_td_error = (1 - pad_transition) * self_td_error
        self_loss += mask_self_td_error ** 2

        # if transition_idx == 20 and eps % 100 == 0 and id == 0:
        #     print("real obs:", obs_transition[0, 1, :, :, :])
        #     print("imagine obs:", j_imagine_obs[1][0])
        # if transition_idx == 20 and eps % 10 == 0 and id == 0:
        #     print("eps: ", eps, "===========================")
        #     print("factor: ",factor)
        #     print("reward: ", reward_transition)
        #     # print("other reward: ", other_reward)
        #     print("action: ", action_transition)
        imagine_loss += (1 - delta) * sum(imagine_loss_1) + delta * sum(imagine_loss_2)

    return self_loss, symp_loss, imagine_loss

wandb.init(project='Empathy', entity='kfq20', name='dqn vs 2good')
step_time = 0
a2c_time = 0
offline_time = 0
test_time = 0
# loss_time = 0
# forward_time = 0
for ep in range(max_episode):
    print("eps", ep, "=====================================")
    # ep_start = time.time()
    o, a, other_a, r, o_next, avail_a, avail_a_next, pad, d = [[] for _ in range(4)], [[] for _ in range(4)], [[] for _ in range(4)], [[] for _ in range(4)], [[] for _ in range(4)], [[] for _ in range(4)], [[] for _ in range(4)], [[] for _ in range(4)], [[] for _ in range(4)]
    obs = env.reset()
    all_done = False
    total_reward = np.zeros(4)
    total_hunt_stag_num = 0
    total_hunt_hare_num = 0
    total_collect_waste_num = 0
    total_collect_apple_num = 0
    last_action = np.zeros((4, env.action_space))
    step = 0
    log_probs = []
    values = []
    rewards = []

    for i in range(4):
        symp_agents[i].init_hidden(1)
    # print('total_time', total_time)
    while not all_done:
        actions = []
        avail_actions = []
        avail_next_actions = []
        probs = []
        all_log_prob = []
        all_value = []
        last_action = torch.flatten(torch.tensor(last_action, dtype=torch.float)).to(device)
        # obs = torch.tensor(obs, dtype=torch.float).to(device)
        # forward_start = time.time()
        for i in range(4):
            if i == 0 or i == 1:
                avail_action = env.__actionmask__(i)
                avail_action_index = np.nonzero(avail_action)[0]

                state = torch.tensor(obs[i], dtype=torch.float).to(device)
                # input = torch.cat((state, last_action))
                hidden_state = symp_agents[i].hidden.to(device)

                action_prob, state_value, symp_agents[i].hidden = symp_agents[i](state.unsqueeze(0), hidden_state)
                action_prob[:, avail_action == 0] = -999999
                action_prob = F.softmax(action_prob, dim=-1)
                dist = Categorical(action_prob[0])
                if np.random.random() < epsilon:
                    action = torch.tensor(np.random.choice(avail_action_index), dtype=torch.int8).to(device)
                else:
                    action = dist.sample()
                actions.append(action.item())
                avail_actions.append(avail_action)
                one_log_prob = dist.log_prob(action)
                all_log_prob.append(one_log_prob)
                all_value.append(state_value)
            
            else: # clean waste
                avail_action = env.__actionmask__(i)
                avail_action_index = np.nonzero(avail_action)[0]
                player_i_pos = np.argwhere(obs[i][i] == 1)[0]
                if np.sum(obs[i][-2]) == 0:
                    action = np.random.choice(avail_action_index)
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
                avail_actions.append(avail_action)
            
            # else: # collect apple
            #     avail_action = env.__actionmask__(i)
            #     avail_action_index = np.nonzero(avail_action)[0]
            #     player_i_pos = np.argwhere(obs[i][i] == 1)[0]
            #     if np.sum(obs[i][-1]) == 0:
            #         action = np.random.choice(avail_action_index)
            #     else:
            #         waste_pos = np.argwhere(obs[i][-1] == 1)
            #         distances = np.sum(np.abs(waste_pos - player_i_pos), axis=1)
            #         nearest_waste_pos = waste_pos[np.argmin(distances)]
            #         if nearest_waste_pos[0] < player_i_pos[0]:
            #             action = 0  # 上
            #         elif nearest_waste_pos[0] > player_i_pos[0]:
            #             action = 1  # 下
            #         elif nearest_waste_pos[1] < player_i_pos[1]:
            #             action = 2  # 左
            #         elif nearest_waste_pos[1] > player_i_pos[1]:
            #             action = 3  # 右
            #         else:
            #             action = 6  # collect apple
            #     if avail_action[action] == 0: # not avail, just stay
            #         action = 4
            #     actions.append(action)
            #     avail_actions.append(avail_action)

            # else: # random
            #     avail_action = env.__actionmask__(i)
            #     avail_action_index = np.nonzero(avail_action)[0]
            #     action = np.random.choice(avail_action_index)
            #     actions.append(action)
            #     avail_actions.append(avail_action)

        # forward_end = time.time()
        # forward_time += forward_end - forward_start
        next_obs, reward, done, info = env.step(actions)
        collect_waste_num = info[0]
        collect_apple_num = info[1]
        total_collect_waste_num += collect_waste_num
        total_collect_apple_num += collect_apple_num

        log_probs.append(all_log_prob)
        values.append(all_value)
        rewards.append(reward)
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
            o[i].append(obs)
            a[i].append(actions)
            r[i].append(reward[i])
            o_next[i].append(next_obs)
            avail_a[i].append(avail_actions[i])
            avail_a_next[i].append(avail_next_actions[i])
            other_a[i].append(other_action)
            d[i].append(done[i])
            pad[i].append(0.)
            
        actions = np.array(actions)
        one_hot = np.zeros((len(actions), env.action_space))
        one_hot[np.arange(len(actions)), actions] = 1
        last_action = one_hot
        obs = next_obs     
        all_done = done[-1]
        step += 1
        
        for i in range(2):
            buffer[i].add(o[i], a[i], avail_a[i], avail_a_next[i], other_a[i], r[i], o_next[i], d[i], pad[i])

    total_actor_loss = 0
    total_critic_loss = 0

    for i in range(2):
        returns = np.zeros(len(rewards))
        advantages = np.zeros(len(rewards))
        R = 0
        other_reward = [0 for _ in range(4)]
        other_reward_log = [0 for _ in range(4)]
        factor_log = [0 for _ in range(4)]
        j_imagine_obs = [0 for _ in range(4)]
        j_next_imagine_obs = [0 for _ in range(4)]
        agents_clone = [Qself(config).to(device) for _ in range(4)]
        agents_target_clone = [Qself(config).to(device) for _ in range(4)]
        j_action = [0 for _ in range(4)]
        j_q = [0 for _ in range(4)]
        other_q_value = [0 for _ in range(4)]
        other_q_target = [0 for _ in range(4)]
        factor = [0 for _ in range(4)]
        last_factor = [0 for _ in range(4)]
        total_factor = [0 for _ in range(4)]
        weighted_rewards = [0 for _ in range(env.final_time)]
        with torch.no_grad():
            for t in range(len(rewards)):
                if t == 0:
                    last_action = np.zeros((1, 4, env.action_space))
                else:
                    last_action = a[i][t - 1]
                    last_action = np.eye(env.action_space)[last_action] # one-hot

                obs_transition = o[i][t]
                next_obs_transition = o_next[i][t]
                action_transition = a[i][t]
                reward_transition = r[i][t]
                obs_i = obs_transition[i]
                next_obs_i = next_obs_transition[i]
                state = torch.tensor(obs_i, dtype=torch.float).to(device).unsqueeze(0)
                next_state = torch.tensor(next_obs_i, dtype=torch.float).to(device).unsqueeze(0)

                last_action = torch.tensor(last_action, dtype=torch.float).to(device).view(1, -1)
                action_i = torch.tensor(action_transition[i]).view(-1, 1).to(device)
                # selfish_inputs = torch.cat((state, last_action), dim=1)

                self_q_value = selfish_agents[i](state, last_action).gather(1, action_i)
                player_in_view = np.zeros((1, env.player_num))
                for j in range(4):
                    if j == i:
                        continue
                    else:
                        player_in_view[:, j] = np.sum(obs_i[j, :, :], axis=(0, 1))

                this_action = a[i][t]
                this_action = np.eye(env.action_space)[this_action]
                this_action = torch.tensor(this_action, dtype=torch.float).to(device).view(1, -1)
                other_action_transition = torch.tensor(other_a[i][t], dtype=torch.int64).to(device)

                # get others' reward and weights needed
                for j in range(4):
                    if j == i:
                        continue
                    else:
                        reward_j = r[j][t]
                        other_reward[j] = reward_j

                        if t == 0:
                            factor[j] = torch.zeros((1, 1)).to(device).detach()
                            last_factor[j] = torch.zeros((1, 1)).to(device).detach()
                            factor_log[j] = 0
                        else:
                            # t_factor = torch.ones((1, 1)).to(device).detach()
                            # t_factor = cleanup_compute_factor(state, state_j, next_state, next_state_j, action_i, action_j, i, j)
                            t_factor = counterfactual_factor(i, j, selfish_agents, state, last_action, action_i, self_q_value, player_in_view[:, j], avail_a[j][t])
                            factor[j] = 0.8*last_factor[j] + 0.2*t_factor
                            # factor[j] = t_factor
                            # factor[j] += counterfactual_factor(i, j, selfish_agents, state, last_action, action_i, self_q_value, player_in_view[:, j])
                            factor_log[j] = factor[j][0].item()
                            total_factor[j] += factor[j][0].item()
                            last_factor[j] = factor[j]

                if i == 0 and ep % 10 == 0:
                    print("act", action_transition)
                    # print("infer reward", other_reward_log)
                    print("factor", factor_log)
            
            # for m in range(4):
            #     factor[m] /= len(rewards)
                weighted_reward = 0
                # weighted_reward = reward_transition
                for k in range(4):
                    if k != i:
                        weighted_reward += (1-factor[k])*reward_transition + factor[k] * other_reward[k]
                        # weighted_reward += other_reward[k]

                weighted_rewards[t] = weighted_reward

        if i == 0:
            wandb.log({'factor_2':total_factor[1]/100, 'factor_3':total_factor[2]/100, 'factor_4':total_factor[3]/100})

        for t in reversed(range(len(rewards))):
            R = weighted_rewards[t] + 0.99 * R
            returns[t] = R
            advantages[t] = R - values[t][i].item()

        returns = torch.FloatTensor(returns).to(device)
        advantages = torch.FloatTensor(advantages).to(device)

        actor_loss = 0
        critic_loss = 0

        for log_prob, value, return_, advantage in zip(log_probs, values, returns, advantages):
            actor_loss -= log_prob[i] * advantage
            critic_loss += F.smooth_l1_loss(value[i].squeeze(), return_)

        total_actor_loss += actor_loss
        total_critic_loss += critic_loss
        loss = actor_loss + critic_loss
        optim_symp[i].zero_grad()
        loss.backward()
        optim_symp[i].step()

    wandb.log({'actor loss': total_actor_loss, 'critic loss': total_critic_loss, 'reward_1':total_reward[0], 'reward_2':total_reward[1], 'reward_3':total_reward[2], 'reward_4':total_reward[3], 'total_reward':sum(total_reward), 'waste_num':total_collect_waste_num, 'apple_num':total_collect_apple_num})
    if buffer[0].size() > minimal_size:
        # print(total_reward)
        total_self_loss = 0
        total_symp_loss = 0
        total_imagine_loss = 0
        # wandb.log({'reward_1':total_reward[0], 'reward_2':total_reward[1], 'reward_3':total_reward[2], 'reward_4':total_reward[3], 'total_reward':sum(total_reward), 'hare_num':total_hunt_hare_num, 'stag_num':total_hunt_stag_num})
        for i in range(2):
            obs, action, avail_action, avail_next_action, other_action, reward, next_obs, done, pad = buffer[i].sample(min(buffer[i].size(), batch_size))
            episode_num = obs.shape[0]
            batch = [obs, action, avail_action, avail_next_action, other_action, reward, next_obs, done, pad]
            self_loss, symp_loss, imagine_loss = get_loss(batch, i, ep)
            
            self_loss = torch.mean(self_loss)
            # symp_loss = torch.mean(symp_loss)
            optim_selfish[i].zero_grad()
            self_loss.backward()
            optim_selfish[i].step()
            total_self_loss += self_loss.item()

            # optim_symp[i].zero_grad()
            # symp_loss.backward(retain_graph=True)
            # optim_symp[i].step()
            # total_symp_loss += symp_loss.item()

            optim_imagine[i].zero_grad()
            imagine_loss.backward()
            optim_imagine[i].step()
            total_imagine_loss += imagine_loss
            # loss_end = time.time()
            # loss_time += loss_end - loss_time_start
            
        # if cnt % 100 == 0:
        #     print("eps: ", ep, "==========================")
        #     print("my reward:", reward.reshape(reward.shape[0]))
        #     # print("other reward: ", other_reward)
        #     print("my action: ", action)
        #     print("other action", other_action)
        count += 1
        if count % target_update == 0:
            for i in range(4):
                target_selfish_agents[i].load_state_dict(selfish_agents[i].state_dict())
                # target_symp_agents[i].load_state_dict(symp_agents[i].state_dict())
        # wandb.log({'self_loss': total_self_loss, 'symp_loss': total_symp_loss, 'imagine_loss':imagine_loss})