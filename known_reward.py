import torch
from env import *
from model_sd import *
from replay import *
import torch.nn.functional as F
import wandb

# device = torch.device("cpu")
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# symp_agents = [RNN().to(device) for _ in range(4)]
# target_symp_agents = [RNN().to(device) for _ in range(4)]
# selfish_agents = [Qself().to(device) for _ in range(4)]
# target_selfish_agents = [Qself().to(device) for _ in range(4)]
# for i in range(4):
#     target_symp_agents[i].load_state_dict(symp_agents[i].state_dict())
#     target_selfish_agents[i].load_state_dict(selfish_agents[i].state_dict())

# agents_imagine = [Imagine().to(device) for _ in range(4)]

# optim_symp = [torch.optim.Adam(symp_agents[i].parameters(), lr=1e-4) for i in range(4)]
# optim_imagine = [torch.optim.Adam(agents_imagine[i].parameters(), lr=1e-4) for i in range(4)]
# optim_selfish = [torch.optim.Adam(selfish_agents[i].parameters(), lr=1e-4) for i in range(4)]
# buffer = [ReplayBuffer(capacity=5000, id=i) for i in range(4)]

gamma = 0.95
max_episode = 100000
anneal_eps = 5000
batch_size = 32
# similar_factor = 1
alpha = 0.5
target_update = 10
minimal_size = 50
total_time = 0
epsilon = 0.99
min_epsilon = 0.01
anneal_epsilon = (epsilon - min_epsilon) / anneal_eps
delta = 0
beta = 1
count = 0
env_obs_mode = 'complete'
window_height = 16
window_width = 15
cnt = 0

env = CleanupEnv()
config = {}
config["channel"] = env.channel
config["height"] = env.height
# config["width"] = env.width - 3
config["width"] = 15 # cleanup 16*15
config["player_num"] = env.player_num
config["action_space"] = env.action_space

symp_agents = [RNN(config).to(device) for _ in range(4)]
target_symp_agents = [RNN(config).to(device) for _ in range(4)]
selfish_agents = [Qself(config).to(device) for _ in range(4)]
target_selfish_agents = [Qself(config).to(device) for _ in range(4)]
for i in range(4):
    target_symp_agents[i].load_state_dict(symp_agents[i].state_dict())
    target_selfish_agents[i].load_state_dict(selfish_agents[i].state_dict())

agents_imagine = [Imagine().to(device) for _ in range(4)]

optim_symp = [torch.optim.Adam(symp_agents[i].parameters(), lr=1e-4) for i in range(4)]
optim_imagine = [torch.optim.Adam(agents_imagine[i].parameters(), lr=1e-4) for i in range(4)]
optim_selfish = [torch.optim.Adam(selfish_agents[i].parameters(), lr=1e-4) for i in range(4)]
buffer = [ReplayBuffer(capacity=5000, id=i) for i in range(4)]

def cleanup_compute_factor(my_state, other_state, my_next_state, other_next_state, my_action, other_action, i, j):
    other_state = other_state.detach().cpu().numpy()
    other_next_state = other_next_state.detach().cpu().numpy()
    my_action = my_action.squeeze().detach().cpu().numpy()
    other_action = other_action.squeeze().detach().cpu().numpy()
    factor = []
    for k in range(32):
        my_value = 0
        other_value = 0
        my_pos = np.argwhere(my_state[k, i, :, :] == 1)
        my_next_pos = np.argwhere(my_next_state[k, i, :, :] == 1)
        other_pos = np.argwhere(other_state[k, j, :, :] == 1)
        other_next_pos = np.argwhere(other_next_state[k, j, :, :] == 1)
        my_rubbish_pos = np.argwhere(my_state[k, -2, :, :] == 1)
        my_apple_pos = np.argwhere(my_state[k, -1, :, :] == 1)
        my_next_rubbish_pos = np.argwhere(my_next_state[k, -2, :, :] == 1)
        my_next_apple_pos = np.argwhere(my_next_state[k, -1, :, :] == 1)
        other_rubbish_pos = np.argwhere(other_state[k, -2, :, :] == 1)
        other_apple_pos = np.argwhere(other_state[k, -1, :, :] == 1)
        other_next_rubbish_pos = np.argwhere(other_next_state[k, -2, :, :] == 1)
        other_next_apple_pos = np.argwhere(other_next_state[k, -1, :, :] == 1)
        my_rubbish_dist = np.sum(np.abs(my_pos - my_rubbish_pos), axis=1)
        my_apple_dist = np.sum(np.abs(my_pos - my_apple_pos), axis=1)
        my_next_rubbish_dist = np.sum(np.abs(my_next_pos - my_next_rubbish_pos), axis=1)
        my_next_apple_dist = np.sum(np.abs(my_next_pos - my_next_apple_pos), axis=1)
        other_rubbish_dist = np.sum(np.abs(other_pos - other_rubbish_pos), axis=1)
        other_apple_dist = np.sum(np.abs(other_pos - other_apple_pos), axis=1)
        other_next_rubbish_dist = np.sum(np.abs(other_next_pos - other_next_rubbish_pos), axis=1)
        other_next_apple_dist = np.sum(np.abs(other_next_pos - other_next_apple_pos), axis=1)
        if my_action[k] == 5:
            if my_state[k, -2, my_pos[0][0], my_pos[0][1]] == 1: # I pick up the rubbish
                my_value = 2
            elif my_state[k, -1, my_pos[0][0], my_pos[0][1]] == 1: # I pick up the apple
                my_value = -2
        else:
            my_value = np.sum(np.exp(-my_next_rubbish_dist)) - np.sum(np.exp(-my_rubbish_dist)) - np.sum(np.exp(-my_next_apple_dist)) + np.sum(np.exp(-my_apple_dist))
        if other_action[k] == 5:
            if other_state[j, -2, other_pos[0][0], other_pos[0][1]] == 1: # other pick up the rubbish
                other_value = 2
            elif other_state[j, -1, other_pos[0][0], other_pos[0][1]] == 1: # other pick up the apple
                other_value = -2
        else:
            other_value = np.sum(np.exp(-other_next_rubbish_dist)) - np.sum(np.exp(-other_rubbish_dist)) - np.sum(np.exp(-other_next_apple_dist)) + np.sum(np.exp(-other_apple_dist))
        factor.append(other_value - my_value)

    return torch.tensor(factor).view(-1, 1).to(device).detach() 

def counterfactual_factor(my_id, other_id, agents, state, last_action, my_action, real_q_value, observable):
    counterfactual_result = []
    counterfactual_result.append(real_q_value)
    other_action = copy.deepcopy(last_action[:, env.action_space*other_id:env.action_space*(other_id+1)])
    virtual_action = torch.nonzero(other_action == 0).view(32, env.action_space-1, 2)
    other_action.fill_(0)
    for i in range(env.action_space-1):
        cur_try_action_pos = virtual_action[:, i, :] # 32*2
        other_action[cur_try_action_pos[:, 0], cur_try_action_pos[:, 1]] = 1 # try another action
        last_action[:, env.action_space*other_id:env.action_space*(other_id+1)] = other_action
        cf_input = torch.cat((state, last_action), dim=1)
        cf_q_value = agents[my_id](cf_input).gather(1, my_action)
        counterfactual_result.append(cf_q_value)
        other_action.fill_(0)
    cf_result = torch.transpose(torch.stack(counterfactual_result), 0, 1)
    min_value, _ = cf_result.min(dim=1)
    max_value, _ = cf_result.max(dim=1)
    factor = 2 * (real_q_value - min_value) / (max_value - min_value) - 1
    return factor.view(-1, 1).to(device).detach() * torch.tensor(observable).to(device).view(-1, 1)

def snowdrift_compute_factor(my_state, other_state, my_next_state, other_next_state, my_action, other_action, i, j):
    other_state = other_state.detach().cpu().numpy()
    other_next_state = other_next_state.detach().cpu().numpy()
    my_action = my_action.squeeze().detach().cpu().numpy()
    other_action = other_action.squeeze().detach().cpu().numpy()
    factor = []
    for k in range(32):
        my_value = 0
        other_value = 0
        my_pos = np.argwhere(my_state[k, i, :, :] == 1)
        my_next_pos = np.argwhere(my_next_state[k, i, :, :] == 1)
        other_pos = np.argwhere(other_state[k, j, :, :] == 1)
        other_next_pos = np.argwhere(other_next_state[k, j, :, :] == 1)
        my_drift_pos = np.argwhere(my_state[k, 4, :, :] == 1)
        my_next_drift_pos = np.argwhere(my_next_state[k, 4, :, :] == 1)
        other_drift_pos = np.argwhere(other_state[k, 4, :, :] == 1)
        other_next_drift_pos = np.argwhere(other_next_state[k, 4, :, :] == 1)
        my_dist = np.sum(np.abs(my_pos - my_drift_pos), axis=1)
        my_next_dist = np.sum(np.abs(my_next_pos - my_next_drift_pos), axis=1)
        other_dist = np.sum(np.abs(other_pos - other_drift_pos), axis=1)
        other_next_dist = np.sum(np.abs(other_next_pos - other_next_drift_pos), axis=1)
        if my_action[k] == 5:
            my_value = 2
        else:
            my_value = np.sum(np.exp(-my_next_dist)) - np.sum(np.exp(-my_dist))
        if other_action[k] == 5:
            other_value = 2
        else:
            other_value = np.sum(np.exp(-other_next_dist)) - np.sum(np.exp(-other_dist))
        factor.append(other_value - my_value)

    return torch.tensor(factor).view(-1, 1).to(device).detach()

def get_loss(batch, id, eps):
    obs, action, avail_action, avail_next_action, other_action, reward, next_obs, done, pad = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6], batch[7], batch[8]
    episode_num = obs.shape[0]
    self_loss = 0
    symp_loss = 0
    imagine_loss = 0
    random_start = random.randint(0, env.final_time-51)
    for transition_idx in range(random_start, random_start + 50):
        inputs, inputs_next = [], []
        obs_transition, next_obs_transition, action_transition, avail_action_transition = obs[:, transition_idx, :, :, :, :], next_obs[:, transition_idx, :, :, :, :], action[:, transition_idx, :], avail_action[:, transition_idx]
        avail_next_action_transition = avail_next_action[:, transition_idx]
        reward_transition = torch.tensor(reward[:, transition_idx], dtype=torch.float).to(device)
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

        state = torch.tensor(obs_i, dtype=torch.float).to(device).view(episode_num, -1)
        next_state = torch.tensor(next_obs_i, dtype=torch.float).to(device).view(episode_num, -1)
        last_action = torch.tensor(last_action, dtype=torch.float).to(device).view(episode_num, -1)
        # next_action = torch.tensor(next_action, dtype=torch.float).to(device).view(episode_num, 4*6)
        this_action = torch.tensor(this_action, dtype=torch.float).to(device).view(episode_num, -1)

        inputs = torch.cat((state, last_action), dim=1)
        inputs_next = torch.cat((next_state, this_action), dim=1)

        selfish_inputs = torch.cat((state, last_action), dim=1)
        selfish_next_inputs = torch.cat((next_state, this_action), dim=1)
        self_q_value = selfish_agents[id](selfish_inputs).gather(1, action_i)
        self_q_target = target_selfish_agents[id](selfish_next_inputs)
        self_q_target[avail_next_action_transition == 0] = -99999
        self_q_target = self_q_target.max(1)[0].view(-1, 1)

        symp_q_value, symp_agents[id].eval_hidden = symp_agents[id](inputs, symp_agents[id].eval_hidden)
        symp_q_target, target_symp_agents[id].eval_hidden = target_symp_agents[id](inputs_next, target_symp_agents[id].eval_hidden)
        symp_q_value = symp_q_value.gather(1, action_i)
        symp_q_target[avail_next_action_transition == 0] = -99999
        symp_q_target = symp_q_target.max(1)[0].view(-1, 1)

        factor = [0 for _ in range(4)]
        # imagine_loss_1 = [0 for _ in range(4)]
        # imagine_loss_2 = [0 for _ in range(4)]
        # j_imagine_obs = [0 for _ in range(4)]
        # j_next_imagine_obs = [0 for _ in range(4)]
        j_action = [0 for _ in range(4)]
        j_q = [0 for _ in range(4)]
        # other_q_value = [0 for _ in range(4)]
        # other_q_target = [0 for _ in range(4)]
        other_reward = [0 for _ in range(4)]
        # agents_clone = [Qself(config).to(device) for _ in range(4)]
        # agents_target_clone = [Qself(config).to(device) for _ in range(4)]

        for j in range(4):
            if j == id:
                continue
            else:
                # obs_j = torch.tensor(obs_transition[:, j, :, :, :], dtype=torch.float).to(device)
                # next_obs_j = torch.tensor(next_obs_transition[:, j, :, :, :], dtype=torch.float).to(device)
                # j_imagine_obs[j] = agents_imagine[id](obs_j).view(batch_size, env.channel, window_height, window_width)
                # j_next_imagine_obs[j] = agents_imagine[id](next_obs_j).view(batch_size, env.channel, window_height, window_width)
                # imagine_loss_2[j] = torch.mean(torch.abs(j_imagine_obs[j]-obs_j))
                # j_action[j] = other_action_transition[:, j].to(torch.int64)
                # j_action_onehot = F.one_hot(j_action[j], num_classes=env.action_space).squeeze()
                # agents_clone[j].load_state_dict(selfish_agents[i].state_dict())
                # j_imagine_obs[j] = j_imagine_obs[j].view(episode_num, -1)
                # j_inputs = torch.cat((j_imagine_obs[j], last_action), dim=1)
                # j_q[j] = agents_clone[j](j_inputs)
                # j_pi = F.softmax(j_q[j], dim=1)
                # imagine_loss_1[j] = F.cross_entropy(j_pi, j_action_onehot.to(torch.float))
                # other_q_value[j] = j_q[j].gather(1, (j_action[j]).view(-1, 1)).detach()
                
                # agents_target_clone[j].load_state_dict(target_selfish_agents[i].state_dict())
                # j_next_imagine_obs[j] = j_next_imagine_obs[j].view(episode_num, -1)
                # next_j_inputs = torch.cat((j_next_imagine_obs[j], this_action), dim=1)
                # next_j_q_value = agents_target_clone[j](next_j_inputs)
                # other_q_target[j] = next_j_q_value.max(1)[0].view(-1, 1).detach()
                other_reward[j] = reward_transition[:, j].view(-1, 1)

                if transition_idx == 0:
                    factor[j] = torch.zeros((32, 1)).to(device).detach()
                # factor[j] = snowdrift_compute_factor(obs_i, obs_j, next_obs_i, next_obs_j, action_i, j_action[j], id, j)
                # factor[j] = cleanup_compute_factor(obs_i, obs_j, next_obs_i, next_obs_j, action_i, j_action[j], id, j)
                else:
                    factor[j] = counterfactual_factor(id, j, selfish_agents, state, last_action, action_i, self_q_value, done_player[:, j])

        my_reward = reward_transition[:, id].view(-1, 1)
        self_target = my_reward + gamma * self_q_target * (1 - done_transition)
        self_td_error = (self_q_value - self_target)
        mask_self_td_error = (1 - pad_transition) * self_td_error
        self_loss += mask_self_td_error ** 2

        target_reward = 0
        for k in range(4):
            if isinstance(factor[k], torch.Tensor):
                target_reward += (1 - factor[k]) * reward_transition + factor[k] * other_reward[k]
        if transition_idx == 20 and eps % 100 == 0 and id == 0:
            print("real obs:", obs_transition[0, 1, :, :, :])
            # print("imagine obs:", j_imagine_obs[1][0])
        if transition_idx == 20 and eps % 10 == 0 and id == 0:
            print("eps: ", eps, "===========================")
            print("factor: ",factor)
            print("reward: ", reward_transition)
            print("other reward: ", other_reward)
            print("action: ", action_transition)
        symp_target = target_reward + gamma * symp_q_target * (1 - done_transition)
        symp_td_error = symp_q_value - symp_target
        mask_symp_td_error = (1 - pad_transition) * symp_td_error
        symp_loss += mask_symp_td_error ** 2
        # imagine_loss += (1 - delta) * sum(imagine_loss_1) + delta * sum(imagine_loss_2)

    return self_loss, symp_loss, imagine_loss

wandb.init(project='Empathy', entity='kfq20', name='modify cleanup known reward', notes='modify rnn')
# total_time = 0
# loss_time = 0
# forward_time = 0
for ep in range(max_episode):
    # ep_start = time.time()
    o, a, other_a, r, o_next, avail_a, avail_a_next, pad, d = [[] for _ in range(4)], [[] for _ in range(4)], [[] for _ in range(4)], [[] for _ in range(4)], [[] for _ in range(4)], [[] for _ in range(4)], [[] for _ in range(4)], [[] for _ in range(4)], [[] for _ in range(4)]
    obs = env.reset()
    all_done = False
    total_reward = np.zeros(4)
    total_hunt_stag_num = 0
    total_hunt_hare_num = 0
    total_collect_waste_num = 0
    total_collect_apple_num = 0
    total_punish_num = 0
    last_action = np.zeros((4, env.action_space))
    step = 0
    for i in range(4):
        symp_agents[i].init_hidden(1)
    # print('total_time', total_time)
    while not all_done:
        actions = []
        avail_actions = []
        avail_next_actions = []
        probs = []
        last_action = torch.flatten(torch.tensor(last_action, dtype=torch.float)).to(device)
        # obs = torch.tensor(obs, dtype=torch.float).to(device)
        # forward_start = time.time()
        for i in range(4):
            avail_action = env.__actionmask__(i)
            avail_action_index = np.nonzero(avail_action)[0]
            
            state = torch.flatten(torch.tensor(obs[i], dtype=torch.float)).to(device)
            input = torch.cat((state, last_action))
            hidden_state = symp_agents[i].eval_hidden.to(device)
            
            q_value, symp_agents[i].eval_hidden = symp_agents[i](input.unsqueeze(0), hidden_state)
            
            
            q_value[:, avail_action == 0] = -float("inf")
            if np.random.random() < epsilon:
                action = np.random.choice(avail_action_index)
            else:
                action = q_value.argmax().item()
            actions.append(action)
            avail_actions.append(avail_action)

        # forward_end = time.time()
        # forward_time += forward_end - forward_start
        next_obs, reward, done, info = env.step(actions)
        collect_waste_num = info[0]
        collect_apple_num = info[1]
        punish_num = info[2]
        total_collect_waste_num += collect_waste_num
        total_collect_apple_num += collect_apple_num
        total_punish_num += punish_num

        # hunt_hare_num = info[0]
        # hunt_stag_num = info[1]
        # total_hunt_stag_num += hunt_stag_num
        # total_hunt_hare_num += hunt_hare_num
        for i in range(4):
            avail_next_action = env.__actionmask__(i)
            avail_next_actions.append(avail_next_action)
        # obs = next_obs
        total_reward += reward
        if len(obs.shape) != 4:
            print(1)
        for i in range(4):
            other_action = np.ones(4) * -1
            for j in range(4):
                if j == i:
                    continue
                else:
                    other_action[j] = actions[j]
            o[i].append(obs)
            a[i].append(actions)
            r[i].append(reward)
            o_next[i].append(next_obs)
            avail_a[i].append(avail_actions[i])
            avail_a_next[i].append(avail_next_actions[i])
            other_a[i].append(other_action)
            d[i].append(done[i])
            pad[i].append(0.)
            
            # buffer[i].add(obs, actions[i], last_action, avail_actions[i], avail_next_actions[i], other_action, reward[i], next_obs, done[i])
        actions = np.array(actions)
        one_hot = np.zeros((len(actions), env.action_space))
        one_hot[np.arange(len(actions)), actions] = 1
        last_action = one_hot
        obs = next_obs     
        all_done = done[-1]
        step += 1

    for i in range(step, env.final_time):
        for j in range(4):
            o[j].append(np.zeros((env.player_num, env.channel, window_height, window_width)))
            a[j].append([0,0,0,0])
            r[j].append([0,0,0,0])
            o_next[j].append(np.zeros((env.player_num, env.channel, window_height, window_width)))
            avail_a[j].append(np.zeros(env.action_space))
            avail_a_next[j].append(np.zeros(env.action_space))
            other_a[j].append(np.zeros(4))
            d[j].append(1)
            pad[j].append(1)

    for i in range(4):
        buffer[i].add(o[i], a[i], avail_a[i], avail_a_next[i], other_a[i], r[i], o_next[i], d[i], pad[i])

    if buffer[0].size() > minimal_size:
        cnt += 1
        # print(total_reward)
        total_self_loss = 0
        total_symp_loss = 0
        total_imagine_loss = 0
        wandb.log({'reward_1':total_reward[0], 'reward_2':total_reward[1], 'reward_3':total_reward[2], 'reward_4':total_reward[3], 'total_reward':sum(total_reward), 'waste_num':total_collect_waste_num, 'apple_num':total_collect_apple_num, 'punish num':total_punish_num})
        for i in range(4):
            obs, action, avail_action, avail_next_action, other_action, reward, next_obs, done, pad = buffer[i].sample(min(buffer[i].size(), batch_size))
            episode_num = obs.shape[0]
            symp_agents[i].init_hidden(episode_num)
            target_symp_agents[i].init_hidden(episode_num)
            symp_agents[i].eval_hidden = symp_agents[i].eval_hidden.to(device)
            target_symp_agents[i].eval_hidden = target_symp_agents[i].eval_hidden.to(device)
            batch = [obs, action, avail_action, avail_next_action, other_action, reward, next_obs, done, pad]
            # loss_time_start = time.time()
            self_loss, symp_loss, imagine_loss = get_loss(batch, i, ep)
            
            self_loss = torch.mean(self_loss)
            symp_loss = torch.mean(symp_loss)
            optim_selfish[i].zero_grad()
            self_loss.backward()
            optim_selfish[i].step()
            total_self_loss += self_loss.item()

            optim_symp[i].zero_grad()
            symp_loss.backward(retain_graph=True)
            optim_symp[i].step()
            total_symp_loss += symp_loss.item()

            # optim_imagine[i].zero_grad()
            # imagine_loss.backward()
            # optim_imagine[i].step()
            # total_imagine_loss += imagine_loss
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
                target_symp_agents[i].load_state_dict(symp_agents[i].state_dict())
        wandb.log({'self_loss': total_self_loss, 'symp_loss': total_symp_loss, 'imagine_loss':imagine_loss})

    epsilon = epsilon - anneal_epsilon if epsilon > min_epsilon else epsilon

    # ep_end = time.time()
    # total_time += ep_end - ep_start
    # print(total_time, loss_time, forward_time)