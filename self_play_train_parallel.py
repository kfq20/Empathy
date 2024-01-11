import torch
from env import *
from model_sd import *
from replay import *
import torch.nn.functional as F
import wandb
from torch.distributions import Categorical

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

is_wandb = True
seed = 798
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

gamma = 0.98
max_episode = 100000
anneal_eps = 2000
batch_size = 1000
update_freq = 1
self_update_freq = 30
minimal_size = 1000
total_time = 0
epsilon = 0.5
min_epsilon = 0.05
anneal_epsilon = (epsilon - min_epsilon) / anneal_eps
delta = 0.1
# gifting_mode = ['prosocial', 'prosocial', 'prosocial', 'prosocial']

window_height = 5
window_width = 5
env_parallel_num = 20

forward_step_time = 0
forward_action_time = 0
backward_time = 0
total_time = 0
test_time = 0

env_parallel = [CoinGame() for _ in range(env_parallel_num)]
env = CoinGame()
gifting_mode = ['empathy' for _ in range(env.player_num)]
config = {}
config["channel"] = env.channel
config["height"] = env.obs_height
# config["width"] = env.width - 3
config["width"] = env.obs_width
# config["width"] = 15 # cleanup 16*15
config["player_num"] = env.player_num
config["action_space"] = env.action_space

symp_agents = [ActorCritic(config).to(device) for _ in range(env.player_num)]
# target_symp_agents = [ActorCritic(config).to(device) for _ in range(env.player_num)]
selfish_agents = [ACself(config).to(device) for _ in range(env.player_num)]
# target_selfish_agents = [Qself(config).to(device) for _ in range(env.player_num)]
# for i in range(env.player_num):
#     target_selfish_agents[i].load_state_dict(selfish_agents[i].state_dict())

agents_imagine = [Imagine(config).to(device) for _ in range(env.player_num)]

optim_symp = [torch.optim.Adam(symp_agents[i].parameters(), lr=1e-4) for i in range(env.player_num)]
optim_imagine = [torch.optim.Adam(agents_imagine[i].parameters(), lr=3e-5) for i in range(env.player_num)]
optim_selfish = [torch.optim.Adam(selfish_agents[i].parameters(), lr=5e-5) for i in range(env.player_num)]
buffer = [ReplayBuffer(capacity=5000, id=i) for i in range(env.player_num)]

def other_obs_process(obs, id):
    batch_size = obs.shape[0]
    total_other_obs = []
    total_condition = []
    observable = np.zeros((batch_size, env.player_num))

    if env.name == 'mp_cleanup':
        for i in range(env.player_num):
            if i == id:
                total_condition.append(np.zeros(batch_size))
            else:
                total_condition.append(np.ones(batch_size))

    else:
        for j in range(env.player_num):
            if j == id:
                continue
            else:
                j_observable = np.any(obs[:, id, j, :, :] == 1, axis=(1, 2))
                observable[:, j] = j_observable
        for i in range(env.player_num):
            if i == id:
                total_other_obs.append(np.zeros((batch_size, env.channel, env.obs_height, env.obs_width)))
                total_condition.append(np.zeros(batch_size))
                continue
            else:
                condition = observable[:, i] == 1
                condition = condition.astype(float)
                total_condition.append(condition)
            
    return obs, np.array(total_condition)

def counterfactual_factor(my_id, other_id, selfish_agents, imagine_nn, state, this_action, real_q_value):
    batch_size = real_q_value.shape[0]
    other_id_onehot = F.one_hot(torch.tensor(other_id), num_classes=env.player_num)
    other_id_batch = torch.stack([other_id_onehot]*batch_size).to(device)
    counterfactual_result = []
    counterfactual_result.append(real_q_value)
    other_action = copy.deepcopy(this_action[:, env.action_space*other_id:env.action_space*(other_id+1)])
    virtual_action = torch.nonzero(other_action == 0).view(batch_size, env.action_space-1, 2)
    other_action.fill_(0)
    imagine_state = imagine_nn(state, other_id_batch).view(batch_size, env.channel, env.obs_height, env.obs_width)
    this_action_logits, _ = selfish_agents[my_id](imagine_state, this_action)
    real_action = this_action[:, env.action_space*other_id:env.action_space*(other_id+1)]
    this_action_prob = F.softmax(this_action_logits, dim=1)
    # this_action_prob = F.softmax(torch.rand(env.final_time*env_parallel_num, env.action_space), dim=1).to(device)
    cf_baseline = torch.sum(this_action_prob*real_action, dim=1, keepdim=True) * real_q_value
    for i in range(env.action_space-1):
        cur_try_action_pos = virtual_action[:, i, :] # 32*2
        other_action[cur_try_action_pos[:, 0], cur_try_action_pos[:, 1]] = 1 # try another action
        this_action[:, env.action_space*other_id:env.action_space*(other_id+1)] = other_action

        cf_action_prob = torch.sum(this_action_prob*other_action, dim=1, keepdim=True)
        # cf_input = torch.cat((state, last_action), dim=1)
        _, cf_q_value = selfish_agents[my_id](state, this_action)
        
        cf_baseline += cf_action_prob * cf_q_value
        counterfactual_result.append(cf_q_value)
        # log.append(cf_q_value.item())
        other_action.fill_(0)
    # if_log = random.random()
    # if if_log < 0.001:
    #     print("other action", virtual_action)
    #     print(log)
    cf_result = torch.transpose(torch.stack(counterfactual_result), 0, 1)
    min_value, _ = cf_result.min(dim=1)
    max_value, _ = cf_result.max(dim=1)
    factor = (real_q_value - cf_baseline) / (max_value - min_value)
    # factor = 2 * (real_q_value - min_value) / (max_value - min_value) - 1
    return factor.detach().cpu()

def get_loss(batch, id, eps):
    obs, action, other_action, reward, next_obs, done = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]
    # batch_size = min(obs.shape[0], batch_size)
    episode_num = obs.shape[0]
    self_actor_loss = 0
    self_critic_loss = 0
    symp_loss = 0
    imagine_loss = 0
    random_log = random.randint(0, env.final_time-1)
    # random_start = random.randint(0, env.final_time-51)
    for transition_idx in range(env.final_time):
        # st1 = time.time()
        obs_transition, next_obs_transition, action_transition = obs[:, transition_idx, :, :, :, :], next_obs[:, transition_idx, :, :, :, :], action[:, transition_idx, :]
        # avail_next_action_transition = avail_next_action[:, transition_idx]
        _, observable = other_obs_process(obs_transition, id)
        _, _ = other_obs_process(next_obs_transition, id)
        # et1 = time.time()
        # other_obs_transition = torch.tensor(other_obs_transition, dtype=torch.float).to(device)
        # other_next_obs_transition = torch.tensor(other_next_obs_transition, dtype=torch.float).to(device)
        observable = torch.tensor(observable, dtype=torch.float).to(device)
        reward_transition = torch.tensor(reward[:, transition_idx], dtype=torch.float).view(-1, 1).to(device)
        done_transition = torch.tensor(done[:, transition_idx], dtype=torch.float).view(-1, 1).to(device)
        other_action_transition = torch.tensor(other_action[:, transition_idx, :], dtype=torch.float).to(device)
        obs_i = obs_transition[:, id, :, :, :]
        done_player = np.zeros((batch_size, env.player_num))
        for j in range(env.player_num):
            if j == id:
                continue
            else:
                done_player[:, j] = np.sum(obs_i[:, j, :, :], axis=(1, 2))
        next_obs_i = next_obs_transition[:, id, :, :, :]
        action_i = torch.tensor(action_transition[:, id]).view(-1, 1).to(device)
        # inputs_next.append(next_obs_transition)
        if transition_idx == 0:
            last_action = np.zeros((episode_num, env.player_num, env.action_space))
        else:
            last_action = action[:, transition_idx - 1]
            last_action = np.eye(env.action_space)[last_action] # one-hot
        
        if transition_idx == env.final_time - 1:
            next_action = np.zeros((episode_num, env.player_num, env.action_space))
        else:
            next_action = action[:, transition_idx + 1]
            next_action = np.eye(env.action_space)[next_action]

        this_action = action[:, transition_idx]
        this_action = np.eye(env.action_space)[this_action]

        state = torch.tensor(obs_i, dtype=torch.float).to(device)
        next_state = torch.tensor(next_obs_i, dtype=torch.float).to(device)
        last_action = torch.tensor(last_action, dtype=torch.float).to(device).view(episode_num, -1)
        # next_action = torch.tensor(next_action, dtype=torch.float).to(device).view(episode_num, env.player_num*6)
        this_action = torch.tensor(this_action, dtype=torch.float).to(device).view(episode_num, -1)
        next_action = torch.tensor(next_action, dtype=torch.float).to(device).view(episode_num, -1)
        
        # selfish_inputs = torch.cat((state, last_action), dim=1)
        # selfish_next_inputs = torch.cat((next_state, this_action), dim=1)
        logits, value = selfish_agents[id](state, this_action)
        _, next_value = selfish_agents[id](next_state, next_action)
        log_probs = torch.log(torch.softmax(logits, dim=1).gather(1, action_i)+1e-10)
        # log_probs = torch.log(torch.softmax(logits, dim=1).gather(1, action_i))
        if torch.isnan(log_probs).int().sum() > 0:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!warning!!!!!!!!!!!!!!!!!")
            print("log is wrong!!!!")
        td_target = reward_transition + 0.99 * next_value * (1 - done_transition
                                                                       )
        td_delta = td_target - value
        self_actor_loss += torch.mean(-log_probs * td_delta.detach())
        self_critic_loss += torch.mean(
            F.mse_loss(value, td_target.detach()))
        
        # time1 += et1-st1
        # factor = [0 for _ in range(env.player_num)]
        imagine_loss_1 = [0 for _ in range(env.player_num)]
        imagine_loss_2 = [0 for _ in range(env.player_num)]
        j_imagine_obs = [0 for _ in range(env.player_num)]
        j_next_imagine_obs = [0 for _ in range(env.player_num)]
        j_action = [0 for _ in range(env.player_num)]
        j_q = [0 for _ in range(env.player_num)]
        agents_clone = [ACself(config).to(device) for _ in range(env.player_num)]
        # st2 = time.time()
        for j in range(env.player_num):
            if j == id:
                continue
            else:
                j_id = F.one_hot(torch.tensor(j), num_classes=env.player_num)
                j_id_batch = torch.stack([j_id]*state.shape[0]).to(device)
                j_imagine_obs[j] = agents_imagine[id](state, j_id_batch).view(batch_size, env.channel, window_height, window_width)
                j_next_imagine_obs[j] = agents_imagine[id](next_state, j_id_batch).view(batch_size, env.channel, window_height, window_width)
                imagine_loss_2[j] = torch.mean(torch.sum(torch.abs((j_imagine_obs[j]-state)), dim=(1,2,3), keepdim=True)*observable[j].unsqueeze(0))
                j_action[j] = (other_action_transition[:, j]*observable[j]).to(torch.int64)
                j_action_onehot = F.one_hot(j_action[j], num_classes=env.action_space).squeeze()
                agents_clone[j].load_state_dict(selfish_agents[i].state_dict())
                j_q[j], _ = agents_clone[j](j_imagine_obs[j], this_action)
                j_pi = F.softmax(j_q[j], dim=1)
                if j_pi[observable[j] == 1].shape[0] > 0:
                    imagine_loss_1[j] = F.cross_entropy(j_pi[observable[j] == 1], j_action_onehot.to(torch.float)[observable[j] == 1])
                if transition_idx == random_log and eps % 200 == 0:
                    print(f"agent_{id} imagine agent_{j}")
                    print("imagine obs",j_imagine_obs[j])
                    print("real obs", state)
        # et2 = time.time()
        # time2 += et2-st2
        # print('t1',time1)
        # print('t2', time2)
        imagine_loss += (1 - delta) * sum(imagine_loss_1) + delta * sum(imagine_loss_2)

    return self_actor_loss, self_critic_loss, symp_loss, imagine_loss

if is_wandb:
    wandb.init(project='Empathy', entity='kfq20', name='final_empathy_coingame_seed_3')
step_time = 0
a2c_time = 0
offline_time = 0
test_time = 0

def evaluate():
    obs = env.reset()
    all_done = False
    total_reward = np.zeros(env.player_num)
    total_coin_1to1 = 0
    total_coin_1to2 = 0
    total_coin_2to1 = 0
    total_coin_2to2 = 0
    total_hunt_stag_num = 0
    total_hunt_hare_num = 0
    total_sd_num = 0
    total_collect_apple_num = 0
    total_collect_waste_num = np.zeros(env.player_num)
    last_action = np.zeros((env.player_num, env.action_space))
    log_probs = [[] for i in range(env.player_num)]
    values = [[] for i in range(env.player_num)]
    for i in range(env.player_num):
        symp_agents[i].init_hidden(1)
    # print('total_time', total_time)
    while not all_done:
        actions = []
        avail_actions = []
        last_action = torch.flatten(torch.tensor(last_action, dtype=torch.float)).to(device)
        for i in range(env.player_num):
            if i == 0 or i == 1 or i == 2 or i == 3:
                avail_action = env.__actionmask__(i)

                state = torch.tensor(obs[i], dtype=torch.float).to(device)
                # input = torch.cat((state, last_action))
                h_in = symp_agents[i].hx.to(device)
                c_in = symp_agents[i].cx.to(device)
                action_prob, state_value, symp_agents[i].hx, symp_agents[i].cx = symp_agents[i](state.unsqueeze(0), h_in, c_in)
                # print(action_prob)
                # action_prob[:, avail_action == 0] = -999999
                action_prob = F.softmax(action_prob, dim=-1)

                dist = Categorical(action_prob[0])
                action = torch.argmax(action_prob)
                one_log_prob = dist.log_prob(action)
                actions.append(action.item())
                avail_actions.append(avail_action)
                log_probs[i].append(one_log_prob.unsqueeze(0))
                values[i].append(state_value)

        next_obs, reward, done, info = env.step(actions)
        if env.name == 'coingame':
            total_coin_1to1 += info[0][0]
            total_coin_1to2 += info[0][1]
            total_coin_2to1 += info[1][0]
            total_coin_2to2 += info[1][1]
        elif env.name == 'snowdrift':
            total_sd_num += info
        elif env.name == 'staghunt':
            hare_num = info[0]
            stag_num = info[1]
            total_hunt_hare_num += hare_num
            total_hunt_stag_num += stag_num
        elif env.name == 'cleanup':
            total_collect_waste_num += info[0]
            total_collect_apple_num += info[1]

        # obs = next_obs
        total_reward += reward

        actions = np.array(actions)
        one_hot = np.zeros((len(actions), env.action_space))
        one_hot[np.arange(len(actions)), actions] = 1
        last_action = one_hot
        obs = next_obs     
        all_done = done[-1]
    
    if is_wandb:
        if env.name == 'staghunt':
            wandb.log({'eval reward1':total_reward[0], 
                    'eval reward2':total_reward[1],
                    'eval reward3':total_reward[2],
                    'eval reward4':total_reward[3], 
                    'eval total r':np.sum(total_reward), 
                    'eval hare num':total_hunt_hare_num,
                    'eval stag num':total_hunt_stag_num,
                    })
        elif env.name == 'cleanup':
            wandb.log({'eval reward1':total_reward[0], 
                    'eval reward2':total_reward[1],
                    'eval reward3':total_reward[2],
                    'eval reward4':total_reward[3], 
                    'eval total r':np.sum(total_reward), 
                    'eval waste num 1':total_collect_waste_num[0],
                    'eval waste num 2':total_collect_waste_num[1],
                    'eval waste num 3':total_collect_waste_num[2],
                    'eval waste num 4':total_collect_waste_num[3],
                    'eval apple num':total_collect_apple_num
                    })
        elif env.name == 'coingame':
            wandb.log({'eval reward1':total_reward[0], 
                    'eval reward2':total_reward[1],
                    'eval total r':sum(total_reward), 
                    'eval 1to1 coin':total_coin_1to1,
                    'eval 1to2 coin':total_coin_1to2,
                    'eval 2to1 coin':total_coin_2to1,
                    'eval 2to2 coin':total_coin_2to2,
                    })
        elif env.name == 'snowdrift':
            wandb.log({'eval reward1':total_reward[0], 
                    'eval reward2':total_reward[1],
                    'eval reward3':total_reward[2],
                    'eval reward4':total_reward[3], 
                    'eval total r':sum(total_reward), 
                    'eval sd num':total_sd_num,
                    })
        
update_info = [{
    'log_probs':[],
    'advantage':[]
} for _ in range(env.player_num)]

for ep in range(max_episode):
    # total_s = time.time()
    if (ep+1) % 50 == 0:
        evaluate()
    o, a, other_a, r, o_next, d = [[] for _ in range(env.player_num)], [[] for _ in range(env.player_num)], [[] for _ in range(env.player_num)], [[] for _ in range(env.player_num)], [[] for _ in range(env.player_num)], [[] for _ in range(env.player_num)]
    obs_par = []
    for i in range(env_parallel_num):
        obs = env_parallel[i].reset()
        obs_par.append(obs)
    obs_par = np.array(obs_par)
    all_done = False
    total_reward = np.zeros(env.player_num)
    total_coin_1to1 = 0
    total_coin_1to2 = 0
    total_coin_2to1 = 0
    total_coin_2to2 = 0
    total_hunt_stag_num = 0
    total_hunt_hare_num = 0
    total_collect_waste_num = np.zeros(env.player_num)
    total_collect_apple_num = 0
    total_punish_num = 0
    total_sd_num = 0
    last_action = np.zeros((env.player_num, env.action_space))
    step = 0
    log_probs = [[] for i in range(env.player_num)]
    values = [[] for i in range(env.player_num)]
    rewards = []
    env_done = [False for _ in range(env_parallel_num)]
    env_step_len = [0 for _ in range(env_parallel_num)]
    for i in range(env.player_num):
        symp_agents[i].init_hidden(env_parallel_num)
    # print('total_time', total_time)
    # step_s = time.time()
    while step < env.final_time:
        
        actions = []
        avail_actions = []
        avail_next_actions = []
        probs = []
        all_log_prob = [[] for i in range(env.player_num)]
        all_value = []
        # last_action = torch.flatten(torch.tensor(last_action, dtype=torch.float)).to(device)
        # obs = torch.tensor(obs, dtype=torch.float).to(device)
        # forward_start = time.time()
        for i in range(env.player_num):
            # test_s = time.time()
            # if i == 0 or i == 1 or i == 2 or i == 3:
            state = torch.tensor(obs_par[:, i], dtype=torch.float).to(device)
            # input = torch.cat((state, last_action))
            h_in = symp_agents[i].hx.to(device)
            c_in = symp_agents[i].cx.to(device)
            # action_s = time.time()
            action_prob, state_value, symp_agents[i].hx, symp_agents[i].cx = symp_agents[i](state, h_in, c_in)
            # action_e = time.time()
            
            # forward_action_time += action_e - action_s
            # print(action_prob)
            # action_prob[:, avail_action == 0] = -999999
            action_prob = F.softmax(action_prob, dim=-1)
            p = torch.sum(action_prob)
            dist = Categorical(action_prob)
            
            if np.random.random() < epsilon:
                action = torch.tensor(np.array([np.random.choice(env.action_space) for _ in range(env_parallel_num)]), dtype=torch.int8).to(device)
            else:
                action = dist.sample()
            # test_e = time.time()
            # test_time += test_e - test_s
            actions.append(action.cpu().detach().numpy())
            one_log_prob = dist.log_prob(action)
            log_probs[i].append(one_log_prob)
            values[i].append(state_value)
            
        actions = np.array(actions)
        next_obs_par, reward_par, done_par = [], [], []
        for env_index in range(env_parallel_num):
            if not env_done[env_index]:
                # step_s = time.time()
                next_obs, reward, done, info = env_parallel[env_index].step(actions[:, env_index])
                # step_e = time.time()
                # forward_step_time += step_e - step_s
                next_obs_par.append(next_obs)
                reward_par.append(reward)
                done_par.append(done)
                if env.name == 'coingame':
                    total_coin_1to1 += info[0][0]
                    total_coin_1to2 += info[0][1]
                    total_coin_2to1 += info[1][0]
                    total_coin_2to2 += info[1][1]
                elif env.name == 'snowdrift':
                    total_sd_num += info
                elif env.name == 'staghunt':
                    hare_num = info[0]
                    stag_num = info[1]
                    total_hunt_hare_num += hare_num
                    total_hunt_stag_num += stag_num
                elif env.name == 'cleanup':
                    total_collect_waste_num += info[0]
                    total_collect_apple_num += info[1]
                total_reward += reward
                env_step_len[env_index] += 1
                if done[-1] == True:
                    env_done[env_index] = True
            else:
                next_obs_par.append(np.zeros((env.player_num, env.channel, window_height, window_width)))
                reward_par.append(np.zeros(env.player_num))
                done_par.append(np.array([True for _ in range(env.player_num)]))
        
        next_obs_par, reward_par, done_par = np.array(next_obs_par), np.array(reward_par), np.array(done_par)
        # for env_i in range(env_parallel_num):
        for i in range(env.player_num):
            other_action = np.ones((env_parallel_num, env.player_num)) * -1
            for j in range(env.player_num):
                if j == i:
                    continue
                else:
                    other_action[:, j] = actions[j, :]
            o[i].append(obs_par)
            a[i].append(actions)
            r[i].append(reward_par[:, i])
            o_next[i].append(next_obs_par)
            other_a[i].append(other_action)
            d[i].append(done_par[:, i])

        obs_par = next_obs_par
        all_done = done[-1]
        step += 1
        # print(step)

    this_eps_len = step
    print("eps:", ep, "len:", this_eps_len)

    for i in range(env.player_num):
        o[i] = np.array(o[i])
        a[i] = np.array(a[i])
        other_a[i] = np.array(other_a[i])
        r[i] = np.array(r[i])
        o_next[i] = np.array(o_next[i])
        d[i] = np.array(d[i])
        for env_i in range(env_parallel_num):
            buffer[i].add(o[i][:, env_i], a[i][:, :, env_i], other_a[i][:, env_i], r[i][:, env_i], o_next[i][:, env_i], d[i][:, env_i])

    total_actor_loss = 0
    total_critic_loss = 0
    total_give_factor = []
    
    for i in range(env.player_num):
        factor = [0 for _ in range(env.player_num)]
        last_factor = [0 for _ in range(env.player_num)]
        weighted_rewards = [0 for _ in range(env.final_time)]
        give_factor = [0 for _ in range(env.player_num)]
        if gifting_mode[i] == 'random':
            give_factor = torch.rand(this_eps_len, env.player_num)
            give_factor = give_factor / give_factor.sum(dim=1, keepdim=True)
            give_factor = torch.transpose(give_factor, 0, 1)
        with torch.no_grad():
            # back_s = time.time()
            y = a[i][:this_eps_len, :, :]
            y = np.transpose(y, (0, 2, 1))
            one_hot_array_y = np.eye(env.action_space)[y]
            this_action = one_hot_array_y

            obs_batch = o[i][:this_eps_len]

            i_obs_batch = obs_batch[:, :, i, :,:,:]
            state = torch.tensor(i_obs_batch, dtype=torch.float).to(device).view(i_obs_batch.shape[0]*i_obs_batch.shape[1], i_obs_batch.shape[2], i_obs_batch.shape[3], i_obs_batch.shape[4])
            this_action = torch.tensor(this_action, dtype=torch.float).to(device).view(this_action.shape[0]*this_action.shape[1], -1)

            _, self_q_value = selfish_agents[i](state, this_action)
            weighted_rewards = 0
            if gifting_mode[i] == 'selfish':
                give_factor = []
                for j in range(env.player_num):
                    if j != i:
                        give_factor.append(torch.zeros(1, this_eps_len*env_parallel_num))
                    else:
                        give_factor.append(torch.ones(1, this_eps_len*env_parallel_num))
                give_factor = torch.cat(give_factor, dim=0)
                total_give_factor.append(give_factor)
            else:
                for j in range(env.player_num):
                    if j == i:
                        if gifting_mode[i] == 'prosocial':
                            prosocial_factor = torch.tensor(1/env.player_num, dtype=torch.float).expand(this_eps_len*env_parallel_num, 1)
                            give_factor[j] = prosocial_factor
                        continue
                    else:
                        one_factor = torch.ones((this_eps_len*env_parallel_num, 1))
                        if gifting_mode[i] == 'empathy':
                            origin_factor = counterfactual_factor(i, j, selfish_agents, agents_imagine[i], state, this_action, self_q_value) # len(eps)-1
                            give_factor[j] = torch.clamp(origin_factor, min=0) / (env.player_num - 1) # cf factor
                            if isinstance(give_factor[i], torch.Tensor):
                                give_factor[i] += torch.clamp((one_factor - origin_factor), max=1) / (env.player_num - 1)
                            else:
                                give_factor[i] = torch.clamp((one_factor - origin_factor), max=1) / (env.player_num - 1)
                        elif gifting_mode[i] == 'prosocial':
                            prosocial_factor = torch.tensor(1/env.player_num, dtype=torch.float).expand(this_eps_len*env_parallel_num, 1)
                            give_factor[j] = prosocial_factor
                total_give_factor.append(give_factor)
    
    for i in range(env.player_num):
        i_log_probs = torch.cat(log_probs[i], dim=0)
        i_values = torch.cat(values[i], dim=0).squeeze()
        weighted_rewards = 0
        for j in range(env.player_num):
            j_reward = torch.tensor(r[j][:this_eps_len], dtype=torch.float).view(this_eps_len*env_parallel_num, -1)
            weighted_rewards += total_give_factor[j][i].squeeze() * j_reward.squeeze()
        # compute returns
        state = torch.tensor(obs_par[:, i], dtype=torch.float).to(device)
        h_in = symp_agents[i].hx.to(device)
        c_in = symp_agents[i].cx.to(device)
        _, final_value, _, _ = symp_agents[i](state, h_in, c_in)
        R = final_value.squeeze()
        returns = []
        weighted_rewards = weighted_rewards.view(this_eps_len, env_parallel_num).to(device)
        d[i] = torch.tensor(d[i], dtype=torch.float).to(device)
        for step in reversed(range(this_eps_len)):
            # a = 1-d[i][step, :]
            # b = R
            # c = weighted_rewards[step, :]
            R = weighted_rewards[step, :] + gamma * R * (1-d[i][step, :])
            returns.insert(0, R)

        returns = torch.cat(returns).squeeze().detach()
        if this_eps_len > 1:
            advantage = returns - i_values
        else:
            advantage = (returns - i_values).unsqueeze(0)
        update_info[i]['log_probs'].append(i_log_probs)
        update_info[i]['advantage'].append(advantage)
    
    if (ep+1) % update_freq == 0:
        for i in range(env.player_num):
            log_probs = torch.cat(update_info[i]['log_probs'])
            advantages = torch.cat(update_info[i]['advantage'])
            # test = -log_probs.squeeze() * advantages.detach()
            actor_loss = torch.mean(-log_probs * advantages.detach())
            critic_loss = advantages.pow(2).mean()
            total_actor_loss += actor_loss
            total_critic_loss += critic_loss
            loss = actor_loss + 0.5 * critic_loss
            optim_symp[i].zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(symp_agents[i].parameters(), max_norm=1.0)
            optim_symp[i].step()
            update_info[i]['log_probs'] = []
            update_info[i]['advantage'] = []
    
    total_self_loss = 0
    total_imagine_loss = 0
    total_symp_loss = 0
    # total_e = time.time()
    # total_time += total_e - total_s
    # print('total time', total_time)
    # print('step time', forward_step_time)
    # print('action time', forward_action_time)
    # print('backward time', backward_time)
    # print('test time', test_time)
    if buffer[0].size() > minimal_size and ep % self_update_freq == 0:
        for i in range(env.player_num):
            if gifting_mode[i] == 'empathy':
                obs, action, other_action, reward, next_obs, done = buffer[i].sample(min(buffer[i].size(), batch_size))
                episode_num = obs.shape[0]
                batch = [obs, action, other_action, reward, next_obs, done]
                self_actor_loss, self_critic_loss, symp_loss, imagine_loss = get_loss(batch, i, ep)
                self_loss = self_actor_loss + 0.5*self_critic_loss
                optim_selfish[i].zero_grad()
                self_loss.backward()
                torch.nn.utils.clip_grad_norm_(selfish_agents[i].parameters(), max_norm=1.0)
                optim_selfish[i].step()
                total_self_loss += self_loss

                optim_imagine[i].zero_grad()
                imagine_loss.backward()
                optim_imagine[i].step()
                total_imagine_loss += imagine_loss

    total_reward /= env_parallel_num
    total_coin_1to1 /= env_parallel_num
    total_coin_1to2 /= env_parallel_num
    total_coin_2to1 /= env_parallel_num
    total_coin_2to2 /= env_parallel_num
    total_collect_waste_num /= env_parallel_num
    total_collect_apple_num /= env_parallel_num
    if is_wandb:
        if env.name == 'cleanup':
            wandb.log({
                   'reward_1':total_reward[0],
                   'reward_2':total_reward[1],
                   'reward_3':total_reward[2],
                   'reward_4':total_reward[3],
                   'total_reward':sum(total_reward),
                   'waste_num_1':total_collect_waste_num[0],
                   'waste_num_2':total_collect_waste_num[1],
                   'waste_num_3':total_collect_waste_num[2],
                   'waste_num_4':total_collect_waste_num[3],
                   'apple_num':total_collect_apple_num,
                   'actor loss': total_actor_loss,
                    'critic loss': total_critic_loss,
                    'self_loss':total_self_loss,
                    'image_loss':total_imagine_loss,
                    'factor_1to2':total_give_factor[0][1].mean(), 'factor_1to3':total_give_factor[0][2].mean(), 'factor_1to4':total_give_factor[0][3].mean(),
                    'factor_2to1':total_give_factor[1][0].mean(), 'factor_2to3':total_give_factor[1][2].mean(), 'factor_2to4':total_give_factor[1][3].mean(),
                    'factor_3to1':total_give_factor[2][0].mean(), 'factor_3to2':total_give_factor[2][1].mean(), 'factor_3to4':total_give_factor[2][3].mean(),
                    'factor_4to1':total_give_factor[3][0].mean(), 'factor_4to2':total_give_factor[3][1].mean(), 'factor_4to3':total_give_factor[3][2].mean()
                    })
        elif env.name == 'staghunt':
            wandb.log({
                    'reward_1':total_reward[0],
                   'reward_2':total_reward[1],
                   'reward_3':total_reward[2],
                   'reward_4':total_reward[3],
                   'total_reward':sum(total_reward),
                   'stag num':total_hunt_stag_num,
                   'hare num':total_hunt_hare_num,
                   'actor loss': total_actor_loss,
                    'critic loss': total_critic_loss,
                    # 'self_loss':total_self_loss,
                    # 'image_loss':total_imagine_loss,
                    # 'factor_1to2':total_give_factor[0][1].mean(), 'factor_1to3':total_give_factor[0][2].mean(), 'factor_1to4':total_give_factor[0][3].mean(),
                    # 'factor_2to1':total_give_factor[1][0].mean(), 'factor_2to3':total_give_factor[1][2].mean(), 'factor_2to4':total_give_factor[1][3].mean(),
                    # 'factor_3to1':total_give_factor[2][0].mean(), 'factor_3to2':total_give_factor[2][1].mean(), 'factor_3to4':total_give_factor[2][3].mean(),
                    # 'factor_4to1':total_give_factor[3][0].mean(), 'factor_4to2':total_give_factor[3][1].mean(), 'factor_4to3':total_give_factor[3][2].mean()
                    })
        elif env.name == 'snowdrift':
            wandb.log({
                    'reward_1':total_reward[0],
                    'reward_2':total_reward[1],
                    'reward_3':total_reward[2],
                    'reward_4':total_reward[3],
                    'total_reward':sum(total_reward),
                    'snow_num':total_sd_num/env_parallel_num,
                   'actor loss': total_actor_loss,
                    'critic loss': total_critic_loss,
                    'self_loss':total_self_loss,
                    'image_loss':total_imagine_loss,
                    'factor_1to2':total_give_factor[0][1].mean(), 'factor_1to3':total_give_factor[0][2].mean(), 'factor_1to4':total_give_factor[0][3].mean(),
                    'factor_2to1':total_give_factor[1][0].mean(), 'factor_2to3':total_give_factor[1][2].mean(), 'factor_2to4':total_give_factor[1][3].mean(),
                    'factor_3to1':total_give_factor[2][0].mean(), 'factor_3to2':total_give_factor[2][1].mean(), 'factor_3to4':total_give_factor[2][3].mean(),
                    'factor_4to1':total_give_factor[3][0].mean(), 'factor_4to2':total_give_factor[3][1].mean(), 'factor_4to3':total_give_factor[3][2].mean()
                    })
        elif env.name == 'coingame':
            wandb.log({
                    'reward_1':total_reward[0],
                   'reward_2':total_reward[1],
                   'total_reward':sum(total_reward),
                    '1to1 coin':total_coin_1to1,
                    '1to2 coin':total_coin_1to2,
                    '2to1 coin':total_coin_2to1,
                    '2to2 coin':total_coin_2to2,
                   'actor loss': total_actor_loss,
                    'critic loss': total_critic_loss,
                    
                    'self_loss':total_self_loss,
                    'image_loss':total_imagine_loss,
                    'factor_1to2':total_give_factor[0][1].mean(),
                    'factor_2to1':total_give_factor[1][0].mean(), 
                    })
        elif env.name == 'mp_cleanup':
            wandb.log({
                   'total_reward':sum(total_reward),
                   'actor loss': total_actor_loss,
                    'critic loss': total_critic_loss,
                    # 'self_loss':total_self_loss,
                    # 'image_loss':total_imagine_loss,
                    # 'factor_1to2':total_give_factor[0][1].mean(), 'factor_1to3':total_give_factor[0][2].mean(), 'factor_1to4':total_give_factor[0][3].mean(),
                    # 'factor_2to1':total_give_factor[1][0].mean(), 'factor_2to3':total_give_factor[1][2].mean(), 'factor_2to4':total_give_factor[1][3].mean(),
                    # 'factor_3to1':total_give_factor[2][0].mean(), 'factor_3to2':total_give_factor[2][1].mean(), 'factor_3to4':total_give_factor[2][3].mean(),
                    # 'factor_4to1':total_give_factor[3][0].mean(), 'factor_4to2':total_give_factor[3][1].mean(), 'factor_4to3':total_give_factor[3][2].mean()
                    })

    epsilon = epsilon - anneal_epsilon if epsilon > min_epsilon else epsilon
    #     count += 1