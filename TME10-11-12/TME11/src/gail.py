import pickle
import argparse
import sys
import matplotlib
#matplotlib.use("Qt5agg")
#matplotlib.use("TkAgg")
import gym
import torch
from utils import *
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import torch.nn.functional as F
import torch
from torch.distributions import Categorical
from random import random
from pathlib import Path
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#---parameters---#
h = 120           # taille des réseaux actor, critic et discriminator
lr_actor = 3e-4   # lr pi
lr_disc = 3e-4    # lr discriminateur
lr_critic = 3e-4  # ls critique
gamma = 0.98      # inutile dans l'approche moyenne des  reward
lam = 0.05        # inutile 
lam_ent = 1e-2    # poid de l'entropie dans l'apprentissage
nu = 1e-2         # variance du bruit des actions
n_rollout = 1000  # freqence des optimisations
K = 10            # nombre de pas de la politique pour chaque apprentissage
beta = 1e-3       # poid de la dk_l (inutile dans l'approche cliped)
delta = .8        # taille maximale du pas (inutile dans l'approche cliped)
eps = 0.2         # cliped des ratio d'is
eps_disc = 1e-4   # cliped des valeurs de sorties du discriminateur (éviter les log(0))
extension = 'cliped' # choix cliped ou d_kl
reward_type = 'positive' #'positive' #'negative'
baseline = True   # utilisation d'une baseline
mini_batch = 1000 # taille maximale des batchs d'apprentissage du discriminateur
action_space = 4  # nbre actions
nb_features = 8   # tailles des étapes
p = .4            # ratio dropout

class GAILAgent():

    def __init__(self, env, opt):

        #---parameters---#
        self.epoch , self.iteration = 0, 0
        self.featureExtractor = opt.featExtractor(env)

        #---networks---#
        self.discriminator = nn.Sequential(
            nn.Linear(action_space + nb_features, h),
            # nn.BatchNorm1d(h),
            # nn.Dropout(p=p),
            nn.Tanh(),
            nn.Linear(h, 1),
            nn.Sigmoid()).to(device, dtype=float)

        self.v = nn.Sequential(
            nn.Linear(nb_features, h),
            nn.Tanh(),
            nn.Linear(h,1),
            ).to(device, dtype=float)
        
        self.pi  = nn.Sequential(
            nn.Linear(nb_features, h),
            nn.Tanh(),
            nn.Linear(h, action_space),
            nn.Softmax(dim=-1),
            ).to(device, dtype=float)

        #---optimizers tools---#
        self.optim_disc = torch.optim.Adam(params=self.discriminator.parameters(), lr=lr_disc)
        self.optim_v = torch.optim.Adam(params=self.v.parameters(), lr=lr_critic)
        self.optim_pi = torch.optim.Adam(params=self.pi.parameters(), lr=lr_actor)

        #---memory---#
        self.memory = []

        #---expert---#
        with open('data/expert.pkl','rb') as handle :
            expert_data = pickle.load(handle) 
            expert_states = expert_data [: ,:nb_features]
            expert_actions = expert_data[:, nb_features:]
            self.expert_states = expert_states.contiguous()
            self.expert_actions = expert_actions.contiguous()

        #---device---#
        self.longTensor = torch.empty(1, dtype=int, device='cpu')
        self.floatTensor = torch.empty(1, dtype=float, device=device)

    def toOneHot(self, actions):
        actions=actions.view(-1).to(self.longTensor)
        oneHot = torch.zeros(actions.size()[0], action_space).to(self.floatTensor)
        oneHot[range(actions.size()[0]), actions] = 1.
        return oneHot.to(self.floatTensor)

    def toIndexAction(self, oneHot):
        ac = self.longTensor.new(range(nbActions)).view(1, -1) 
        ac = ac.expand(oneHot.size()[0], -1).contiguous().view(-1) 
        actions = ac[oneHot.view(-1)>0].view(-1)
        return actions


    def act(self, ob, reward, done):
        with torch.no_grad():
            phi = self.featureExtractor.getFeatures(ob)
            s = torch.tensor(self.featureExtractor.getFeatures(ob),  device=device, dtype=float)
            probs = self.pi(s)
            # action = probs.argmax(dim=-1).item() # for deterministic choice
            m = Categorical(probs) # for probabilistic choice
            try :
                action = m.sample().item()
            except RuntimeError:
                print('m:', m, 'phi:', phi, 's:', s, 'probs:', probs)
                raise RuntimeError
        return action, probs[action]

    def store(self, ob, action, ob_new, reward, done, prob_action):
        """
        Store the transition in the replay buffer
        """
        self.memory.append([
            self.featureExtractor.getFeatures(ob).reshape(-1),
            self.featureExtractor.getFeatures(ob_new).reshape(-1),
            action, reward, float(done), float(prob_action)])

    def sample_expert(self, T):
        """
        Sample a minibatch from expert data
        """
        s_list = []
        a_list = []
        length = 0
        while length < T:
            perm = torch.randperm(self.expert_states.size(0))
            idx = perm[:min(mini_batch,T)]
            s = self.expert_states[idx]
            a = self.expert_actions[idx]
            s_list.append(s)
            a_list.append(a)
            length += len(s)
        s = torch.cat(s_list, dim=0)
        a = torch.cat(a_list, dim=0)
        s_a = torch.cat((s, a), dim=-1).to(dtype=float, device=device)
        return s_a[:T]

    def sample_agent(self):
        """
        Sample a minibatch from expert data
        """
        states, actions = torch.tensor([e[0] for e in self.memory], device=device, dtype=float),  \
                            torch.tensor([e[2] for e in self.memory], device=device, dtype=float),
        perm = torch.randperm(states.size(0))
        idx = perm[:mini_batch]
        s = states[idx]
        a = actions[idx]
        a = self.toOneHot(a)
        return torch.cat((s, a), dim=-1).to(dtype=float, device=device)

    def sample(self):
        s, s_prime, actions, reward, done, prob_actions = \
            torch.tensor([e[0] for e in self.memory], device=device, dtype=float),  \
            torch.tensor([e[1] for e in self.memory], device=device, dtype=float),  \
            torch.tensor([e[2] for e in self.memory], device=device, dtype=int),    \
            torch.tensor([e[3] for e in self.memory], device=device, dtype=float).view(-1,1),  \
            torch.tensor([e[4] for e in self.memory], device=device, dtype=float).view(-1,1), \
            torch.tensor([e[5] for e in self.memory], device=device, dtype=float),

        self.memory = []
        return s, s_prime, actions, reward, done, prob_actions

    def train(self, beta):

        # adversial
        T = len(self.memory)
        
        self.discriminator.train()                                             # to use dropout and batchnorm
        for _ in range(K):
            x_expert, x_agent = self.sample_expert(T), self.sample_agent()     # get data for discriminator
            score_expert = self.discriminator(x_expert)
            # score_expert = torch.clamp(score_expert, eps_disc, 1 - eps_disc) # prevent 0 values
            score_agent = self.discriminator(x_agent)
            # score_agent = torch.clamp(score_agent, eps_disc, 1 - eps_disc)   # prevent 0 values
            one = torch.ones_like(score_agent, device=device)
            adv_loss = torch.log(score_expert) + torch.log(one - score_agent)  # discriminator loss
            adv_loss = - adv_loss.mean()
            self.optim_disc.zero_grad()
            adv_loss.backward()
            self.optim_disc.step()
            with torch.no_grad():
                for param in self.discriminator.parameters():
                    param.add_(torch.randn(param.size(), device=device) * nu)

        # loading memory
        s, s_prime, action, reward, done, prob_sample = self.sample()


        # critic
        v_s = self.v(s).flatten()
        self.discriminator.eval()
        x = torch.cat((s, self.toOneHot(action)), dim=-1)
        r = self.discriminator(x).flatten()
        # if reward_type == 'negative':
            # r = torch.clamp(r, eps_disc, 1 - eps_disc)
            # r_p = torch.log(r)
        # elif reward_type == 'positive':

        # rewards
        one = torch.ones_like(r, device=device)
        r_p = - torch.log(torch.clamp(one - r, eps_disc, 1 - eps_disc))
        if baseline:
             r_p -= v_s

        # else :
        #     r = torch.clamp(r, eps_disc, 1 - eps_disc)
        #     one = torch.ones_like(r, device=device)
        #     one_m_r = one - r
        #     one_m_r = torch.clamp(one_m_r, eps_disc, 1 - eps_disc)
        #     r_p = torch.log(r) - torch.log(one_m_r)
        # r_p = torch.clamp(r_p, min=-100.)
        
        # divide trajectories
        idx_traj  = list(((done == 1).nonzero(as_tuple=True)[0]).cpu()) # get list of limits of trajectories
        tupple_idx_traj = [(idx_traj[i]+1, idx_traj[i+1]+1) for i in range(len(idx_traj)-1)] # get coordonates of begining and end of trajectories
        tupple_idx_traj.insert(0, (0, idx_traj[0]+1))
        tupple_idx_traj.append((idx_traj[-1]+1, T))

        R = [ r_p[i_start:i_end].flip(0).cumsum(0).flip(0) / 
                torch.arange(1, i_end - i_start + 1, device=device).flip(0) for i_start, i_end in tupple_idx_traj ]

        adv = torch.cat(R).to(device=device, dtype=float)
        y = adv + v_s
        adv = (adv - adv.mean()) / adv.std()        
        
        critic_loss = F.smooth_l1_loss(v_s, y.detach())
        self.optim_v.zero_grad()
        critic_loss.backward()
        self.optim_v.step()

        idx = torch.range(0,T-1, dtype=int, device=device)
        for _ in range(K):
            pi = self.pi(s)
            prob = pi[idx, action]
            ratio = torch.exp(torch.log(prob) - torch.log(prob_sample))
            
            # if extension=='cliped':
            surr1 = ratio * adv.detach()
            surr2 = torch.clamp(ratio, 1 - eps, 1 + eps) * adv.detach()
            L = torch.min(surr1, surr2)
            L = L.mean()
            actor_loss = - L
            # elif extension=='DKL':
            #     L = (ratio * adv.detach()).mean()
            #     d_kl = -(prob_sample * (torch.log(prob_sample) / torch.log(prob))).mean()
            #     actor_loss = (- L - beta * d_kl) 
            H = (pi*torch.log(pi)).mean(dim=-1)
            H = - H.mean()
            self.optim_pi.zero_grad()
            (actor_loss - lam_ent * H).backward()
            self.optim_pi.step()


        with torch.no_grad():
            prob_new_actions = self.pi(s)[idx, action]
            # d_kl = (prob_new_actions * (torch.log(prob_new_actions)/\
            #     torch.log(prob_sample))).mean()
            d_kl = F.kl_div(prob_new_actions, prob_sample)

        self.iteration += 1 
        return score_expert.mean().item(), score_agent.mean().item(), adv_loss.item(), L.item(), H.item(), critic_loss.item(), d_kl.item() 


if __name__ == '__main__':

    #---environment---#
    config = load_yaml('./configs/config_random_lunar.yaml')

    env = gym.make(config.env)
    if hasattr(env, 'setPlan'):
        env.setPlan(config.map, config.rewards)
    tstart = str(time.time()).replace(".", "_")
    env.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    episode_count = config.nbEpisodes
    ob = env.reset()

    #---agent---#
    agent_id = f'_h{h}_lrct{lr_critic}_lract{lr_actor}_lrdisc{lr_disc}_nu{nu}_K{K}_eps{eps}_epsdisc{eps_disc}_ext{extension}_rwd{reward_type}_p{p}_baseline{baseline}'
    agent_dir = f'models/{config["env"]}/'
    os.makedirs(agent_dir, exist_ok=True)
    savepath = Path(f'{agent_dir}{agent_id}.pch')
    agent = GAILAgent(env, config)
    # agent.load(savepath)                        # the agent already exists
        
    #---yaml and tensorboard---#
    outdir = "./runs/" + config.env + "/ppo/" + agent_id + "_" + tstart
    print("Saving in " + outdir)
    os.makedirs(outdir, exist_ok=True)
    save_src(os.path.abspath(outdir))
    write_yaml(os.path.join(outdir, 'info.yaml'), config)
    writer = SummaryWriter(outdir)
    rsum = 0
    mean = 0
    verbose = True
    itest = 0
    it = 0
    reward = 0
    done = False
    for i in range(episode_count):
        if i % int(config.freqVerbose) == 0 and i >= config.freqVerbose:
            verbose = False #True
        else:
            verbose = False

        if i % config.freqTest == 0 and i >= config.freqTest:
            print("Test time! ")
            mean = 0
            agent.test = True

        if i % config.freqTest == config.nbTest and i > config.freqTest:
            print("End of test, mean reward=", mean / config.nbTest)
            itest += 1
            writer.add_scalar("rewardTest", mean / config.nbTest, itest)
            agent.test = False

        if i % config.freqSave == 0:
            with open(savepath, 'wb') as f:
                torch.save(agent, f)
        j = 0
        if verbose:
            env.render()

        done = False
        while not(done):              
            if verbose:
                env.render()
            action, prob = agent.act(ob, reward, done)          # choose action and determine the prob of that action
            ob_new, reward, done, _ = env.step(action)          # process action
            agent.store(ob, action, ob_new, reward, done, prob) # storing the transition
            ob = ob_new
            j += 1
            it +=1
            rsum += reward
            if it % n_rollout == 0 and i > 0:
                sc_expert, sc_agent, adv_loss, L, H, critic_loss, d_kl = agent.train(beta)
                if d_kl >= 1.5 * delta:                             # more value needs to be given to proximity loss
                    beta *= 2
                elif d_kl <= (2 / 3) * delta:                         # less value needs to be given to proximity loss
                    beta /= 2
                writer.add_scalar("discriminant/score expert", sc_expert, it)            
                writer.add_scalar("discriminant/score_agent", sc_agent, it)            
                writer.add_scalar("loss/adversial", adv_loss, it)
                writer.add_scalar("loss/actor", L, it)
                writer.add_scalar("loss/critic", critic_loss, it)
                writer.add_scalar("regularisation/entropy", H, it)
                writer.add_scalar("regularisation/d_kl", d_kl, it)
                writer.add_scalar("regularisation/Beta", beta, it)
                agent.epoch += 1
            if done:
                if i % config.freqPrint == 0:
                    print(f'{i} rsum={int(rsum)}, {j} actions')
                writer.add_scalar("reward", rsum, i)
                agent.nbEvents = 0
                mean += rsum
                rsum = 0
                ob = env.reset()
                break
    env.close()
