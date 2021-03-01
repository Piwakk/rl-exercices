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
lr_actor = 3e-3   # lr pi
lam_ent = 1e-3
n_rollout = 1000  # freqence des optimisations
action_space = 4  # nbre actions
nb_features = 8   # tailles des étapes

class BCAgent():

    def __init__(self, env, opt):

        #---parameters---#
        self.epoch , self.iteration = 0, 0
        self.featureExtractor = opt.featExtractor(env)

        #---networks---#
        self.pi  = nn.Sequential(
            nn.Linear(nb_features, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, action_space),
            nn.Softmax(dim=-1),
            ).to(device, dtype=float)

        #---optimizers tools---#
        self.optim_pi = torch.optim.Adam(params=self.pi.parameters(), lr=lr_actor)

        #---memory---#
        self.memory = []

        #---device---#
        self.longTensor = torch.empty(1, dtype=int, device='cpu')
        self.floatTensor = torch.empty(1, dtype=float, device=device)

        #---expert---#
        with open('data/expert.pkl','rb') as handle :
            expert_data = pickle.load(handle) 
            expert_states = expert_data [: ,:nb_features]
            expert_actions = expert_data[:, nb_features:]
            self.expert_states = expert_states.contiguous()
            self.expert_actions = expert_actions.contiguous().argmax(dim=-1).to(dtype=int)


    def toOneHot(self, actions):
        actions=actions.view(-1).to(self.longTensor)
        oneHot = torch.zeros(actions.size()[0], action_space).to(self.floatTensor)
        oneHot[range(actions.size()[0]), actions] = 1.
        return oneHot.to(self.floatTensor)

    def toIndexAction(self, oneHot):
        nbActions = len(oneHot)
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

    def sample_expert(self):
        """
        Sample a minibatch from expert data
        """
        s = torch.tensor(self.expert_states, dtype=float, device=device)
        a = torch.tensor(self.expert_actions, dtype=float, device=device)
        return s, a

    def train(self):

        s, a = self.sample_expert()
        a = a.to(dtype=int)
        T = len(s)
        idx = torch.arange(0,T, dtype=int, device=device)
        pi = self.pi(s)[idx, a]
        log_pi = torch.log(pi)
        L = log_pi.mean()

        H = (pi*log_pi).mean(dim=-1)
        H = - H.mean()

        self.optim_pi.zero_grad()
        (- L - lam_ent * H).backward()
        self.optim_pi.step()

        self.iteration += 1

        return L.item(), H.item() 


if __name__ == '__main__':

    #---environment---#
    config = load_yaml('./configs/config_random_lunar_bc.yaml')

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
    agent_id = f'BC_lr{lr_actor}_rollout{n_rollout}'
    agent_dir = f'models/{config["env"]}/'
    os.makedirs(agent_dir, exist_ok=True)
    savepath = Path(f'{agent_dir}{agent_id}.pch')
    agent = BCAgent(env, config)
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
            ob = ob_new
            j += 1
            it +=1
            rsum += reward
            if it % n_rollout == 0 and i > 0:
                L, H = agent.train()
                writer.add_scalar("loss/actor", L, it)
                writer.add_scalar("regularisation/entropy", H, it)
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
