import argparse
import sys
import matplotlib
#matplotlib.use("Qt5agg")
matplotlib.use("TkAgg")
import gym
import gridworld
import torch
from utils import *
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import torch
from random import random
from pathlib import Path
from memory import Memory
import numpy as np

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Qfunction(nn.Module):
    def __init__(self, size_in, size_out, layers=[], activation=nn.ReLU()):
        super(Qfunction,self).__init__()
        self.module = nn.ModuleList([])   
        for x in layers:
            self.module.append(nn.Linear(size_in, x))
            self.module.append(activation)
            size_in = x
        self.module.append(nn.Linear(size_in, size_out))
    
    def setcuda(self, device):
        #FeatureExtractor.floatTensor = torch.cuda.FloatTensor(1, device=device)
        #FeatureExtractor.longTensor = torch.cuda.LongTensor(1, device=device)
        self.cuda(device=device)

    def forward(self, x, g):
        x = torch.cat((x,g), dim=-1)
        for l in self.module:
            x = l(x)
        return x

class DQNAgent(object):

    def __init__(self, env, opt, H=[30], lr=1e-1, gamma=0.95,
        eps0=1., nu=1e-1, freq_update_target=100):
        self.opt=opt
        self.env=env
        if opt.fromFile is not None:
            self.load(opt.fromFile)
    
        #---parameters---#
        self.action_space = env.action_space
        self.featureExtractor = opt.featExtractor(env)
        # self.eps = lambda t: min(0.1, eps0 / (1 + nu * t))
        self.eps = lambda t : eps0
        self.epoch , self.iteration, self.update_target = 0, 0, 0
        self.freq_update_target = freq_update_target

        #---create the two Q functions---#
        self.q = Qfunction(size_in=2*self.featureExtractor.outSize, 
            size_out=self.action_space.n,
            layers=H,
            activation=nn.Hardtanh())
        self.q_hat = Qfunction(size_in=2*self.featureExtractor.outSize, 
            size_out=self.action_space.n,
            layers=H,
            activation=nn.Hardtanh())
        self.q_hat.load_state_dict(self.q.state_dict()) # they start with the exact same weights
        self.q.to(dtype=float, device=device)  
        self.q_hat.to(dtype=float, device=device)

        #---optimizers tools---#
        self.optim = torch.optim.Adam(
            params=self.q.parameters(), lr=lr)
        # self.criterion = torch.nn.SmoothL1Loss().to(dtype=float)
        self.criterion = torch.nn.MSELoss().to(dtype=float)



    def act(self, observation, reward, done, t, goal):
        if random()<self.eps(t):
            return self.action_space.sample()
        with torch.no_grad(): 
            s = torch.tensor(self.featureExtractor.getFeatures(observation), device=device, dtype=float)
            g = torch.tensor(goal, device=device, dtype=float)
            q = self.q(s,g)
            a = int(np.argmax(q))
            return a

    def train(self, batch):
        
        # data
        n = len(batch)
        phi, actions, phi_new, reward, done, goal = [], [], [], [], [], []
        for e in batch:
            phi.append(e[0])
            actions.append(e[1])
            phi_new.append(e[2])
            reward.append(e[3])
            done.append(e[4])
            goal.append(e[5])
        s = torch.tensor(phi, device=device, dtype=float).view(n,-1)
        a = torch.tensor(actions, device=device, dtype=int)
        s_p = torch.tensor(phi_new, device=device, dtype=float).view(n,-1)
        r = torch.tensor(reward, device=device, dtype=float)
        done = torch.tensor(done, device=device, dtype=float)
        g = torch.tensor(goal, device=device, dtype=float).view(n,-1)

        y = r + gamma*torch.max(agent.q_hat(s_p, g), dim=1).values
        indices = torch.arange(n, device=device)
        q = self.q(s, g)[indices, a]
        loss = self.criterion(y.detach(), q)

        # learning
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.iteration += 1

        # update target
        if self.iteration % self.freq_update_target == 0:
            self.q_hat.load_state_dict(self.q.state_dict())
            self.update_target += 1
        return loss


if __name__ == '__main__':

    #---environment---#
    config = load_yaml('./configs/config_random_gridworldHER.yaml')
    # config = load_yaml('./configs/config_random_cartpole.yaml')
    # config = load_yaml('./configs/config_random_lunar.yaml')
    freqTest = config.freqTest
    freqSave = config.freqSave
    nbTest = config.nbTest
    env = gym.make(config.env)
    if hasattr(env, 'setPlan'):
        env.setPlan(config.map, config.rewards)
    tstart = str(time.time())
    tstart = tstart.replace(".", "_")
    env.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    episode_count = config.nbEpisodes
    ob = env.reset()

    featureExtractor = config.featExtractor(env)
    print("fe", featureExtractor.outSize)

    #---parameters---#
    H = [128,128]
    lr = 1e-3
    gamma = 0.99
    eps0 = 0.2
    nu = 1e-1
    freq_update_target = 100
    mem_size = 1000000
    mini_batch = 1000
    freqOptim = 10

    #---agent---#
    agent_id = f'HER_h{H}_lr{lr}_g{gamma}_eps0{eps0}_nu{nu}_clear{freq_update_target}'
    agent_dir = f'models/{config["env"]}/'
    os.makedirs(agent_dir, exist_ok=True)
    savepath = Path(f'{agent_dir}{agent_id}.pch')
    agent = DQNAgent(env, config, H=H, lr=lr, gamma=gamma,
        eps0=eps0, nu=nu, freq_update_target=freq_update_target)
    # agent.load(savepath)                        # the agent already exists
        

    #---yaml and tensorboard---#
    outdir = "./XP/" + config.env + "/dqn_" + "-" + agent_id + "-" + tstart
    print("Saving in " + outdir)
    os.makedirs(outdir, exist_ok=True)
    save_src(os.path.abspath(outdir))
    write_yaml(os.path.join(outdir, 'info.yaml'), config)
    logger = LogMe(SummaryWriter(outdir))
    # loadTensorBoard(outdir)

    #---buffer---#
    replay = Memory(mem_size=mem_size, prior=False)

    rsum = 0
    mean = 0
    verbose = True
    itest = 0
    reward = 0
    done = False
    for i in range(episode_count):
        if i % int(config.freqVerbose) == 0 and i >= config.freqVerbose:
            verbose = False # True
        else:
            verbose = False

        if i % freqTest == 0 and i >= freqTest:  ##### Same as train for now
            print("Test time! ") #remember that isa loves you
            mean = 0
            agent.test = True

        if i % freqTest == nbTest and i > freqTest:
            print("End of test, mean reward=", mean / nbTest)
            itest += 1
            logger.direct_write("rewardTest", mean / nbTest, itest)
            agent.test = False

        if i % freqSave == 0:
            pass
            # with savepath.open("wb") as fp:
            #     agent.save(savepath)
        j, k = 0, 0
        if verbose:
            env.render()

        loss = 0
        goal, _ = env.sampleGoal()
        goal = featureExtractor.getFeatures(goal)
        temp_replay = []

        while True:
            if verbose:
                env.render()

            action = agent.act(ob, reward, done, i, goal)
            # ob_new, reward, done, _ = env.step(action)
            ob_new, _, _, _ = env.step(action)
            done = (featureExtractor.getFeatures(ob_new)==goal).all()
            reward = 1. if done else -0.1
            temp_replay.apend((featureExtractor.getFeatures(ob), action, featureExtractor.getFeatures(ob_new), reward, done, goal))
            
            ob = ob_new
            if i % freqOptim == 0 and replay.nentities > 2000:
                batch = replay.sample(n=mini_batch)
                loss += agent.train(batch=batch)
                k += 1
            j+=1

            rsum += reward
            if done or j == 100:
                # HER replay
                for s, a, s_p, r, d, g in temp_replay:
                    replay.store((s, a, s_p, r, d, g))
                g_artifical = temp_replay[-1][2] # the last state
                for s, a, s_p, _, _, _ in temp_replay:
                    d = (s_p==g_artifical).all()
                    r = 1. if d else -0.1
                    replay.store((s, a, s_p, r, d, g))
                temp_replay = []
                x, y = featureExtractor.getFeatures(ob)
                print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")
                logger.direct_write("reward", rsum, i)
                logger.direct_write("loss", loss/max(k,1), i)
                logger.direct_write("finalposition/x", x, i)
                logger.direct_write("finalposition/y", y, i)
                agent.nbEvents = 0
                mean += rsum
                rsum = 0
                ob = env.reset()
                break
        agent.epoch += 1
    env.close()