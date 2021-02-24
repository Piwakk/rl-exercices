import pickle
import argparse
import sys
import matplotlib

# matplotlib.use("Qt5agg")
matplotlib.use("TkAgg")
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

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# ---parameters---#
h = 300
lr = 3e-4  # rather use 1e-4 for lunar lander
gamma = 0.98
lam = 0.05
lam_ent = 1e-3
nu = 1e-2
n_rollout = 20
K = 5
beta = 1e-3
delta = 0.8
eps = 0.2
extension = "ppo"
mini_batch = 64
action_space = 4
nb_features = 8


class GAILAgent(nn.Module):
    def __init__(self, env, opt):
        super().__init__()

        # ---parameters---#
        self.epoch, self.iteration = 0, 0
        self.featureExtractor = opt.featExtractor(env)

        # ---create the actor and critic nn---#
        self.discriminator = nn.Sequential(
            nn.Linear(action_space + nb_features, h),
            nn.ReLU(),
            nn.Linear(h, 1),
            nn.Sigmoid(),
        )

        self.v = nn.Sequential(nn.Linear(nb_features, h), nn.ReLU(), nn.Linear(h, 1))

        # self.q = nn.Sequential(
        # 	nn.Linear(self.featureExtractor.outSize, h),
        # 	nn.ReLU(),
        # 	nn.Linear(h,action_space),
        # 	)

        self.pi = nn.Sequential(
            nn.Linear(nb_features, h),
            nn.ReLU(),
            nn.Linear(h, action_space),
            nn.Softmax(dim=-1),
        )

        # ---optimizers tools---#
        self.optim_disc = torch.optim.Adam(
            params=self.discriminator.parameters(), lr=lr
        )
        self.optim_v = torch.optim.Adam(params=self.v.parameters(), lr=lr)
        self.optim_pi = torch.optim.Adam(params=self.pi.parameters(), lr=lr)

        # ---memory---#
        self.memory = []

        # ---expert---#
        with open("data/expert.pkl", "rb") as handle:
            expert_data = pickle.load(handle)
            expert_states = expert_data[:, :nb_features]
            expert_actions = expert_data[:, nb_features:]
            self.expert_states = expert_states.contiguous()
            self.expert_actions = expert_actions.contiguous()

        # ---device---#
        self.longTensor = torch.empty(1, dtype=int, device="cpu")
        self.floatTensor = torch.empty(1, dtype=float, device=device)
        self.to(dtype=float, device=device)

    def toOneHot(self, actions):
        actions = actions.view(-1).to(self.longTensor)
        oneHot = torch.zeros(actions.size()[0], action_space).to(self.floatTensor)
        oneHot[range(actions.size()[0]), actions] = 1.0
        return oneHot.to(self.floatTensor)

    def toIndexAction(self, oneHot):
        ac = self.longTensor.new(range(nbActions)).view(1, -1)
        ac = ac.expand(oneHot.size()[0], -1).contiguous().view(-1)
        actions = ac[oneHot.view(-1) > 0].view(-1)
        return actions

    def act(self, ob, reward, done):
        with torch.no_grad():
            phi = self.featureExtractor.getFeatures(ob)
            s = torch.tensor(
                self.featureExtractor.getFeatures(ob), device=device, dtype=float
            )
            probs = self.pi(s)
            action = probs.argmax(dim=-1).item()
            # m = Categorical(probs)
            # try :
            # 	action = m.sample().item()
            # except RuntimeError:
            # 	print('m:', m, 'phi:', phi, 's:', s, 'probs:', probs)
            # 	raise RuntimeError
        return action, probs[action]

    def store(self, ob, action, ob_new, reward, done, prob_action):
        self.memory.append(
            [
                self.featureExtractor.getFeatures(ob).reshape(-1),
                self.featureExtractor.getFeatures(ob_new).reshape(-1),
                action,
                reward,
                float(done),
                float(prob_action),
            ]
        )

    def sample_expert(self, T):
        perm = torch.randperm(self.expert_states.size(0))
        idx = perm[: min(mini_batch, T)]
        s = self.expert_states[idx]
        a = self.expert_actions[idx]
        return torch.cat((s, a), dim=-1).to(dtype=float)

    def sample_agent(self):
        states, actions = (
            torch.tensor([e[0] for e in self.memory], device=device, dtype=float),
            torch.tensor([e[2] for e in self.memory], device=device, dtype=float),
        )
        perm = torch.randperm(states.size(0))
        idx = perm[:mini_batch]
        s = states[idx]
        a = actions[idx]
        a = self.toOneHot(a)
        return torch.cat((s, a), dim=-1).to(dtype=float)

    def sample(self):
        s, s_prime, actions, reward, done, prob_actions = (
            torch.tensor([e[0] for e in self.memory], device=device, dtype=float),
            torch.tensor([e[1] for e in self.memory], device=device, dtype=float),
            torch.tensor([e[2] for e in self.memory], device=device, dtype=int),
            torch.tensor([e[3] for e in self.memory], device=device, dtype=float).view(
                -1, 1
            ),
            torch.tensor([e[4] for e in self.memory], device=device, dtype=float).view(
                -1, 1
            ),
            torch.tensor([e[5] for e in self.memory], device=device, dtype=float),
        )

        self.memory = []
        return s, s_prime, actions, reward, done, prob_actions

    def train(self, beta):

        # adversial
        T = len(self.memory)
        x_expert, x_agent = self.sample_expert(T), self.sample_agent()
        x_agent = x_agent.to(dtype=float)
        one = torch.ones_like(x_expert)
        adv_loss = torch.log(self.discriminator(x_expert)) + torch.log(
            one - self.discriminator(x_agent)
        )
        adv_loss = -adv_loss.mean()
        self.optim_disc.zero_grad()
        adv_loss.backward()
        self.optim_disc.step()
        with torch.no_grad():
            for param in self.discriminator.parameters():
                param.add_(torch.randn(param.size()) * nu)

        # loading memory
        s, s_prime, a, reward, done, prob_sample = self.sample()

        # critic
        v_s = self.v(s).flatten()
        x = torch.cat((s, self.toOneHot(a)), dim=-1)
        # print("x:", x)
        # print("x.shape:", x.shape)
        r_p = torch.log(self.discriminator(x).flatten())
        r_p = torch.where(r_p > -100.0, r_p, -100.0)
        R = [r_p[i:].mean() for i in range(T)]
        adv = torch.stack(R).to(device=device, dtype=float)
        # print('adv.shape:', adv.shape)
        # print('v_s:', v_s.shape)
        # td_target = reward + gamma * (1 - done) * v_s_p
        # delta = td_target - v_s
        # A = 0
        # adv = []
        # for d in delta.detach().view(-1).numpy()[::-1]:                     # calculating A requires a reverse loop on delta
        # 	A = d + gamma * lam * A
        # 	adv.insert(0,A)
        # adv = torch.tensor(adv, device=device, dtype=float).view(-1,1)
        critic_loss = F.smooth_l1_loss(
            v_s, adv.detach()
        )  # what permits a better judgment
        self.optim_v.zero_grad()
        critic_loss.backward()
        self.optim_v.step()

        # actor learning in k iteration with ppo
        for _ in range(K):
            pi = self.pi(s)
            prob = pi.gather(
                1, a.view(1, -1)
            ).flatten()  # calculate the new probabilities for the actions already taken
            # print("prob.shape:", prob.shape)
            # print("prob_sample.shape:", prob_sample.shape)
            ratio = torch.exp(
                torch.log(prob) - torch.log(prob_sample)
            )  # a/b = exp(log(a)-log(b))
            # print("ratio:", ratio)
            # print("ratio.shape:", ratio.shape)
            surr1 = ratio * adv.detach()
            surr2 = torch.clamp(ratio, 1 - eps, 1 + eps) * adv.detach()
            # print("surr1.shape:", surr1.shape)
            # print("surr2.shape:", surr2.shape)
            L = -torch.min(surr1, surr2)
            # print("L:", L)
            # print("L.shape:", L.shape)
            L = L.mean()
            H = (pi * torch.log(pi)).mean(dim=-1)
            H = H.mean()
            self.optim_pi.zero_grad()
            (L - lam_ent * H).backward()
            self.optim_pi.step()
        self.iteration += 1

        with torch.no_grad():
            prob_new_actions = self.pi(s).gather(1, a.view(1, -1))
            d_kl = (
                prob_new_actions
                * (torch.log(prob_new_actions) / torch.log(prob_sample))
            ).mean()  # KL div btw theta k et theta k+1
        return adv_loss.item(), L.item(), H.item(), critic_loss.item(), d_kl.item()


if __name__ == "__main__":

    # ---environment---#
    config = load_yaml("./configs/config_random_lunar.yaml")

    env = gym.make(config.env)
    if hasattr(env, "setPlan"):
        env.setPlan(config.map, config.rewards)
    tstart = str(time.time()).replace(".", "_")
    env.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    episode_count = config.nbEpisodes
    ob = env.reset()

    # ---agent---#
    agent_id = f"_h{h}_lr{lr}_g{gamma}_delta{delta}_beta{beta}"
    agent_dir = f'models/{config["env"]}/'
    os.makedirs(agent_dir, exist_ok=True)
    savepath = Path(f"{agent_dir}{agent_id}.pch")
    agent = GAILAgent(env, config)
    # agent.load(savepath)                        # the agent already exists

    # ---yaml and tensorboard---#
    outdir = "./runs/" + config.env + "/ppo/" + agent_id + "_" + tstart
    print("Saving in " + outdir)
    os.makedirs(outdir, exist_ok=True)
    save_src(os.path.abspath(outdir))
    write_yaml(os.path.join(outdir, "info.yaml"), config)
    writer = SummaryWriter(outdir)
    rsum = 0
    mean = 0
    verbose = True
    itest = 0
    reward = 0
    done = False
    for i in range(episode_count):
        if i % int(config.freqVerbose) == 0 and i >= config.freqVerbose:
            verbose = True
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
            with open(savepath, "wb") as f:
                torch.save(agent, f)
        j = 0
        if verbose:
            env.render()

        done = False
        while not (done):
            for _ in range(n_rollout):
                if verbose:
                    env.render()
                action, prob = agent.act(
                    ob, reward, done
                )  # choose action and determine the prob of that action
                ob_new, reward, done, _ = env.step(action)  # process action
                agent.store(
                    ob, action, ob_new, reward, done, prob
                )  # storing the transition
                ob = ob_new
                j += 1
                rsum += reward
                if done:
                    print(f"{i} rsum={rsum}, {j} actions")
                    writer.add_scalar("reward", rsum, i)
                    agent.nbEvents = 0
                    mean += rsum
                    rsum = 0
                    ob = env.reset()
                    break
            adv_loss, L, H, critic_loss, d_kl = agent.train(beta)
        writer.add_scalar("adversial", adv_loss, i)
        writer.add_scalar("actor", L, i)
        writer.add_scalar("critic", critic_loss, i)
        writer.add_scalar("entropy", H, i)
        writer.add_scalar("d_kl", d_kl, i)
        agent.epoch += 1
    env.close()
