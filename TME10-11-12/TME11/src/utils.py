import time
import subprocess
from collections import namedtuple,defaultdict
import logging
import json
import os
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import threading

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class State:

    def __init__(self, model, optim, criterion):
        self.model = model
        self.optim = optim
        self.criterion = criterion
        self.epoch , self.iteration = 0,0

def loadTensorBoard(outdir):
    t = threading.Thread(target=launchTensorBoard, args=([outdir]))
    t.start()

def launchTensorBoard(tensorBoardPath):
    print('tensorboard --logdir=' + tensorBoardPath)
    ret=os.system('tensorboard --logdir='  + tensorBoardPath)
    if ret!=0:
        syspath = os.path.dirname(sys.executable)
        print(os.path.dirname(sys.executable))
        ret = os.system(syspath+"/"+'tensorboard --logdir=' + tensorBoardPath)
    return

class Orn_Uhlen:
    def __init__(self, n_actions, mu=0, theta=0.15, sigma=0.2):
        self.n_actions = n_actions
        self.X = np.ones(n_actions) * mu
        self.mu = mu
        self.sigma = sigma
        self.theta = theta

    def reset(self):
        self.X = np.ones(self.n_actions) * self.mu

    def sample(self):
        dX = self.theta * (self.mu - self.X)
        dX += self.sigma * np.random.randn(self.n_actions)
        self.X += dX
        return torch.FloatTensor(self.X)

class FeatureExtractor(object):
    def __init__(self):
        super().__init__()

    def getFeatures(self,obs):
        pass

class NothingToDo(FeatureExtractor):
    def __init__(self,env):
        super().__init__()
        ob=env.reset()
        self.outSize=len(ob)

    def getFeatures(self,obs):
        return obs

######  Pour Gridworld #############################"

class MapFromDumpExtractor(FeatureExtractor):
    def __init__(self,env):
        super().__init__()
        outSize = env.start_grid_map.reshape(1, -1).shape[1]
        self.outSize=outSize

    def getFeatures(self, obs):
        #prs(obs)
        return obs.reshape(1,-1)

class MapFromDumpExtractor2(FeatureExtractor):
    def __init__(self,env):
        super().__init__()
        outSize=env.start_grid_map.reshape(1, -1).shape[1]
        self.outSize=outSize*3

    def getFeatures(self, obs):
        state=np.zeros((3,np.shape(obs)[0],np.shape(obs)[1]))
        state[0]=np.where(obs == 2,1,state[0])
        state[1] = np.where(obs == 4, 1, state[1])
        state[2] = np.where(obs == 6, 1, state[2])
        return state.reshape(1,-1)




class DistsFromStates(FeatureExtractor):
    def __init__(self,env):
        super().__init__()
        self.outSize=16

    def getFeatures(self, obs):
        #prs(obs)
        #x=np.loads(obs)
        x=obs
        #print(x)
        astate = list(map(
            lambda x: x[0] if len(x) > 0 else None,
            np.where(x == 2)
        ))
        astate=np.array(astate)
        a3=np.where(x == 3)
        d3=np.array([0])
        if len(a3[0])>0:
            astate3 = np.concatenate(a3).reshape(2,-1).T
            d3=np.power(astate-astate3,2).sum(1).min().reshape(1)

            #d3 = np.array(d3).reshape(1)
        a4 = np.where(x == 4)
        d4 = np.array([0])
        if len(a4[0]) > 0:
            astate4 = np.concatenate(a4).reshape(2,-1).T
            d4 = np.power(astate - astate4, 2).sum(1).min().reshape(1)
            #d4 = np.array(d4)
        a5 = np.where(x == 5)
        d5 = np.array([0])
        #prs(a5)
        if len(a5[0]) > 0:
            astate5 = np.concatenate(a5).reshape(2,-1).T
            d5 = np.power(astate - astate5, 2).sum(1).min().reshape(1)
            #d5 = np.array(d5)
        a6 = np.where(x == 6)
        d6 = np.array([0])
        if len(a6[0]) > 0:
            astate6 = np.concatenate(a6).reshape(2,-1).T
            d6 = np.power(astate - astate6, 2).sum(1).min().reshape(1)
            #d6=np.array(d6)

        #prs("::",d3,d4,d5,d6)
        ret=np.concatenate((d3,d4,d5,d6)).reshape(1,-1)
        ret=np.dot(ret.T,ret)
        return ret.reshape(1,-1)

#######################################################################################


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

    def forward(self, x):
        for l in self.module:
            x = l(x)
        return x


class convMDP(nn.Module):
    def __init__(self, inSize, outSize, layers=[], convs=None, finalActivation=None, batchNorm=False,init_batchNorm=False,activation=torch.tanh):
        super(convMDP, self).__init__()
        #print(inSize,outSize)

        self.inSize=inSize
        self.outSize=outSize
        self.batchNorm=batchNorm
        self.init_batchNorm = init_batchNorm
        self.activation=activation

        self.convs=None
        if convs is not None:
            self.convs = nn.ModuleList([])
            for x in convs:
                self.convs.append(nn.Conv2d(x[0], x[1], x[2], stride=x[3]))
                inSize = np.sqrt(inSize / x[0])
                inSize=((inSize-x[2])/x[3])+1
                inSize=inSize*inSize*x[1]
        #print(inSize)

        self.layers = nn.ModuleList([])
        self.bn = nn.ModuleList([])
        i=0
        if batchNorm or init_batchNorm:
            self.bn.append(nn.BatchNorm1d(num_features=inSize))
        for x in layers:
            self.layers.append(nn.Linear(inSize, x))
            if batchNorm:
                self.bn.append(nn.BatchNorm1d(num_features=x))

            #nn.init.xavier_uniform_(self.layers[i].weight)
            nn.init.normal_(self.layers[i].weight.data, 0.0, 0.02)
            nn.init.normal_(self.layers[i].bias.data,0.0,0.02)
            i+=1
            inSize = x
        self.layers.append(nn.Linear(inSize, outSize))

        #nn.init.uniform_(self.layers[-1].weight)
        nn.init.normal_(self.layers[-1].weight.data, 0.0, 0.02)
        nn.init.normal_(self.layers[-1].bias.data, 0.0, 0.02)
        self.finalActivation=finalActivation

    def setcuda(self, device):
        self.cuda(device=device)

    def forward(self, x):
        #print("d", x.size(),self.inSize)
        x=x.view(-1,self.inSize)

        if self.convs is not None:

            n=x.size()[0]
            i=0
            for c in self.convs:
                if i==0:
                    w=np.sqrt(x.size()[1])
                    x=x.view(n,c.in_channels,w,w)
                x=c(x)
                x=self.activation(x)
                i+=1
            x=x.view(n,-1)

        #print(x.size())
        if self.batchNorm or self.init_batchNorm:
            x=self.bn[0](x)
        x = self.layers[0](x)


        for i in range(1, len(self.layers)):
            x = self.activation(x)
            #if self.drop is not None:
            #    x = nn.drop(x)
            if self.batchNorm:
                x = self.bn[i](x)
            x = self.layers[i](x)


        if self.finalActivation is not None:
            x=self.finalActivation(x)
        #print("f",x.size())
        return x

class NN(nn.Module):
    def __init__(self, inSize, outSize, layers=[]):
        super(NN, self).__init__()
        self.layers = nn.ModuleList([])
        for x in layers:
            self.layers.append(nn.Linear(inSize, x))
            inSize = x
        self.layers.append(nn.Linear(inSize, outSize))

    def setcuda(self, device):
        self.cuda(device=device)

    def forward(self, x):
        x = self.layers[0](x)
        for i in range(1, len(self.layers)):
            x = torch.tanh(x)
            x = self.layers[i](x)

        return x

class LogMe(dict):
    def __init__(self,writer,term=True):
        self.writer = writer
        self.dic = defaultdict(list)
        self.term = term
    def write(self,i):
        if len(self.dic)==0: return
        s=f"Epoch {i} : "
        for k,v in self.dic.items():
            self.writer.add_scalar(k,sum(v)*1./len(v),i)
            s+=f"{k}:{sum(v)*1./len(v)} -- "
        self.dic.clear()
        if self.term: logging.info(s)
    def update(self,l):
        for k,v in l:
            self.add(k,v)
    def direct_write(self,k,v,i):
        self.writer.add_scalar(k,v,i)
    def add(self,k,v):
        self.dic[k].append(v)

def save_src(path):
    current_dir = os.getcwd()
    package_dir = current_dir.split('RL', 1)[0]
    #path = os.path.abspath(path)
    os.chdir(package_dir)
    #print(package_dir)
    src_files = subprocess.Popen(('find', 'RL', '-name', '*.py', '-o', '-name', '*.yaml'),
                                 stdout=subprocess.PIPE)
    #print(package_dir,path)
    #path=os.path.abspath(path)


    #print(str(src_files))

    subprocess.check_output(('tar', '-zcf', path+"/arch.tar", '-T', '-'), stdin=src_files.stdout, stderr=subprocess.STDOUT)
    src_files.wait()
    os.chdir(current_dir)



def prs(*args):
    st = ""
    for s in args:
        st += str(s)
    print(st)


class DotDict(dict):
    """dot.notation access to dictionary attributes (Thomas Robert)"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def load_yaml(path):
    with open(path, 'r') as stream:
        opt = yaml.load(stream,Loader=yaml.Loader)
    return DotDict(opt)

def write_yaml(file,dotdict):
    d=dict(dotdict)
    with open(file, 'w', encoding='utf8') as outfile:
        yaml.dump(d, outfile, default_flow_style=False, allow_unicode=True)