import torch
import numpy as np
from gym.spaces.dict import Dict 
from gym.spaces.discrete import Discrete
import operator
from functools import reduce
from contextlib import contextmanager
import logging, time

@contextmanager
def timed(msg):
    logging.info(msg)
    tstart = time.time()
    yield
    logging.info(msg + " done in %.3f seconds"%(time.time() - tstart))

def get_space_dim(space):
    if type(space) == Dict:
        return sum([get_space_dim(v) for k, v in space.spaces.items()])
    elif type(space) == Discrete:
        return space.n
    else:
        return prod(space.shape)

def prod(factors):
    return reduce(operator.mul, factors, 1)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prRed(prt): print("\033[91m {}\033[00m" .format(prt))
def prGreen(prt): print("\033[92m {}\033[00m" .format(prt))
def prYellow(prt): print("\033[93m {}\033[00m" .format(prt))
def prLightPurple(prt): print("\033[94m {}\033[00m" .format(prt))
def prPurple(prt): print("\033[95m {}\033[00m" .format(prt))
def prCyan(prt): print("\033[96m {}\033[00m" .format(prt))
def prLightGray(prt): print("\033[97m {}\033[00m" .format(prt))
def prBlack(prt): print("\033[98m {}\033[00m" .format(prt))

def index_sampler(mask, batch_size):
    """
    Creates batch of data from the index of mask.
    """
    index = np.where(mask)[0]
    np.random.shuffle(index)
    n = len(index)//batch_size
    sampler_size = n
    sampler = [index[i*batch_size:(i+1)*batch_size] for i in range(n)]
    if len(index)%batch_size != 0:
        sampler.append(index[n*batch_size:])
        sampler_size += 1
    return (s for s in sampler), sampler_size

def sampler(data, batch_size):
    np.random.shuffle(data)
    n = len(data)//batch_size
    sampler_size = n
    sampler = [data[i*batch_size:(i+1)*batch_size] for i in range(n)]
    if len(data)%batch_size != 0:
        sampler.append(data[n*batch_size:])
        sampler_size += 1
    return (s for s in sampler), sampler_size