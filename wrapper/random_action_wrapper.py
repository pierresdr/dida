import numpy as np
import random as rnd
from gym import ActionWrapper



class Gaussian:
    def __init__(self, std=1.0):
        self.std = std
    
    def sample(self, action):
        return action + np.random.normal(scale=self.std)


class Uniform:
    def __init__(self, env_action_space, epsilon=0.1):
        self.epsilon = epsilon
        self.env_action_space = env_action_space
    
    def sample(self, action):
        if rnd.random() < self.epsilon:
            return self.env_action_space.sample()

        return action


class LogNormal:
    def __init__(self, mean=0.0, sigma=1.0, shift=-1.0):
        self.mean = mean
        self.sigma = sigma
        self.shift = shift

    def sample(self, action):
        return action + np.random.lognormal(mean=self.mean, sigma=self.sigma) - self.shift


class Triangular:
    def __init__(self, left=-2.0, mode=0.0, right=2.0):
        self.left = left
        self.mode = mode
        self.right = right

    def sample(self, action):
        return action + np.random.triangular(left=self.left, mode=self.mode, right=self.right)


class Beta:
    def __init__(self, alpha=0.5, beta=0.5, shift=-0.5, scale=2.0):
        self.alpha = alpha
        self.beta = beta
        self.shift = shift
        self.scale = scale

    def sample(self, action):
        return action + (np.random.beta(a=self.alpha, b=self.beta) - self.shift) * self.scale


class StochActionWrapper(ActionWrapper):
    def __init__(self, env, distrib='Gaussian', param=0.1, seed=0):
        super(StochActionWrapper, self).__init__(env)
        rnd.seed(seed)
        if distrib == 'Gaussian':
            self.stoch_perturbation = Gaussian(std=param)
        elif distrib == 'Uniform':
            self.stoch_perturbation = Uniform(epsilon=param, env_action_space=self.env.action_space)
        elif distrib == 'LogNormal':
            self.stoch_perturbation = LogNormal(sigma=param)
        elif distrib == 'Triangular':
            self.stoch_perturbation = Triangular(mode=param)
        elif distrib == 'Quadratic':
            assert param > 1
            self.stoch_perturbation = Beta(alpha=param, beta=param)
        elif distrib == 'U-Shaped':
            assert 0 < param < 1
            self.stoch_perturbation = Beta(alpha=param, beta=param)
        elif distrib == 'Beta':
            self.stoch_perturbation = Beta(alpha=8, beta=2, shift=-0.5, scale=2.0)

    def action(self, action):
        action = self.stoch_perturbation.sample(action)
        return action


class RandomActionWrapper(ActionWrapper):

    def __init__(self, env, epsilon=0.1):
        super(RandomActionWrapper, self).__init__(env)
        self.epsilon = epsilon

    def action(self, action):

        if rnd.random() < self.epsilon:

            print("Random!")

            return self.env.action_space.sample()

        return action