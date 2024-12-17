

class DelayProcess:
    def __init__(self, init, max_value=50):
        self.current = init
        self.init = init
        self.max = max_value

    def sample(self):
        raise NotImplementedError

    def reset(self):
        self.current = self.init



class ConstantDelay(DelayProcess):
    def __init__(self, init):
        super().__init__(init, max_value=init)

    def sample(self):
        return self.current, 1



class NormedPositiveCompoundBernoulliProcess(DelayProcess):
    def __init__(self, p, init=1, max_value=50):
        """
        A compound Bernoulli process which is forced positive by normalizing
        probabilities. The process is built as follows:
            Y_t = sum_{i=0}^t (Z_i)
        where (Z_i) are such that:
            p( Z_t=a | Y_t=b ) = 1/C . (1-p)p^(1-a), for a \in [-b,1]
                                it is 0 otherwise.

        Arguments:
            p (float): the probability of a downard jump of size one
                    before normalization
            init (int): the initial value of the series
        """
        super().__init__(init, max_value)
        self.p = p
        self.series = [init]

    def sample(self):
        C = 1 - self.p ** (self.current + 1)
        u = rnd.random()
        jump = 1 - int(np.log(1 - C * u) / np.log(self.p))
        self.current = self.current + jump

        if self.current > self.max:
            self.current = self.max
            n_obs = 1
        else:
            n_obs = 1 - jump
        self.series.append(self.current)
        return self.current, n_obs


class PositiveCompoundBernoulliProcess(DelayProcess):
    def __init__(self, p, init=1, max_value=50):
        """
        A compound Bernoulli process which is forced positive by truncation of the
        negative probabilities. The process is built as follows:
            Y_t = sum_{i=0}^t (Z_i)
        where (Z_i) are such that:
            p( Z_t=a | Y_t=b ) = (1-p)p^(1-a), for a \in [-b+1,1]
                                (1-p)p^(b+1) + p^(b+2) for a=-b

        Arguments:
            p (float): the probability of a downard jump of size one
                    before normalization
            init (int): the initial value of the series
        """
        super().__init__(init, max_value)
        self.p = p
        self.series = [init]

    def sample(self):
        u = rnd.random()
        jump = 1 - int(np.log(1 - u) / np.log(self.p))
        jump = max(jump, -self.current + 1)
        self.current = min(self.current + jump, self.max)

        if self.current > self.max - 1:
            self.current = self.max - 1
            n_obs = 1
        else:
            n_obs = 1 - jump
        self.series.append(self.current)
        return self.current, n_obs
