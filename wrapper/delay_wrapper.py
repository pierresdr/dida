import numpy as np
from gym import Wrapper, spaces
import copy
import random
import warnings
from gym.spaces.dict import Dict
from gym.spaces.discrete import Discrete
from functools import reduce
from operator import mul
from wrapper.delay_processes import ConstantDelay, NormedPositiveCompoundBernoulliProcess


def add_observation_buffer(obs, info, persistence, delay_shift):
    if persistence > 1:
        return info["obs"][delay_shift - 1]
    else:
        return obs

def prod(factors):
    return reduce(mul, factors, 1)


def get_space_dim(space):
    if type(space) == Dict:
        return sum([get_space_dim(v) for k, v in space.spaces.items()])
    elif type(space) == Discrete:
        return 1
    else:
        return prod(space.shape)


def get_space_size(space):
    if type(space) == Dict:
        return sum([get_space_dim(v) for k, v in space.spaces.items()])
    elif type(space) == Discrete:
        return space.n
    else:
        return prod(space.shape)



class DelayWrapper(Wrapper):
    def __init__(
        self, env, delay=0, stochastic_delays=False, p_delay=0.70, max_delay=50
    ):
        super(DelayWrapper, self).__init__(env)

        # Delay Process initialization
        self.stochastic_delays = stochastic_delays
        if stochastic_delays:
            self.delay = NormedPositiveCompoundBernoulliProcess(
                p_delay, delay, max_value=max_delay
            )
        else:
            self.delay = ConstantDelay(delay)

        # Create State and Observation Space
        self.state_space = self.observation_space

        if isinstance(self.action_space, spaces.Discrete):
            size = self.action_space.n * self.delay.max
            stored_actions = spaces.Tuple(
                [spaces.Discrete(size) for d in range(self.delay.max)]
            )
        else:
            high = np.tile(self.action_space.high, self.delay.max)
            low = np.tile(self.action_space.high, self.delay.max)
            shape = [self.delay.max * i for i in self.action_space.shape]
            dtype = self.action_space.dtype
            stored_actions = spaces.Box(low=low, high=high, shape=shape, dtype=dtype)

        self.observation_space = spaces.Dict(
            {
                "last_obs": copy.deepcopy(self.observation_space),
                "stored_actions": stored_actions,
            }
        )

        self.action_dim = get_space_dim(self.action_space)

        # Delay Variables initialization
        self._hidden_obs = None
        self._reward_stock = None
        self.augmented_obs = None

    def reset(self, same_action=False, **kwargs):
        # Reset the underlying Environment
        obs = self.env.reset(**kwargs)

        # Reset the Delay Process
        self.delay.reset()

        # Prepare the first Augmented State by acting randomly until it is complete:
        # 1. Reset all Variables
        self._hidden_obs = [0 for _ in range(self.delay.current)]
        self.augmented_obs = [
            np.zeros(self.action_dim) for _ in range(self.delay.current + 1)
        ]
        self._reward_stock = np.array([0 for _ in range(self.delay.current)])
        # 2. The state in the first Augmented State is the reset state of the Environment
        self._hidden_obs.append(obs)
        self.augmented_obs[0] = self._hidden_obs[-1]
        # 3. Sample and Execute action to fill the rest of the Augmented State
        obs_before_start = self.delay.current
        if same_action:  # Used to intialise delay with persistence
            action = self.action_space.sample()
        while obs_before_start > 0:
            # 3a. Sample and Execute current action
            if not same_action:
                action = self.action_space.sample()
            _, _, done, info = self.step(action)
            # 3b. If the Environment happen to fail before the first Augmented State is built, notify it
            if done:
                warnings.warn("The environment failed before delay timesteps.")
                return None, None

            obs_before_start -= info["n_obs"]

        if self.delay.current == 0:
            return self.augmented_obs[0], []
        else:
            return self.augmented_obs[0], np.stack(self.augmented_obs[1:])

    def step(self, action):
        # Execute the Action in the underlying environment
        obs, reward, done, info = self.env.step(action)
        action = action if action is np.ndarray else np.array(action).reshape(-1)

        # Sample new delay
        _, n_obs = self.delay.sample()

        # Get current state
        self._hidden_obs.append(obs)

        # Update Augmented state and hidden variables
        self.augmented_obs.append(action)
        hidden_output = None
        if n_obs > 0:
            self.augmented_obs[0] = self._hidden_obs[n_obs]
            del self.augmented_obs[1 : (1 + n_obs)]
            hidden_output = np.array(self._hidden_obs[1 : (1 + n_obs)])
            del self._hidden_obs[:n_obs]

        # Update the reward array and determine current reward output
        self._reward_stock = np.append(self._reward_stock, reward)
        if done:
            reward_output = np.sum(
                self._reward_stock
            )  # -> in this case, the sum is to be done in the algorithm
        else:
            reward_output = np.sum(self._reward_stock[:n_obs])
        self._reward_stock = np.delete(self._reward_stock, range(n_obs))

        # Shaping the output
        if self.delay.current == 0:
            output = (self.augmented_obs[0], [])
        else:
            output = (self.augmented_obs[0], np.stack(self.augmented_obs[1:]))

        # Update info
        info["n_obs"] = n_obs
        info["hidden_output"] = hidden_output

        return output, reward_output, done, info


class MemorylessDelayWrapper(DelayWrapper):
    def __init__(
        self, env, delay=0, stochastic_delays=False, p_delay=0.70, max_delay=50
    ):
        super(MemorylessDelayWrapper, self).__init__(
            env,
            delay=delay,
            stochastic_delays=stochastic_delays,
            p_delay=p_delay,
            max_delay=max_delay,
        )
        # Overwrite observation space
        self.observation_space = self.observation_space["last_obs"]

    def reset(self, state=None, **kwargs):
        # Reset the underlying Environment
        obs = super(MemorylessDelayWrapper, self).reset()
        return obs[0]

    def step(self, action):
        obs, reward_output, done, info = super(MemorylessDelayWrapper, self).step(
            action
        )
        return obs[0], np.sum(reward_output), done, info


class AugmentedDelayWrapper(DelayWrapper):
    def __init__(
        self, env, delay=0, stochastic_delays=False, p_delay=0.70, max_delay=50
    ):
        super(AugmentedDelayWrapper, self).__init__(
            env,
            delay=delay,
            stochastic_delays=stochastic_delays,
            p_delay=p_delay,
            max_delay=max_delay,
        )
        # Overwrite observation space
        if not isinstance(self.action_space, spaces.Discrete):
            self.observation_space = spaces.Box(
                low=np.hstack(
                    (
                        self.observation_space["last_obs"].low,
                        self.observation_space["stored_actions"].low,
                    )
                ),
                high=np.hstack(
                    (
                        self.observation_space["last_obs"].high,
                        self.observation_space["stored_actions"].high,
                    )
                ),
                dtype=self.observation_space["last_obs"].dtype,
            )
        else:
            self.observation_space = spaces.Box(
                low=np.hstack(
                    (
                        self.observation_space["last_obs"].low,
                        [0 for sp in self.observation_space["stored_actions"]],
                    )
                ),
                high=np.hstack(
                    (
                        self.observation_space["last_obs"].high,
                        [sp.n - 1 for sp in self.observation_space["stored_actions"]],
                    )
                ),
                dtype=self.observation_space["last_obs"].dtype,
            )

    def reset(self, state=None, **kwargs):
        # Reset the underlying Environment
        obs = super(AugmentedDelayWrapper, self).reset()
        return np.hstack((obs[0], obs[1].reshape(-1)))

    def step(self, action):
        obs, reward_output, done, info = super(AugmentedDelayWrapper, self).step(action)
        obs = np.hstack((obs[0], obs[1].reshape(-1)))
        return obs, np.sum(reward_output), done, info


class AugmentedDictDelayWrapper(DelayWrapper):
    """The step function returns a dict to fit the stablebaselines3 framework."""

    def __init__(
        self, env, delay=0, stochastic_delays=False, p_delay=0.70, max_delay=50
    ):
        super(AugmentedDictDelayWrapper, self).__init__(
            env,
            delay=delay,
            stochastic_delays=stochastic_delays,
            p_delay=p_delay,
            max_delay=max_delay,
        )
        # Overwrite observation space
        if not isinstance(self.action_space, spaces.Discrete):
            self.observation_space = spaces.Box(
                low=np.hstack(
                    (
                        self.observation_space["last_obs"].low,
                        self.observation_space["stored_actions"].low,
                    )
                ),
                high=np.hstack(
                    (
                        self.observation_space["last_obs"].high,
                        self.observation_space["stored_actions"].high,
                    )
                ),
                dtype=self.observation_space["last_obs"].dtype,
            )
        else:
            # stored_actions_space = dict(
            #     zip(
            #         np.arange(len(self.observation_space["stored_actions"])),
            #         self.observation_space["stored_actions"],
            #     )
            # )
            space_dict = {
                0: copy.deepcopy(self.observation_space["last_obs"]),
            }
            space_dict.update(
                dict(
                    zip(
                        np.arange(1, len(self.observation_space["stored_actions"]) + 1),
                        self.observation_space["stored_actions"],
                    )
                )
            )
            self.observation_space = spaces.Dict(space_dict)

    def reset(self, state=None, **kwargs):
        # Reset the underlying Environment
        obs = super(AugmentedDictDelayWrapper, self).reset()
        return {"obs_0": obs[0], "obs_1": obs[1]}

    def step(self, action):
        obs, reward_output, done, info = super(AugmentedDictDelayWrapper, self).step(
            action
        )
        return {"obs_0": obs[0], "obs_1": obs[1]}, np.sum(reward_output), done, info


class PersistenceDelayWrapper(Wrapper):
    """Implements delay for a persistent agent. Allows non-integer delays."""

    def __init__(
        self,
        env,
        delay=0,
        persistence=1,
        stochastic_delays=False,
        p_delay=0.70,
        max_delay=50,
    ):
        super(PersistenceDelayWrapper, self).__init__(env)

        # Delay Process initialization
        self.stochastic_delays = stochastic_delays
        if stochastic_delays:
            self.delay = NormedPositiveCompoundBernoulliProcess(
                p_delay, delay, max_value=max_delay
            )
        else:
            self.delay = ConstantDelay(delay)

        # Create State and Observation Space
        self.state_space = self.observation_space

        # Compute number of necessary actions in buffer, the action buffer is handled differently than reward and state buffers
        action_buffer_len = int(np.ceil(self.delay.max / persistence))

        if isinstance(self.action_space, spaces.Discrete):
            size = self.action_space.n * action_buffer_len
            stored_actions = spaces.Tuple(
                [spaces.Discrete(size) for d in range(action_buffer_len)]
            )
        else:
            high = np.tile(self.action_space.high, action_buffer_len)
            low = np.tile(self.action_space.high, action_buffer_len)
            shape = [action_buffer_len * i for i in self.action_space.shape]
            dtype = self.action_space.dtype
            stored_actions = spaces.Box(low=low, high=high, shape=shape, dtype=dtype)

        self.observation_space = spaces.Dict(
            {
                "last_obs": copy.deepcopy(self.observation_space),
                "stored_actions": stored_actions,
            }
        )

        self.action_dim = get_space_dim(self.action_space)

        # Delay Variables initialization
        self._hidden_obs = None
        self._reward_stock = None
        self.augmented_obs = None

        # Persistence
        self.persistence = persistence

    def reset(self, **kwargs):
        # Reset the underlying Environment
        obs = self.env.reset(**kwargs)

        # Reset the Delay Process
        self.delay.reset()

        # Prepare the first Augmented State by acting randomly until it is complete:
        # 1. Reset all Variables
        self._hidden_obs = [0 for _ in range(self.delay.current)]

        # Compute number of necessary actions in buffer, the action buffer is handled differently than reward and state buffers
        action_buffer_len = int(np.ceil(self.delay.current / self.persistence))

        self.augmented_obs = [
            np.zeros(self.action_dim) for _ in range(action_buffer_len + 1)
        ]
        self._reward_stock = [0 for _ in range(self.delay.current)]
        # 2. The state in the first Augmented State is the reset state of the Environment
        self._hidden_obs.append(obs)
        self.augmented_obs[0] = self._hidden_obs[-1]
        # 3. Sample and Execute action to fill the rest of the Augmented State
        obs_before_start = self.delay.current
        while obs_before_start > 0:
            # 3a. Sample and Execute current action
            action = self.action_space.sample()
            _, _, done, info = self.step(action)
            # 3b. If the Environment happen to fail before the first Augmented State is built, notify it
            if done:
                warnings.warn("The environment failed before delay timesteps.")
                return None, None

            obs_before_start -= info["n_obs"]

        if self.delay.current == 0:
            return self.augmented_obs[0], []
        else:
            return self.augmented_obs[0], np.stack(self.augmented_obs[1:])

    def step(self, action):
        # Execute the action in the underlying environment with persistence
        for p in range(self.persistence):
            obs, reward, done, info = self.env.step(action)

            # Get current state
            self._hidden_obs.append(obs)

            # Update the reward array and determine current reward output
            self._reward_stock.append(reward)

        action = action if action is np.ndarray else np.array(action).reshape(-1)

        # Sample new delay
        _, n_obs = self.delay.sample()
        n_obs = self.persistence + n_obs - 1  # adapt number of observations with delay

        # Update Augmented state and hidden variables
        self.augmented_obs.append(action)
        hidden_output = None
        if n_obs > 0:
            self.augmented_obs[0] = self._hidden_obs[n_obs]
            # Compute number of necessary actions in buffer
            action_buffer_len = int(np.ceil(self.delay.current / self.persistence))
            del self.augmented_obs[1:-action_buffer_len]
            hidden_output = np.array(self._hidden_obs[1 : (1 + n_obs)])
            del self._hidden_obs[:n_obs]

        if done:
            reward_output = (
                self._reward_stock
            )  # -> in this case, the sum is to be done in the algorithm
        else:
            reward_output = self._reward_stock[:n_obs]
        del self._reward_stock[:n_obs]

        # Shaping the output
        if self.delay.current == 0:
            output = (self.augmented_obs[0], [])
        else:
            output = (self.augmented_obs[0], np.stack(self.augmented_obs[1:]))

        # Update info
        info["n_obs"] = n_obs
        info["hidden_output"] = hidden_output

        return output, reward_output, done, info


class PersistenceWrapper(Wrapper):
    def __init__(
        self,
        env,
        persistence=1,
    ):
        super(PersistenceWrapper, self).__init__(env)

        # Persistence
        self.persistence = persistence

    def reset(self, **kwargs):
        # Reset the underlying Environment
        obs = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        # Execute the Action in the underlying environment
        reward = 0
        info = {"obs": []}
        for p in range(self.persistence):
            obs, r, done, i = self.env.step(action)
            reward += r
            info["obs"].append(obs)
            if done:
                break

        for k, v in i.items():
            info[k] = v
        return obs, reward, done, info

    def step_no_persistence(self, action):
        """Useful to initialize process like the delayed one.

        Args:
            action (_type_): _description_

        Returns:
            _type_: _description_
        """
        return self.env.step(action)

    def step_single(self, action):
        return self.env.step(action)


def remove_right_zero(lambdas):
    if lambdas[-1] == 0:
        return remove_right_zero(lambdas[:-1])
    else:
        return lambdas


class MTDDelayWrapper(Wrapper):
    """Implementation of a MTD-MDP."""

    def __init__(
        self,
        env,
        lambdas=None,
        state_action_model=False,
    ):
        """Lambda is a vector containing weights of the MTDg transition model."""
        super(MTDDelayWrapper, self).__init__(env)

        assert (
            round(sum(lambdas), 3) == 1
        ), "The vector lambdas should define a probability."

        # Delay Process initialization
        self.lambdas = remove_right_zero(lambdas)
        self.order = len(self.lambdas)
        self.max_delay = self.order - 1
        self.state_action_model = state_action_model

        # Create State and Observation Space
        self.state_space = self.observation_space

        if isinstance(self.action_space, spaces.Discrete):
            size = self.action_space.n * self.max_delay
            stored_actions = spaces.Tuple(
                [spaces.Discrete(size) for d in range(self.max_delay)]
            )
        else:
            high = np.tile(self.action_space.high, self.max_delay)
            low = np.tile(self.action_space.high, self.max_delay)
            shape = [self.max_delay * i for i in self.action_space.shape]
            dtype = self.action_space.dtype
            stored_actions = spaces.Box(low=low, high=high, shape=shape, dtype=dtype)

        self.observation_space = spaces.Dict(
            {
                "last_obs": copy.deepcopy(self.observation_space),
                "stored_actions": stored_actions,
            }
        )

        self.action_dim = get_space_dim(self.action_space)

        # Delay Variables initialization
        self._reward_buffer = None
        self._action_buffer = None
        self._state_buffer = None
        self._obs_buffer = None

    def reset(self, **kwargs):
        # Reset the underlying Environment
        obs = self.env.reset(**kwargs)

        # Initialize buffers
        self._action_buffer = [
            self.action_space.sample() for i in range(self.max_delay)
        ]
        self._state_buffer = [
            copy.deepcopy(self.env.state) for i in range(self.max_delay + 1)
        ]
        self._obs_buffer = [obs for i in range(self.max_delay + 1)]

        # Sample the first self.max_delay transitions
        for i in range(self.max_delay):
            action = self.action_space.sample()
            _, _, done, info = self.step(action)
            # If the environment fails during first augmented state initialization
            if done:
                warnings.warn("The environment failed before delay timesteps.")
                return None, None

        if self.state_action_model:
            return (self._obs_buffer, self._action_buffer)
        else:
            return (self._obs_buffer[-1], self._action_buffer)

    def step(
        self,
        action,
    ):

        # Update action buffer
        self._action_buffer.append(action)

        # Select an action from the buffer and apply in the underlying env
        idx = random.choices(np.arange(self.order - 1, -1, -1), weights=self.lambdas)[0]
        if self.state_action_model:
            self.env.state = self._state_buffer[idx]
        obs, reward, done, info = self.env.step(self._action_buffer[idx])

        # Update other buffers
        self._state_buffer.append(copy.deepcopy(self.env.state))
        self._obs_buffer.append(obs)

        # Delete old information
        self._state_buffer.pop(0)
        self._obs_buffer.pop(0)
        self._action_buffer.pop(0)

        # Creating observation output
        if self.state_action_model:
            obs_output = (self._obs_buffer, self._action_buffer)
        else:
            obs_output = (self._obs_buffer[-1], self._action_buffer)

        return obs_output, reward, done, info


class AugmentedMTDDelayWrapper(MTDDelayWrapper):
    def __init__(
        self,
        env,
        lambdas=None,
        state_action_model=False,
    ):
        super(AugmentedMTDDelayWrapper, self).__init__(
            env,
            lambdas=lambdas,
            state_action_model=state_action_model,
        )
        # Overwrite observation space
        if not isinstance(self.action_space, spaces.Discrete):
            self.observation_space = spaces.Box(
                low=np.hstack(
                    (
                        self.observation_space["last_obs"].low,
                        self.observation_space["stored_actions"].low,
                    )
                ),
                high=np.hstack(
                    (
                        self.observation_space["last_obs"].high,
                        self.observation_space["stored_actions"].high,
                    )
                ),
                dtype=self.observation_space["last_obs"].dtype,
            )
        else:
            self.observation_space = spaces.Box(
                low=np.hstack(
                    (
                        self.observation_space["last_obs"].low,
                        [0 for sp in self.observation_space["stored_actions"]],
                    )
                ),
                high=np.hstack(
                    (
                        self.observation_space["last_obs"].high,
                        [sp.n - 1 for sp in self.observation_space["stored_actions"]],
                    )
                ),
                dtype=self.observation_space["last_obs"].dtype,
            )

    def reset(self, state=None, **kwargs):
        # Reset the underlying Environment
        obs = super(AugmentedMTDDelayWrapper, self).reset()
        return np.hstack([obs[0]] + obs[1])

    def step(self, action):
        obs, reward_output, done, info = super(AugmentedMTDDelayWrapper, self).step(
            action
        )
        obs = np.hstack([obs[0]] + obs[1])
        return obs, np.sum(reward_output), done, info
