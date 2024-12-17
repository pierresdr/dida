import gym, fire, os, copy, datetime, json, logging
import numpy as np
from policy.neural_networks import mlp, cnn, AutoregNet, get_number_of_parameters
from utils.util import get_space_dim
from wrapper.delay_wrapper import (
    DelayWrapper,
    PersistenceWrapper,
    add_observation_buffer,
)
import torch
from utils.save_results import save_training
from utils.plot_results import plots_std
from utils.stochastic_wrapper import StochActionWrapper
from tqdm import tqdm
import time
from policy.dagger import imitation_learning, get_beta_routine_weights
from policy.experts import load_expert_policy
from policy.policy import get_policy_dida


# Logging level
logging.basicConfig(
    level="INFO", format="%(asctime)s: %(message)s", datefmt="%m/%d/%Y %H:%M:%S"
)


def get_dida_policy(beta, delayed_policy, expert_policy):
    def dida_policy(obs, delayed_obs, actions):
        expert_action = expert_policy.predict(obs)[0]
        if np.random.uniform() < beta:
            return expert_action, expert_action
        else:
            action = delayed_policy(delayed_obs.unsqueeze(0), actions.unsqueeze(0))
            if action.ndim > 2:
                action = action[:, -1, :]
            return action.reshape(-1).detach().numpy(), expert_action

    return dida_policy


def initialise_env(env, persistence, stoch_mdp_distrib, stoch_mdp_param, **env_kwargs):
    env = gym.make(env, **env_kwargs)
    if persistence > 1:
        env = PersistenceWrapper(copy.deepcopy(env), persistence=persistence)

    # Stochastic mdp
    if stoch_mdp_distrib is not None:
        env = StochActionWrapper(env, distrib=stoch_mdp_distrib, param=stoch_mdp_param)
    return env


def initialise_delayed_env(env, persistence, delay, **env_kwargs):
    env = gym.make(env, **env_kwargs)

    # Add delay to env
    if persistence > 1:
        denv = DelayWrapper(copy.deepcopy(env), delay=delay)
        return PersistenceWrapper(
            copy.deepcopy(denv),
            persistence=persistence,
        )
    else:
        return DelayWrapper(copy.deepcopy(env), delay=delay)


def train_dida(
    env,
    delay=7,
    persistence=1,
    algo_expert="sac",
    training_rounds=4,
    n_steps=1000,
    traj_len=250,
    n_neurons=[
        100,
        100,
    ],
    learning_rate=1e-3,
    optimizer="RMSprop",
    batch_size=32,
    beta_routine=None,
    noise_routine=None,
    random_action_routine=None,
    save_path="test",
    seed=0,
    test_steps=1000,
    expert_sample=False,
    n_channels=[4, 4],
    kernel_size=3,
    policy="mlp",
    keep_dataset=True,
    max_buffer_size=None,
    all_actions=False,
    stoch_mdp_distrib=None,
    stoch_mdp_param=None,
    save_every=1,
    **env_kwargs
):
    """
    _summary_

    Args:
        env (_type_): _description_
        delay (int, optional): the amount of delay for constant delay. 
        persistence (int, optional): See https://proceedings.mlr.press/v119/metelli20a.html. 
        algo_expert (str, optional): The expert algo to be used in DIDA.
        training_rounds (int, optional):Number of imitation training rounds.
        n_steps (int, optional): Number of environment step in each round.
        traj_len (int, optional): Length of the trajectories.
        n_neurons (list, optional): Number of neurons in the MLP. Defaults to [ 100, 100, ].
        learning_rate (_type_, optional): 
        optimizer (str, optional): Defaults to "RMSprop".
        batch_size (int, optional):
        beta_routine (_type_, optional): beta routine for Dagger.
        noise_routine (_type_, optional): Noise routine for the expert action.
        random_action_routine (_type_, optional): _description_. Defaults to None.
        save_path (str, optional): Path where the experiments are saved. Defaults to "test".
        seed (int, optional): Defaults to 0.
        test_steps (int, optional): Defaults to 1000.
        expert_sample (bool, optional): Use expert actions or DIDA's actions in augmented state.
        n_channels (list, optional): For CNN policy.
        kernel_size (int, optional): For CNN policy.
        policy (str, optional): Type of policy. Defaults to "mlp".
        keep_dataset (bool, optional): Drop of keep dataset between each
        round of imitation learning. Defaults to True.
        max_buffer_size (_type_, optional): Defaults to None.
        all_actions (bool, optional):  Defaults to False.
        stoch_mdp_distrib (_type_, optional): Adding noise to the action
        in a deterministic MDP. Defaults to None.
        stoch_mdp_param (_type_, optional):  Defaults to None.
        save_every (int, optional): _description_. Defaults to 1.
    """

    params = {
        "env": env,
        "delay": delay,
        "persistence": persistence,
        "algo_expert": algo_expert,
        "training_rounds": training_rounds,
        "n_steps": n_steps,
        "traj_len": traj_len,
        "n_neurons": n_neurons,
        "learning_rate": learning_rate,
        "optimizer": optimizer,
        "batch_size": batch_size,
        "beta_routine": beta_routine,
        "noise_routine": noise_routine,
        "random_action_routine": random_action_routine,
        "save_path": save_path,
        "seed": seed,
        "expert_sample": expert_sample,
        "n_channels": n_channels,
        "kernel_size": kernel_size,
        "policy": policy,
        "keep_dataset": keep_dataset,
        "max_buffer_size": max_buffer_size,
        "all_actions": all_actions,
        "stoch_mdp_distrib": stoch_mdp_distrib,
        "stoch_mdp_param": stoch_mdp_param,
    }

    # Create save folder
    head, tail = os.path.split(save_path)
    tail = datetime.datetime.now().strftime("%y_%m_%d-%H_%M_") + tail
    save_path = os.path.join(head, tail)
    os.makedirs(save_path)
    with open(os.path.join(save_path, "parameters.json"), "w") as f:
        json.dump(params, f)

    logging.info("\n Launching DIDA with: \t{}.".format(params))

    start_time = time.time()
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Compute number of necessary actions in buffer, the action buffer is handled differently than reward and state buffers
    action_buffer_len = int(np.ceil(delay / persistence))
    delay_shift = persistence - (delay) % persistence

    # Initialise envs
    undelayed_env = initialise_env(
        env, persistence, stoch_mdp_distrib, stoch_mdp_param, **env_kwargs
    )
    denv = initialise_delayed_env(env, persistence, delay)

    # Select delays for the test
    if all_actions:
        if delay < 5:
            test_delays = np.arange(delay).astype(int)
        else:
            test_delays = np.linspace(1, delay, 5, dtype=int)
    else:
        test_delays = [delay]

    expert_policy = load_expert_policy(
        undelayed_env, persistence, stoch_mdp_distrib, stoch_mdp_param
    )

    # Get states dimensions
    state_dim = get_space_dim(undelayed_env.observation_space)
    action_dim = get_space_dim(undelayed_env.action_space)
    aug_state_dim = state_dim + action_dim * action_buffer_len

    # Get DIDA policy
    delayed_policy = get_policy_dida(
        policy, delay, action_dim, state_dim, aug_state_dim, kernel_size, all_actions
    )
    delayed_policy.eval()

    n_params = get_number_of_parameters(delayed_policy)
    logging.info("\n Delayed policy parameters: \t{}.".format(n_params))
    with open(os.path.join(save_path, "number_of_parameters.json"), "w") as f:
        json.dump({"n_params": n_params}, f)

    beta_routine_weights, memoryless_sampling = get_beta_routine_weights(beta_routine, training_rounds)

    # Define random action selection routine
    random_action_routine = (
        [0] * training_rounds
        if random_action_routine is None
        else random_action_routine
    )

    # Define random action selection routine
    noise_routine = [0] * training_rounds if noise_routine is None else noise_routine

    # Buffers
    non_integer_delay = delay % persistence != 0
    if keep_dataset:
        if max_buffer_size is not None:
            rounds_buffer = (
                max_buffer_size  # the maximum buffer size is given in terms of rounds
            )
        else:
            max_buffer_size = training_rounds
            rounds_buffer = training_rounds
    else:
        rounds_buffer = 1
    state_buffer = np.zeros((rounds_buffer * n_steps + 1, state_dim), dtype=float)
    mask_buffer = np.zeros(rounds_buffer * n_steps + non_integer_delay, dtype=bool)
    action_buffer = np.zeros(
        (rounds_buffer * n_steps + non_integer_delay, action_dim), dtype=float
    )
    action_buffer_del = np.zeros(
        (rounds_buffer * n_steps + non_integer_delay, action_dim), dtype=float
    )

    # Training statistics
    returns = np.zeros(training_rounds // save_every + 1)
    traj_returns = np.zeros(training_rounds // save_every + 1)
    test_returns = np.zeros((training_rounds // save_every + 1, len(test_delays)))
    test_traj_returns = np.zeros((training_rounds // save_every + 1, len(test_delays)))
    failed_reset = np.zeros((training_rounds // save_every + 1, len(test_delays)))
    losses = np.zeros(training_rounds // save_every + 1)

    # Initialise optimizer
    if optimizer == "RMSprop":
        set_optimizer = "RMSprop(delayed_policy.parameters(),  lr={},  alpha=0.9, eps=1e-10)".format(
            learning_rate
        )
    elif optimizer == "Adam":
        set_optimizer = "Adam(delayed_policy.parameters(),  lr={},)".format(
            learning_rate
        )
    else:
        raise ValueError
    optimizer = eval(set_optimizer)

    # Training rounds
    for round in range(training_rounds):
        policy = get_dida_policy(beta_routine_weights[round], delayed_policy, expert_policy)

        t = 0
        ret = 0
        done = False
        traj_t = 0  # The first step of the current trajectory
        traj_ret = []
        logging.info(
            "\n DIDA training round {} with beta {}.".format(round, beta_routine_weights[round])
        )
        pbar = tqdm(total=n_steps)
        if keep_dataset:
            buffer_shift = (round % max_buffer_size) * n_steps
        else:
            buffer_shift = 0
            mask_buffer = np.zeros(n_steps, dtype=bool)
        while t < n_steps:
            # if the env is reinitialized, sample the first delay actions without following the policy
            if t + 1 >= n_steps:
                t = n_steps
                pbar.update(1)
            elif (t - traj_t) % traj_len == 0 or done:
                if t + action_buffer_len >= n_steps:
                    t = n_steps
                    pbar.update(n_steps - t)
                else:
                    obs = undelayed_env.reset()
                    done = False
                    traj_t = t
                    traj_ret.append(0)
                    state_buffer[buffer_shift + t] = obs

                    # Initialise the undelayed env for the first steps if persistence
                    if non_integer_delay:
                        action = undelayed_env.action_space.sample()
                        for h in range(delay % persistence):
                            obs, reward, done, info = undelayed_env.step_single(action)
                        action_buffer[buffer_shift + t] = action
                        action_buffer_del[buffer_shift + t] = action

                    # Initialise the undelayed env so that it matches
                    # the delayed one's initial distribution
                    for h in range(delay // persistence):
                        action = undelayed_env.action_space.sample()
                        obs, reward, done, info = undelayed_env.step(action)
                        action_buffer[buffer_shift + t + h + non_integer_delay] = action
                        action_buffer_del[buffer_shift + t + h + non_integer_delay] = (
                            action
                        )
                        state_buffer[buffer_shift + t + h + 1] = add_observation_buffer(
                            obs=obs,
                            info=info,
                            persistence=persistence,
                            delay_shift=delay_shift,
                        )
                        ret += reward
                    mask_buffer[buffer_shift + t] = True
                    t += action_buffer_len
                    pbar.update(action_buffer_len)
            else:
                if np.random.uniform() < random_action_routine[round]:
                    action = undelayed_env.action_space.sample()
                    expert_action = expert_policy.predict(obs)[0]
                else:
                    del_s = torch.from_numpy(
                        state_buffer[buffer_shift + t - action_buffer_len]
                    ).float()

                    # Memoryless sampling to avoid any interaction with undelayed environment
                    if round == 0 and memoryless_sampling:
                        obs = torch.from_numpy(
                            state_buffer[buffer_shift + t - action_buffer_len]
                        ).float()
                    else:
                        theta, thetadot = undelayed_env.state
                        s = torch.from_numpy(
                            np.array(
                                [np.cos(theta), np.sin(theta), thetadot],
                                dtype=np.float32,
                            )
                        ).float()
                        obs = torch.from_numpy(obs).float()

                    # Use expert actions or DIDA's actions in augmented state
                    if expert_sample:
                        e = (
                            torch.from_numpy(
                                np.hstack(
                                    (
                                        action_buffer[
                                            buffer_shift
                                            + t
                                            - action_buffer_len : buffer_shift
                                            + t
                                        ]
                                    )
                                )
                            )
                            .float()
                            .reshape(-1, action_dim)
                        )
                    else:
                        e = (
                            torch.from_numpy(
                                np.hstack(
                                    (
                                        action_buffer_del[
                                            buffer_shift
                                            + t
                                            - action_buffer_len : buffer_shift
                                            + t
                                        ]
                                    )
                                )
                            )
                            .float()
                            .reshape(-1, action_dim)
                        )

                    # Select action under beta sampling scheme
                    if not torch.equal(obs, s):
                        print("pbm")
                    action, expert_action = policy(obs, del_s, e)

                # Step and buffer update
                obs, reward, done, info = undelayed_env.step(action)
                ret += reward
                traj_ret[-1] += reward
                state_buffer[buffer_shift + t + 1] = add_observation_buffer(
                    obs=obs, info=info, persistence=persistence, delay_shift=delay_shift
                )
                action_buffer[buffer_shift + t + non_integer_delay] = (
                    expert_action + np.random.normal(0, noise_routine[round])
                )
                action_buffer_del[buffer_shift + t + non_integer_delay] = action
                t += 1
                pbar.update(1)
                mask_buffer[buffer_shift + t - action_buffer_len] = True

        pbar.close()
        logging.info(
            "\n DIDA sampling round {} DONE with mean reward {}.".format(
                round, ret / n_steps
            )
        )

        loss, optimizer = imitation_learning(
            action_buffer_len,
            delayed_policy,
            state_buffer,
            action_buffer,
            action_buffer_del,
            mask_buffer,
            batch_size=batch_size,
            optimizer=optimizer,
            all_actions=all_actions,
        )
        logging.info(
            "\n DIDA training round {} DONE with mean reward {}.".format(round, loss)
        )
        if round % save_every == 0:
            returns[round // save_every] = ret / n_steps
            losses[round // save_every] = loss
            (
                test_returns[round // save_every],
                test_traj_returns[round // save_every],
                failed_reset[round // save_every],
            ) = test_policy(
                delayed_policy,
                env,
                test_delays,
                test_steps,
                traj_len,
                persistence=persistence,
            )
            traj_returns[round // save_every] = np.mean(traj_ret)

    save_training(
        delayed_policy,
        test_delays,
        returns,
        test_returns,
        losses,
        traj_ret,
        test_traj_returns,
        failed_reset,
        seed=seed,
        save_path=save_path,
    )
    logging.info("Training time {} seconds.".format((time.time() - start_time)))
    return save_path


def test_policy(
    policy,
    env,
    delays,
    steps,
    traj_len,
    persistence=1,
):
    mean_rew = np.zeros(len(delays))
    traj_ret = np.zeros(len(delays))
    failed_reset = np.zeros(len(delays))
    for d_i, d in enumerate(delays):
        denv = initialise_delayed_env(env=env, persistence=persistence, delay=d)
        action_buffer_len = int(np.ceil(d / persistence))
        delay_shift = persistence - (d) % persistence

        done = False
        traj_t = 0  # The first step of the current trajectory
        traj_r = []
        for t in range(steps):
            # if the env is reinitialized, sample the first delay actions without following the policy
            if (t - traj_t) % traj_len == 0 or done:
                obs = denv.reset(same_action=True)
                if obs[0] is None:
                    done = True
                    failed_reset[d_i] += 1
                else:
                    done = False
                traj_t = t
                traj_r.append(0)

            if not done:
                state = torch.from_numpy(obs[0]).float().unsqueeze(0)
                actions = obs[1][persistence * np.arange(action_buffer_len)]
                actions = torch.from_numpy(actions).float().unsqueeze(0)
                action = policy(state, actions)
                if action.ndim > 2:
                    action = action[:, -1, :]
                action = action.reshape(-1).detach().numpy()

                obs, reward, done, info = denv.step(action)
                traj_r[-1] += reward
        traj_ret[d_i] = np.mean(traj_r)
        mean_rew[d_i] = np.sum(traj_r) / steps

    return mean_rew, traj_ret, failed_reset


def train_dida_seeds(
    env,
    delay=7,
    persistence=1,
    training_rounds=4,
    n_steps=1000,
    traj_len=250,
    gamma=1,
    n_neurons=[
        100,
        100,
    ],
    learning_rate=1e-3,
    optimizer="RMSprop",
    batch_size=32,
    beta_routine=None,
    noise_routine=None,
    random_action_routine=None,
    save_path="test",
    seeds=[0, 1, 2],
    expert_sample=False,
    max_buffer_size=None,
    save_every=1,
    **env_kwargs
):
    for s in seeds:
        logging.info("\n Training with seed {}.".format(s))
        save_path = train_dida(
            env,
            delay=delay,
            persistence=persistence,
            training_rounds=training_rounds,
            n_steps=n_steps,
            traj_len=traj_len,
            gamma=gamma,
            n_neurons=n_neurons,
            learning_rate=learning_rate,
            optimizer=optimizer,
            batch_size=batch_size,
            beta_routine=beta_routine,
            noise_routine=noise_routine,
            random_action_routine=random_action_routine,
            save_path=save_path,
            seed=s,
            expert_sample=expert_sample,
            max_buffer_size=max_buffer_size,
            save_every=save_every,
            **env_kwargs
        )

    r = []
    l = []
    for s in seeds:
        with open(os.path.join(save_path, "{}_{}.txt".format("returns", s)), "rb") as f:
            r.append(np.load(f))
        with open(os.path.join(save_path, "{}_{}.txt".format("losses", s)), "rb") as f:
            l.append(np.load(f))
    r = np.stack(r)
    l = np.stack(l)

    series = {
        "losses": [{"mean": l.mean(0), "std": l.std(0)}],
        "returns": [{"mean": r.mean(0), "std": r.std(0)}],
    }
    plots_std(
        series,
        save_path=os.path.join(save_path, "training_mean"),
        title="Imitation loss and return over {} seeds".format(len(seeds)),
    )


if __name__ == "__main__":
    fire.Fire()
