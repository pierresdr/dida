import gym, fire, os, copy, json, logging
import numpy as np
from policy.neural_networks import mlp, cnn
from utils.util import get_space_dim
from wrapper.delay_wrapper import DelayWrapper
import torch
from utils.save_results import save_testing
from utils.stochastic_wrapper import StochActionWrapper
import time
from itertools import product

from matplotlib import animation
import matplotlib.pyplot as plt


def test_dida_rendering_seeds(
    n_steps=1000,
    traj_len=250,
    gamma=1,
    save_path="test",
    seed=0,
    save_every=1,
):
    save_seed = list(filter(lambda x: ".pt" in x, os.listdir(save_path)))
    save_seed = [int(s.split(".")[0].split("_")[-1]) for s in save_seed]
    for s in save_seed:
        print("Running seed {}".format(s))
        test_dida_rendering(
            n_steps=n_steps,
            traj_len=traj_len,
            gamma=gamma,
            save_path=save_path,
            seed=seed,
            save_seed=s,
            save_every=1,
        )


def save_frames_as_gif(frames, path="./", filename="gym_animation.gif"):

    # Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis("off")

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(os.path.join(path, filename), writer="imagemagick", fps=60)


def test_dida_rendering(
    n_steps=1000,
    traj_len=250,
    gamma=1,
    save_path="test",
    seed=0,
    save_seed=0,
    save_every=1,
):

    try:
        os.makedirs(os.path.join(save_path, "testing_seed_{}".format(save_seed)))
    except:
        pass

    with open(os.path.join(save_path, "parameters.json")) as f:
        param = json.load(f)

    start_time = time.time()
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = gym.make(param["env"], **param["env_kwargs"])
    if param["stoch_mdp_distrib"] is not None:
        env = StochActionWrapper(
            env, distrib=param["stoch_mdp_distrib"], param=param["stoch_mdp_param"]
        )
    denv = DelayWrapper(copy.deepcopy(env), delay=param["delay"])

    state_dim = get_space_dim(env.observation_space)
    ext_state_dim = get_space_dim(denv.observation_space)
    action_dim = get_space_dim(env.action_space)
    if param["policy"] == "mlp":
        n_neurons = [i for i in param["n_neurons"]]
        n_neurons = [ext_state_dim] + n_neurons + [action_dim]
        delayed_policy = mlp(
            n_neurons,
        )
    elif param["policy"] == "cnn":
        n_channels = [i for i in param["n_channels"]]
        l_out = cnn.output_size(param["delay"], n_channels, param["kernel_size"])
        n_channels = [action_dim] + n_channels
        n_neurons = [i for i in n_neurons]
        n_neurons = [state_dim + l_out * n_channels[-1]] + n_neurons + [action_dim]
        delayed_policy = cnn(n_channels, n_neurons, param["kernel_size"])
    else:
        raise ValueError

    delayed_policy.load_state_dict(
        torch.load(os.path.join(save_path, "policy_{}.pt".format(save_seed)))
    )

    delayed_policy.eval()

    states_buffer, actions_buffer, rewards_buffer, failed_reset, frames = test_policy(
        delayed_policy,
        denv,
        steps=n_steps,
        traj_len=traj_len,
        delay=param["delay"],
        state_dim=state_dim,
        action_dim=action_dim,
        save_frame=True,
    )

    save_frames_as_gif(
        frames, path=os.path.join(save_path, "testing_seed_{}".format(save_seed))
    )
    logging.info("Training time {} seconds.".format((time.time() - start_time)))


def test_dida_seeds(
    n_steps=1000,
    traj_len=250,
    gamma=1,
    save_path="test",
    seed=0,
    save_every=1,
):
    save_seed = list(filter(lambda x: ".pt" in x, os.listdir(save_path)))
    save_seed = [int(s.split(".")[0].split("_")[-1]) for s in save_seed]
    for s in save_seed:
        print("Running seed {}".format(s))
        test_dida(
            n_steps=n_steps,
            traj_len=traj_len,
            gamma=gamma,
            save_path=save_path,
            seed=seed,
            save_seed=s,
            save_every=1,
        )


def test_dida(
    n_steps=1000,
    traj_len=250,
    gamma=1,
    save_path="test",
    seed=0,
    save_seed=0,
    save_every=1,
):
    """_summary_

    Args:
        n_steps (int, optional): _description_. Defaults to 1000.
        traj_len (int, optional): _description_. Defaults to 250.
        gamma (int, optional): _description_. Defaults to 1.
        save_path (str, optional): _description_. Defaults to 'test'.
        seed (int, optional): _description_. Defaults to 0.
        save_seed (int, optional): _description_. Defaults to 0.
        save_every (int, optional): _description_. Defaults to 1.

    Raises:
        ValueError: _description_
    """

    try:
        os.makedirs(os.path.join(save_path, "testing_seed_{}".format(save_seed)))
    except:
        pass

    with open(os.path.join(save_path, "parameters.json")) as f:
        param = json.load(f)

    start_time = time.time()
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = gym.make(param["env"], **param["env_kwargs"])
    if param["stoch_mdp_distrib"] is not None:
        env = StochActionWrapper(
            env, distrib=param["stoch_mdp_distrib"], param=param["stoch_mdp_param"]
        )
    denv = DelayWrapper(copy.deepcopy(env), delay=param["delay"])

    state_dim = get_space_dim(env.observation_space)
    ext_state_dim = get_space_dim(denv.observation_space)
    action_dim = get_space_dim(env.action_space)
    if param["policy"] == "mlp":
        n_neurons = [i for i in param["n_neurons"]]
        n_neurons = [ext_state_dim] + n_neurons + [action_dim]
        delayed_policy = mlp(
            n_neurons,
        )
    elif param["policy"] == "cnn":
        n_channels = [i for i in param["n_channels"]]
        l_out = cnn.output_size(param["delay"], n_channels, param["kernel_size"])
        n_channels = [action_dim] + n_channels
        n_neurons = [i for i in n_neurons]
        n_neurons = [state_dim + l_out * n_channels[-1]] + n_neurons + [action_dim]
        delayed_policy = cnn(n_channels, n_neurons, param["kernel_size"])
    else:
        raise ValueError

    delayed_policy.load_state_dict(
        torch.load(os.path.join(save_path, "policy_{}.pt".format(save_seed)))
    )

    delayed_policy.eval()

    states_buffer, actions_buffer, rewards_buffer, failed_reset = test_policy(
        delayed_policy,
        denv,
        steps=n_steps,
        traj_len=traj_len,
        delay=param["delay"],
        state_dim=state_dim,
        action_dim=action_dim,
    )

    save_testing(
        states_buffer,
        actions_buffer,
        rewards_buffer,
        failed_reset,
        save_path=os.path.join(save_path, "testing_seed_{}".format(save_seed)),
        seed=seed,
    )
    logging.info("Training time {} seconds.".format((time.time() - start_time)))


def test_expert(
    n_steps=1000,
    traj_len=250,
    gamma=1,
    save_path="test",
    seed=0,
    save_every=1,
):
    try:
        os.makedirs(os.path.join(save_path, "testing_expert"))
    except:
        pass

    with open(os.path.join(save_path, "parameters.json")) as f:
        param = json.load(f)

    start_time = time.time()
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = gym.make(param["env"], **param["env_kwargs"])
    if param["stoch_mdp_distrib"] is not None:
        env = StochActionWrapper(
            env, distrib=param["stoch_mdp_distrib"], param=param["stoch_mdp_param"]
        )
    denv = DelayWrapper(copy.deepcopy(env), delay=param["delay"])

    algo_expert = param["algo_expert"].lower()
    expert_loader = eval(algo_expert.upper())
    if param["stoch_mdp_distrib"] is not None:
        expert_policy = expert_loader.load(
            os.path.join(
                "trained_agent",
                "{}_{}_noise_{}_param_{}".format(
                    param["algo_expert"],
                    env.unwrapped.spec.id,
                    param["stoch_mdp_distrib"],
                    param["stoch_mdp_param"],
                ),
                "policy",
            )
        )
    else:
        expert_policy = expert_loader.load(
            os.path.join(
                "trained_agent",
                "{}_{}".format(param["algo_expert"], env.unwrapped.spec.id),
                "policy",
            )
        )

    state_dim = get_space_dim(env.observation_space)
    ext_state_dim = get_space_dim(denv.observation_space)
    action_dim = get_space_dim(env.action_space)
    states_buffer, actions_buffer, rewards_buffer, failed_reset = test_policy_expert(
        expert_policy,
        env,
        steps=n_steps,
        traj_len=traj_len,
        state_dim=state_dim,
        action_dim=action_dim,
    )

    save_testing(
        states_buffer,
        actions_buffer,
        rewards_buffer,
        failed_reset,
        save_path=os.path.join(save_path, "testing_expert"),
        seed=seed,
    )
    logging.info("Training time {} seconds.".format((time.time() - start_time)))


def test_policy_expert(
    policy,
    env,
    steps,
    traj_len,
    state_dim,
    action_dim,
):
    try:
        max_len = min(env.spec.max_episode_steps, traj_len)
    except:
        max_len = traj_len

    failed_reset = 0
    states_buffer = np.empty((steps // max_len + 1, traj_len + 1, state_dim))
    states_buffer[:] = np.nan
    actions_buffer = np.empty((steps // max_len + 1, traj_len, action_dim))
    actions_buffer[:] = np.nan
    rewards_buffer = np.empty((steps // max_len + 1, traj_len))
    rewards_buffer[:] = np.nan

    done = False
    traj_t = 0  # The first step of the current trajectory
    n_traj = -1
    for t in range(steps):
        # if the env is reinitialized, sample the first delay actions without following the policy
        if (t - traj_t) % traj_len == 0 or done:
            obs = env.reset()
            done = False
            traj_t = t
            n_traj += 1
            if n_traj >= steps // max_len + 1:
                empty_line = np.empty((1, traj_len, state_dim))
                empty_line[:] = np.nan
                states_buffer = np.vstack((states_buffer, empty_line))
                empty_line = np.empty((1, traj_len, action_dim))
                empty_line[:] = np.nan
                actions_buffer = np.vstack((actions_buffer, empty_line))
                empty_line = np.empty((1, traj_len))
                empty_line[:] = np.nan
                rewards_buffer = np.vstack((rewards_buffer, empty_line))
                print(rewards_buffer.shape)

            states_buffer[n_traj, t - traj_t] = obs[0]

        action, _ = policy.predict(obs)

        obs, reward, done, _ = env.step(action)
        states_buffer[n_traj, t - traj_t + 1] = obs[0]
        actions_buffer[n_traj, t - traj_t] = action
        rewards_buffer[n_traj, t - traj_t] = reward

    return states_buffer, actions_buffer, rewards_buffer, failed_reset


def test_policy(
    policy, denv, steps, traj_len, delay, state_dim, action_dim, save_frame=False
):
    try:
        max_len = min(denv.spec.max_episode_steps, traj_len)
    except:
        max_len = traj_len

    failed_reset = 0
    states_buffer = np.empty((steps // max_len + 1, traj_len + delay, state_dim))
    states_buffer[:] = np.nan
    actions_buffer = np.empty((steps // max_len + 1, traj_len + delay, action_dim))
    actions_buffer[:] = np.nan
    rewards_buffer = np.empty((steps // max_len + 1, traj_len))
    rewards_buffer[:] = np.nan

    if save_frame:
        frames = []

    done = False
    traj_t = 0  # The first step of the current trajectory
    n_traj = -1
    for t in range(steps):
        # if the env is reinitialized, sample the first delay actions without following the policy
        if (t - traj_t) % traj_len == 0 or done:
            obs = denv.reset()
            traj_t = t
            if obs[0] is None:
                done = True
                failed_reset += 1
            else:
                done = False
                n_traj += 1
                if n_traj >= steps // max_len + 1:
                    empty_line = np.empty((1, traj_len + delay, state_dim))
                    empty_line[:] = np.nan
                    states_buffer = np.vstack((states_buffer, empty_line))
                    empty_line = np.empty((1, traj_len + delay, action_dim))
                    empty_line[:] = np.nan
                    actions_buffer = np.vstack((actions_buffer, empty_line))
                    empty_line = np.empty((1, traj_len))
                    empty_line[:] = np.nan
                    rewards_buffer = np.vstack((rewards_buffer, empty_line))
                    print(rewards_buffer.shape)

                states_buffer[n_traj, t - traj_t] = obs[0]
                actions_buffer[n_traj, t - traj_t : t - traj_t + delay] = obs[1]

        if not done:
            state = torch.from_numpy(obs[0]).float().unsqueeze(0)
            actions = torch.from_numpy(obs[1]).float().unsqueeze(0)
            action = policy(state, actions)
            if action.ndim > 2:
                action = action[:, -1, :]
            action = action.reshape(-1).detach().numpy()
            obs, reward, done, _ = denv.step(action)
            states_buffer[n_traj, t - traj_t + 1] = obs[0]
            actions_buffer[n_traj, t - traj_t + delay] = action
            if done and (delay > 1 and len(reward) > 1):
                states_buffer[
                    n_traj, t - traj_t + 1 : t - traj_t + 1 + delay
                ] = denv._hidden_obs[1:]
                rewards_buffer[
                    n_traj, t - traj_t : t - traj_t + delay
                ] = denv._reward_stock
            else:
                rewards_buffer[n_traj, t - traj_t] = reward
            if save_frame:
                frames.append(denv.render(mode="rgb_array"))

    denv.close()
    if save_frame:
        return states_buffer, actions_buffer, rewards_buffer, failed_reset, frames
    else:
        return states_buffer, actions_buffer, rewards_buffer, failed_reset


def test_dida_other_delay_seeds(
    n_steps=1000,
    traj_len=250,
    gamma=1,
    save_path="test",
    seed=None,
    delays: list = [1, 2, 3],
    save_every=1,
    test_all_delays_inferior=True,
):
    save_seed = list(filter(lambda x: ".pt" in x, os.listdir(save_path)))
    save_seed = [int(s.split(".")[0].split("_")[-1]) for s in save_seed]
    save_seed.sort()
    np.random.seed(seed)
    torch.manual_seed(seed)

    if test_all_delays_inferior:
        with open(os.path.join(save_path, "parameters.json")) as f:
            delays = np.arange(1, json.load(f)["delay"] + 1)

    for (s, d) in product(save_seed, delays):
        print("Running seed {} with delay {}".format(s, d))
        test_dida_other_delay(
            n_steps=n_steps,
            traj_len=traj_len,
            gamma=gamma,
            save_path=save_path,
            save_seed=s,
            save_every=1,
            delay=d,
            seed=seed,
        )


def test_dida_other_delay(
    n_steps=1000,
    traj_len=250,
    gamma=1,
    save_path="test",
    save_seed=0,
    save_every=1,
    delay=1,
    seed=0,
):
    """_summary_

    Args:
        n_steps (int, optional): _description_. Defaults to 1000.
        traj_len (int, optional): _description_. Defaults to 250.
        gamma (int, optional): _description_. Defaults to 1.
        save_path (str, optional): _description_. Defaults to 'test'.
        seed (int, optional): _description_. Defaults to 0.
        save_seed (int, optional): _description_. Defaults to 0.
        save_every (int, optional): _description_. Defaults to 1.

    Raises:
        ValueError: _description_
    """

    try:
        os.makedirs(
            os.path.join(
                save_path, "testing_delay_{}".format(delay), "seed_{}".format(save_seed)
            )
        )
    except:
        pass

    with open(os.path.join(save_path, "parameters.json")) as f:
        param = json.load(f)
    assert (
        param["delay"] >= delay
    ), "The test delay must be inferior to the train delay."

    start_time = time.time()
    env = gym.make(param["env"], **param["env_kwargs"])
    if param["stoch_mdp_distrib"] is not None:
        env = StochActionWrapper(
            env, distrib=param["stoch_mdp_distrib"], param=param["stoch_mdp_param"]
        )
    denv = DelayWrapper(copy.deepcopy(env), delay=delay)

    state_dim = get_space_dim(env.observation_space)
    # ext_state_dim = get_space_dim(denv.observation_space)
    action_dim = get_space_dim(env.action_space)
    ext_state_dim = state_dim + param["delay"] * action_dim
    if param["policy"] == "mlp":
        n_neurons = [i for i in param["n_neurons"]]
        n_neurons = [ext_state_dim] + n_neurons + [action_dim]
        delayed_policy = mlp(
            n_neurons,
        )
    elif param["policy"] == "cnn":
        n_channels = [i for i in param["n_channels"]]
        l_out = cnn.output_size(param["delay"], n_channels, param["kernel_size"])
        n_channels = [action_dim] + n_channels
        n_neurons = [i for i in n_neurons]
        n_neurons = [state_dim + l_out * n_channels[-1]] + n_neurons + [action_dim]
        delayed_policy = cnn(n_channels, n_neurons, param["kernel_size"])
    else:
        raise ValueError

    delayed_policy.load_state_dict(
        torch.load(os.path.join(save_path, "policy_{}.pt".format(save_seed)))
    )

    delayed_policy.eval()

    (
        states_buffer,
        actions_buffer,
        rewards_buffer,
        failed_reset,
    ) = test_policy_other_delay(
        policy=delayed_policy,
        denv=denv,
        delay_env=delay,
        steps=n_steps,
        traj_len=traj_len,
        delay_policy=param["delay"],
        state_dim=state_dim,
        action_dim=action_dim,
    )

    save_testing(
        states_buffer,
        actions_buffer,
        rewards_buffer,
        failed_reset,
        save_path=os.path.join(
            save_path, "testing_delay_{}".format(delay), "seed_{}".format(save_seed)
        ),
        seed=seed,
    )
    logging.info("Training time {} seconds.".format((time.time() - start_time)))



def test_policy_other_delay(
    policy,
    delay_env,
    denv,
    steps,
    traj_len,
    delay_policy,
    state_dim,
    action_dim,
    save_frame=False,
):
    try:
        max_len = min(denv.spec.max_episode_steps, traj_len)
    except:
        max_len = traj_len

    failed_reset = 0

    # Initialize buffers
    states_buffer = np.empty((1, max_len + delay_policy, state_dim))
    states_buffer[:] = np.nan
    actions_buffer = np.empty((1, max_len + delay_policy, action_dim))
    actions_buffer[:] = np.nan
    rewards_buffer = np.empty((1, max_len))
    rewards_buffer[:] = np.nan

    if save_frame:
        frames = []

    done = False
    traj_t = 0  # The first step of the current trajectory
    n_traj = -1
    for t in range(steps):
        # if the env is reinitialized, sample the first delay actions without following the policy
        if (t - traj_t) % max_len == 0 or done:
            obs = denv.reset()
            traj_t = t
            if obs[0] is None:
                done = True
                failed_reset += 1
            else:
                done = False
                n_traj += 1

                # Append buffer
                empty_line = np.empty((1, max_len + delay_policy, state_dim))
                empty_line[:] = np.nan
                states_buffer = np.vstack((states_buffer, empty_line))
                empty_line = np.empty((1, max_len + delay_policy, action_dim))
                empty_line[:] = np.nan
                actions_buffer = np.vstack((actions_buffer, empty_line))
                empty_line = np.empty((1, max_len))
                empty_line[:] = np.nan
                rewards_buffer = np.vstack((rewards_buffer, empty_line))

                states_buffer[n_traj, t - traj_t] = obs[0]
                actions_buffer[n_traj, t - traj_t : t - traj_t + delay_env] = obs[1]

        if not done:
            if (
                t - traj_t < delay_policy - delay_env
            ):  # select random action to initialize process
                action = denv.action_space.sample()
            else:
                state = (
                    torch.from_numpy(
                        states_buffer[n_traj, t - traj_t - delay_policy + delay_env]
                    )
                    .float()
                    .unsqueeze(0)
                )  # delay state even more
                # actions = torch.from_numpy(obs[1]).float().unsqueeze(0)
                actions = (
                    torch.from_numpy(
                        actions_buffer[
                            n_traj,
                            t
                            - traj_t
                            - delay_policy
                            + delay_env : t
                            - traj_t
                            + delay_env,
                        ]
                    )
                    .float()
                    .unsqueeze(0)
                )  # append older actions to create synthetic augmented state
                action = policy(state, actions)
                action = action.reshape(-1).detach().numpy()

            if action.ndim > 2:
                action = action[:, -1, :]
            obs, reward, done, _ = denv.step(action)
            states_buffer[n_traj, t - traj_t + 1] = obs[0]
            actions_buffer[n_traj, t - traj_t + delay_env] = action
            if done and (delay_env >= 1 and len(reward) > 1):
                states_buffer[
                    n_traj, t - traj_t + 1 : t - traj_t + 1 + delay_env
                ] = denv._hidden_obs[1:]
                rewards_buffer[
                    n_traj, t - traj_t : t - traj_t + delay_env
                ] = denv._reward_stock
            else:
                rewards_buffer[n_traj, t - traj_t] = reward
            if save_frame:
                frames.append(denv.render(mode="rgb_array"))

    denv.close()
    if save_frame:
        return states_buffer, actions_buffer, rewards_buffer, failed_reset, frames
    else:
        return states_buffer, actions_buffer, rewards_buffer, failed_reset


if __name__ == "__main__":
    fire.Fire()
