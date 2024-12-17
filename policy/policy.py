from policy.neural_networks import mlp, cnn, AutoregNet, get_number_of_parameters


def get_policy_dida(policy, delay, action_dim, state_dim, aug_state_dim, kernel_size, all_actions):
    if policy == "mlp":
        n_neurons = [i for i in n_neurons]
        if all_actions:
            action_dims = [action_dim] + n_neurons + [action_dim]
            delayed_policy = AutoregNet(
                state_dim,
                action_dims=action_dims,
                delay=delay,
            )
        else:
            n_neurons = [aug_state_dim] + n_neurons + [action_dim]
            delayed_policy = mlp(
                n_neurons,
            )
    elif policy == "cnn":
        n_channels = [i for i in n_channels]
        l_out = cnn.output_size(delay, n_channels, kernel_size)
        n_channels = [action_dim] + n_channels
        n_neurons = [i for i in n_neurons]
        n_neurons = [state_dim + l_out * n_channels[-1]] + n_neurons + [action_dim]
        delayed_policy = cnn(n_channels, n_neurons, kernel_size)
    else:
        raise ValueError
    
    return delayed_policy