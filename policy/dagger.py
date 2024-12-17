import torch 
from utils.util import index_sampler
from tqdm import tqdm 
import numpy as np

def imitation_learning(
    action_buffer_len,
    delayed_policy,
    state_buffer,
    action_buffer,
    action_buffer_del,
    mask_buffer,
    batch_size=32,
    optimizer=None,
    all_actions=False,
):
    index_batch, sampler_size = index_sampler(mask_buffer, batch_size=batch_size)
    losses = 0
    delayed_policy.train()

    pbar = tqdm(total=sampler_size)
    for idx in index_batch:
        pbar.update(1)
        s = state_buffer[idx]
        e = np.stack(
            [
                np.roll(action_buffer_del, -i, axis=0)[idx]
                for i in range(action_buffer_len)
            ]
        )
        e = e.transpose(1, 0, 2)
        s = torch.from_numpy(s).float()
        e = torch.from_numpy(e).float()
        optimizer.zero_grad()
        a_pred = delayed_policy(s, e)
        if all_actions:
            a = np.stack(
                [
                    np.roll(action_buffer_del, -i, axis=0)[idx]
                    for i in range(1, action_buffer_len + 1)
                ]
            )
            a = a.transpose(1, 0, 2)
            a = torch.from_numpy(a).float()
            loss_weight = torch.linspace(0.1, 1, a.shape[1]).reshape(-1, a.shape[2])
        else:
            a = torch.from_numpy(
                np.roll(action_buffer, -action_buffer_len, axis=0)[idx]
            ).float()
            loss_weight = 1
        loss = torch.sum(torch.abs(a_pred - a) * loss_weight)

        loss.backward()
        optimizer.step()
        losses += loss.item()
    pbar.close()
    delayed_policy.eval()
    return losses / sum(mask_buffer), optimizer

def get_beta_routine_weights(beta_routine, training_rounds):
    memoryless_sampling = False
    if beta_routine is None:
        beta_routine_weights = [1] + [0] * (training_rounds - 1)
    elif beta_routine == "linear":
        beta_routine_weights = np.linspace(1, 0, training_rounds)
    elif beta_routine == "bc":
        beta_routine_weights = np.ones(training_rounds)
    elif beta_routine == "no_undelayed":
        beta_routine_weights = [1] + [0] * (training_rounds - 1)
        memoryless_sampling = True
    return beta_routine_weights, memoryless_sampling