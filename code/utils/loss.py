import torch
import torch.nn.functional as F



def compute_win_rej(rewards, labels):
    """
    rewards: [b, n]
    chose_labels: [b]
    """
    # TODO
    assert rewards.size(1) == 4

    win_rewards = rewards[range(len(rewards)), labels][:, None] # [b, 1]
    i1 = [[i, i, i] for i in range(len(rewards))]
    i2 = [[i for i in range(4) if i!=l] for l in labels]
    rej_rewards = rewards[i1, i2] # [b, 3]

    return win_rewards - rej_rewards, win_rewards, rej_rewards



def dpo_loss(rewards, ref_rewards, chose_labels, beta):
    """
    Direct Preference Optimization: Your Language Model is Secretly a Reward Model

    rewards, ref_rewards: [b, n]
    chose_labels: [b]
    """
    win_rej = compute_win_rej(rewards, chose_labels)[0]
    ref_win_rej = compute_win_rej(ref_rewards, chose_labels)[0]

    loss = -F.logsigmoid((win_rej - ref_win_rej) * beta).mean()

    return loss


def dpo_loss_wo_ref(rewards, chose_labels, beta, margin=0):
    win_rej = compute_win_rej(rewards, chose_labels)[0]-margin
    loss = -F.logsigmoid(win_rej * beta).mean()
    return loss



def rrhf_loss(rewards, chose_labels):
    """
    RRHF: Rank Responses to Align Language Models with Human Feedback without tears

    rewards: [b, n]
    chose_labels: [b]
    """
    win_rej = compute_win_rej(rewards, chose_labels)[0]

    loss = -win_rej
    loss[loss<0] = 0
    loss = loss.mean()

    return loss


def pro_loss(rewards, chose_labels):
    """
    Preference Ranking Optimization for Human Alignment

    rewards: [b, n]
    chose_labels: [b]
    """
    win_rewards, rej_rewards = compute_win_rej(rewards, chose_labels)[1:]

    neg_temperatures = win_rewards - rej_rewards
    pos_temperatures = torch.max(neg_temperatures, dim=1, keepdim=True).values

    win_rewards = win_rewards * pos_temperatures
    rej_rewards = rej_rewards * neg_temperatures
    
    eps = 1e-10
    pro_loss = -win_rewards + (eps+win_rewards.exp()+rej_rewards.exp().sum(dim=-1, keepdim=True)).log()
    pro_loss = pro_loss.mean()

    return pro_loss