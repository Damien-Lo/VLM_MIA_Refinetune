
import torch
import numpy as np

def renyi_probs(token_probs_clamped, alpha):
    if alpha == "inf":
        # For α = ∞, we want to create a one-hot vector with the maximum probability
        max_values, max_indices = torch.max(token_probs_clamped, dim=-1, keepdim=True)
        # Create a zero tensor with the same shape
        renyi_normalized = torch.zeros_like(token_probs_clamped)
        # Set the maximum value to 1 (one-hot encoding)
        renyi_normalized.scatter_(-1, max_indices, 1.0)
        return renyi_normalized
    else:
        renyi_numerator = torch.pow(token_probs_clamped, alpha)
        renyi_denominator = torch.sum(renyi_numerator, dim=-1, keepdim=True)
        renyi_normalized = renyi_numerator / renyi_denominator
        return renyi_normalized


def get_meta_metrics(cfg, goals, input_ids, attention_masks=None):
    """
    input_ids :  mix_input_ids from the mod_infer
    goals: logits, probabilities and log_probabilities of each part
        goals = {
            "img" : {
                "logit" : torch.Tensor(batch x input_ids),
                "probabilities : torch.Tensor (batch x input_ids)
                "log_probabilities : torch.Tensor (batch x input_ids)
            }
            "desp": ...
        }
    """

    input_ids = input_ids[1:]

    meta_metrics = dict()
    epsilon = 1e-10

    for _part in goals:
        logits = goals[_part]["logits"]
        probabilities = goals[_part]["probabilities"]
        log_probabilities = goals[_part]["log_probabilities"]
        meta_metrics[_part] = dict()
        meta_metrics[_part]["ppl"] = []
        meta_metrics[_part]["entropies"] = [[] for _ in range(logits.size(0))]
        meta_metrics[_part]["all_prob"] = [[] for _ in range(logits.size(0))]
        meta_metrics[_part]["modified_entropies"] = [[] for _ in range(logits.size(0))]
        meta_metrics[_part]["max_probs"] = [[] for _ in range(logits.size(0))]
        meta_metrics[_part]["gap_probs"] = [[] for _ in range(logits.size(0))]
        meta_metrics[_part]["renyi_05_entro"] = [[] for _ in range(logits.size(0))]
        meta_metrics[_part]["renyi_2_entro"] = [[] for _ in range(logits.size(0))]
        meta_metrics[_part]["losses"] = [[] for _ in range(logits.size(0))]
        meta_metrics[_part]["modified_entropies"] = [[] for _ in range(logits.size(0))]
        meta_metrics[_part]["modified_entropies_alpha_05"] = [[] for _ in range(logits.size(0))]
        meta_metrics[_part]["modified_entropies_alpha_2"] = [[] for _ in range(logits.size(0))]
        meta_metrics[_part]["renyi_05_probs"] = [[] for _ in range(logits.size(0))]
        meta_metrics[_part]["renyi_1_probs"] = [[] for _ in range(logits.size(0))]
        meta_metrics[_part]["renyi_2_probs"] = [[] for _ in range(logits.size(0))]
        meta_metrics[_part]["reny_inf_probs"] = [[] for _ in range(logits.size(0))]

        for _batch_idx in range(logits.size(0)):
            for _idx, token_id in enumerate(input_ids):
                if attention_masks is not None:
                    if attention_masks[_batch_idx, token_id] = 0:
                        continue
                token_probs = probabilities[_batch_idx, _idx, :]
                token_probs = token_probs.detach().to(dtype=torch.float64)
                token_log_probs = log_probabilities[_batch_idx, _idx, :]
                token_log_probs = token_log_probs.detach().to(dtype=torch.float64)

                token_probs_clamped = torch.clamp(token_probs, min=epsilon, max=1-epsilon)

                # Renyi_1
                entropy = -(token_probs * token_log_probs).sum().item()
                meta_metrics[_part]["entropies"][_batch_idx].append(entropy)
                meta_metrics[_part]["renyi_1_probs"][_batch_idx].append(renyi_probs(token_probs_clamped, 1))

                # Renyi_05
                alpha=0.5
                renyi_05 = (1 / (1-alpha)) * torch.log(torch.sum(torch.pow(token_probs_clamped, alpha))).item()
                meta_metrics[_part]["renyi_05_entro"][_batch_idx].append(renyi_05)
                meta_metrics[_part]["renyi_05_probs"][_batch_idx].append(renyi_probs(token_probs_clamped, 0.5))

                # Renyi_2
                alpha=2
                renyi_2 = (1 / (1-alpha)) * torch.log(torch.sum(torch.pow(token_probs_clamped, alpha))).item()
                meta_metrics[_part]["renyi_2_entro"][_batch_idx].append(renyi_2)
                meta_metrics[_part]["renyi_2_probs"][_batch_idx].append(renyi_probs(token_probs_clamped, 2))

                # Renyi_inf
                max_p = token_log_probs.max().item()
                second_p = token_log_probs[token_log_probs != token_log_probs.max()].max().item()
                gap_p = max_p - second_p
                meta_metrics[_part]["gap_probs"][_batch_idx].append(gap_p)
                meta_metrics[_part]["max_probs"][_batch_idx].append(max_p)
                meta_metrics[_part]["renyi_inf_probs"][_batch_idx].append(renyi_probs(token_probs_clamped, "inf"))

                min_k_p = token_log_probs[token_id].item()
                meta_metrics[_part]["all_prob"][_batch_idx].append(min_k_p)

                cross_entropy_loss = -min_k_p
                meta_metrics[_part]["losses"][_batch_idx].append(cross_entropy_loss)

                # Modified entropy
                p_y = token_probs_safe[token_id].item()
                modified_entropy = -(1 - p_y) * torch.log(torch.tensor(p_y)) - (token_probs * torch.log(1 - token_probs_clamped)).sum().item() + p_y * torch.log(torch.tensor(1 - p_y)).item()
                meta_metrics[_part]["modified_entropies"][_batch_idx].append(modified_entropy)

                token_probs_remaining = torch.cat((token_probs_clamped[:token_id], token_probs_clamped[token_id+1:]))
                
                for alpha in [0.5,2]:
                    entropy = - (1 / abs(1 - alpha)) * (
                        (1-p_y)* p_y**(abs(1-alpha))\
                            - (1-p_y)
                            + torch.sum(token_probs_remaining * torch.pow(1-token_probs_remaining, abs(1-alpha))) \
                            - torch.sum(token_probs_remaining)
                            ).item() 
                    if alpha==0.5:
                        meta_metrics[_part]["modified_entropies_alpha_05"][_batch_idx].append(entropy)
                    if alpha==2:
                        meta_metrics[_part]["modified_entropies_alpha_2"][_batch_idx].append(entropy)

            loss = np.nanmean(meta_metrics[_part]["losses"][_batch_idx])
            meta_metrics[_part]["ppl"].append(np.exp(loss))

    return meta_metrics