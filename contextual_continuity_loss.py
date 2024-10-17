import torch

from LLMCC_training import label2id


def contextual_continuity_loss(sequence_output, labels):
    """
    Compute the contextual continuity loss.
    """
    loss = 0.0
    count = 0
    batch_size, seq_len, hidden_size = sequence_output.shape
    for i in range(batch_size):
        for j in range(seq_len - 1):
            label_current = labels[i, j].item()
            label_next = labels[i, j + 1].item()
            if label_current == -100 or label_next == -100:
                continue
            if label_current == label_next and label_current != label2id['O']:
                h_current = sequence_output[i, j]
                h_next = sequence_output[i, j + 1]
                diff = h_current - h_next
                loss += torch.norm(diff, p=2) ** 2
                count += 1
    if count > 0:
        loss = loss / count
    else:
        loss = torch.tensor(0.0).to(sequence_output.device)
    return loss
