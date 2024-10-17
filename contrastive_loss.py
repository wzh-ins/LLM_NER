import torch
from LLMCC_training import label2id


def generate_positive_pairs(batch_data):
    anchor_embeddings = []
    positive_embeddings = []
    for i in range(len(batch_data)):
        data_i = batch_data[i]
        embeddings_i = data_i['sequence_output']
        labels_i = data_i['labels']
        dataset_type_i = data_i['dataset_type']
        for j in range(i + 1, len(batch_data)):
            data_j = batch_data[j]
            embeddings_j = data_j['sequence_output']
            labels_j = data_j['labels']
            dataset_type_j = data_j['dataset_type']

            min_len = min(len(labels_i), len(labels_j))
            for k in range(min_len):
                label_i = labels_i[k].item()
                label_j = labels_j[k].item()
                if label_i == -100 or label_j == -100:
                    continue

                if dataset_type_i == 'train' and dataset_type_j == 'train':
                    if label_i == label_j and label_i != label2id['O']:
                        h_i = embeddings_i[k]
                        h_j = embeddings_j[k]
                        anchor_embeddings.append(h_i)
                        positive_embeddings.append(h_j)

                elif (dataset_type_i == 'llm' and dataset_type_j == 'train') or (dataset_type_i == 'train' and dataset_type_j == 'llm'):
                    if label_i == label_j and label_i != label2id['O']:
                        h_i = embeddings_i[k]
                        h_j = embeddings_j[k]
                        anchor_embeddings.append(h_i)
                        positive_embeddings.append(h_j)
    return anchor_embeddings, positive_embeddings

def generate_negative_pairs(batch_data):
    anchor_embeddings = []
    negative_embeddings = []
    for i in range(len(batch_data)):
        data_i = batch_data[i]
        embeddings_i = data_i['sequence_output']
        labels_i = data_i['labels']
        dataset_type_i = data_i['dataset_type']
        for j in range(len(batch_data)):
            if i == j:
                continue
            data_j = batch_data[j]
            embeddings_j = data_j['sequence_output']
            labels_j = data_j['labels']
            dataset_type_j = data_j['dataset_type']

            min_len = min(len(labels_i), len(labels_j))
            for k in range(min_len):
                label_i = labels_i[k].item()
                label_j = labels_j[k].item()
                if label_i == -100 or label_j == -100:
                    continue

                if dataset_type_i == 'train' and dataset_type_j == 'train':
                    if label_i != label2id['O'] and label_j == label2id['O']:
                        h_i = embeddings_i[k]
                        h_j = embeddings_j[k]
                        anchor_embeddings.append(h_i)
                        negative_embeddings.append(h_j)

                elif dataset_type_i == 'llm' and dataset_type_j == 'llm':
                    if label_i != label_j and label_i != label2id['O'] and label_j != label2id['O']:
                        h_i = embeddings_i[k]
                        h_j = embeddings_j[k]
                        anchor_embeddings.append(h_i)
                        negative_embeddings.append(h_j)
    return anchor_embeddings, negative_embeddings

def contrastive_loss(anchor_embeddings, positive_embeddings, negative_embeddings, margin=1.0):
    if not anchor_embeddings:
        return torch.tensor(0.0).to(device)
    anchor_embeddings = torch.stack(anchor_embeddings)
    positive_embeddings = torch.stack(positive_embeddings)
    negative_embeddings = torch.stack(negative_embeddings)

    pos_distance = torch.norm(anchor_embeddings - positive_embeddings, dim=1)
    neg_distance = torch.norm(anchor_embeddings - negative_embeddings, dim=1)

    losses = torch.clamp(pos_distance - neg_distance + margin, min=0.0)
    loss = losses.mean()
    return loss
