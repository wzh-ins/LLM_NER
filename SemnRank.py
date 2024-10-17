import openai

import torch
from itertools import combinations
from openai import OpenAI


client = OpenAI(
    api_key = "",
)


def read_bio_data(file_path):
    sentences = []
    current_tokens = []
    current_labels = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == '':
                if current_tokens:
                    sentences.append({
                        'tokens': current_tokens,
                        'labels': current_labels
                    })
                    current_tokens = []
                    current_labels = []
            else:
                if ' ' in line:
                    token, label = line.split(' ', 1)
                else:

                    parts = line.split()
                    if len(parts) == 2:
                        token, label = parts
                    else:
                        continue  # Skip malformed lines
                    token = parts[0]
                    label = parts[-1]
                current_tokens.append(token)
                current_labels.append(label)

        if current_tokens:
            sentences.append({
                'tokens': current_tokens,
                'labels': current_labels
            })
    return sentences


def get_embeddings(sentences, tokenizer, model, device):
    texts = [' '.join(sentence['tokens']) for sentence in sentences]
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    # Use the [CLS] token embedding
    cls_embeddings = outputs.last_hidden_state[:, 0, :] 
    return cls_embeddings 

def compute_equiv(a_tokens, b_tokens):
    a_text = ' '.join(a_tokens)
    b_text = ' '.join(b_tokens)
    prompt = f"Do the following two sentences convey the same meaning?\n\nSentence 1: {a_text}\nSentence 2: {b_text}\n\n Only Answer 'Yes' or 'No'."
    response = client.chat.completions.create(
        model="",
        messages=[
            {"role": "system",
             "content": "You are a helpful assistant."},
            {"role": "user",
             "content": prompt}
        ]
    )

    answer = (response.choices[0].message.content)
    if 'yes' in answer:
        return 1
    else:
        return 0

def build_equivalence_matrix(D):
    n = len(D)
    equiv_matrix = torch.zeros((n, n), dtype=torch.int)
    for i, j in combinations(range(n), 2):
        a_tokens = D[i]['tokens']
        b_tokens = D[j]['tokens']
        Imp_ab = compute_equiv(a_tokens, b_tokens)
        Imp_ba = compute_equiv(b_tokens, a_tokens)
        Equiv_ab = Imp_ab * Imp_ba
        equiv_matrix[i, j] = equiv_matrix[j, i] = Equiv_ab

    equiv_matrix.fill_diagonal_(1)
    return equiv_matrix

def cluster_demonstrations(D, equiv_matrix):
    clusters = []
    n = len(D)
    visited = [False] * n
    for i in range(n):
        if not visited[i]:
            cluster_indices = [i]
            visited[i] = True
            for j in range(i + 1, n):
                if not visited[j] and equiv_matrix[i, j] == 1:
                    cluster_indices.append(j)
                    visited[j] = True
            clusters.append(cluster_indices)
    return clusters

def select_demonstrations(clusters, D_embeddings, v_T, k):
    m = len(clusters)
    cluster_reps = []
    for cluster_indices in clusters:
        embeddings = D_embeddings[cluster_indices]  # Shape: (cluster_size, hidden_size)
        distances = torch.norm(embeddings - v_T, dim=1)  # Shape: (cluster_size,)
        min_idx = torch.argmin(distances)
        rep_idx = cluster_indices[min_idx]
        cluster_reps.append({
            'index': rep_idx,
            'distance': distances[min_idx],
            'cluster_indices': cluster_indices,
            'distances': distances
        })

    cluster_reps.sort(key=lambda x: x['distance'])
    selected_indices = []
    if k <= m:
        selected_indices = [rep['index'] for rep in cluster_reps[:k]]
    else:
        selected_indices = [rep['index'] for rep in cluster_reps]
        remaining_k = k - m

        candidates = []
        for rep in cluster_reps:
            indices = rep['cluster_indices']
            distances = rep['distances']
            sorted_indices = torch.argsort(distances)
            for idx in sorted_indices[1:]:
                candidate_idx = indices[idx]
                candidate_distance = distances[idx]
                candidates.append({'index': candidate_idx, 'distance': candidate_distance})

        candidates.sort(key=lambda x: x['distance'])
        selected_indices.extend([cand['index'] for cand in candidates[:remaining_k]])
    return selected_indices

def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    training_sentences = read_bio_data('')
    test_sentences = read_bio_data('')
    test_sentences = test_sentences[0:7]
    training_sentences = training_sentences[0:7]

    tokenizer = ""
    model = ""
    model.to(device)

    training_embeddings = get_embeddings(training_sentences, tokenizer, model, device)
    test_embeddings = get_embeddings(test_sentences, tokenizer, model, device)

    v_T = torch.mean(test_embeddings, dim=0, keepdim=True)

    distances = torch.norm(training_embeddings - v_T, dim=1)

    n = 4
    top_n_distances, top_n_indices = torch.topk(distances, n, largest=False)
    top_n_indices = top_n_indices.cpu().numpy()
    D = [training_sentences[i] for i in top_n_indices]
    D_embeddings = training_embeddings[top_n_indices]

    print("Building equivalence matrix using GPT-4...")
    equiv_matrix = build_equivalence_matrix(D)

    clusters_indices = cluster_demonstrations(D, equiv_matrix)

    k = 2
    selected_indices_in_D = select_demonstrations(clusters_indices, D_embeddings, v_T, k)
    selected_demonstrations = [D[i] for i in selected_indices_in_D]
    print(selected_demonstrations)

    for idx, demo in enumerate(selected_demonstrations):
        tokens = demo['tokens']
        labels = demo['labels']
        print(f"Demonstration {idx + 1}:")
        for token, label in zip(tokens, labels):
            print(f"{token}\t{label}")
        print("\n")

if __name__ == "__main__":
    main()
