import openai
import time
import random


openai.api_key = ''

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
                # Split token and label
                parts = line.split()
                if len(parts) >= 2:
                    token = parts[0]
                    label = parts[-1]
                    current_tokens.append(token)
                    current_labels.append(label)
                else:
                    continue

        if current_tokens:
            sentences.append({
                'tokens': current_tokens,
            })
    return sentences

def expand_sentence(sentence_tokens, field):

    original_text = ''.join(sentence_tokens)
    prompt = f"As an expert in the {field} field, please expand on the given text based on your understanding.\n\nOriginal Text: {original_text}"
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system",
             "content": "You are a helpful assistant."},
            {"role": "user",
             "content": prompt}
        ]
    )
    expanded_text = response.choices[0].message.content.strip()
    return expanded_text


def main():
    training_sentences = read_bio_data('')
    field = ''
    C_extended = []

    for idx, sentence in enumerate(training_sentences):
        tokens = sentence['tokens']
        print(f"Expanding sentence {idx + 1}/{len(training_sentences)}")
        expanded_text = expand_sentence(tokens, field)
        C_extended.append(expanded_text)

