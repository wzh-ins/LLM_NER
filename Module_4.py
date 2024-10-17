import openai
import torch

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
                parts = line.strip().split()
                if len(parts) >= 2:
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

def write_bio_data(sentences, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            tokens = sentence['tokens']
            labels = sentence['labels']
            for token, label in zip(tokens, labels):
                f.write(f"{token} {label}\n")
            f.write('\n')

def construct_verification_prompt(tokens, predicted_labels,law):

    law_added_information = ""
    labeled_text = '\n'.join([f"{token}\t{label}" for token, label in zip(tokens, predicted_labels)])
    prompt = f"""Assume the role of an NLP expert with a deep understanding of entity tagging in 
    BIO format, brought into a team with three distinguished peers from your field. Your collective 
    task is to verify the accuracy of BIO labels provided for the text [T] where the types of 
    entities include ... Begin by independently reviewing [T] through the information law [S], and its associated BIO labels. 
    Identify any inaccuracies or inconsistencies in the labeling. 
    After your individual assessment, convene with your peers to discuss your findings. 
    Each of you will present the discrepancies you've discovered and propose corrections based on 
    your expertise. Engage in a detailed collaborative review, challenging each other’s assessments 
    and refining the labels through collective insight. This iterative process of evaluation, 
    debate, and adjustment continues until all of you agree on the correctness of every label. 
    Conclude by documenting the final, consensus-driven labels for [T], ensuring they accurately 
    represent the entities as per your combined expert analysis.
    The text [T] is {labeled_text};
    The law [S] is {law}, {law_added_information}.
"""
    return prompt

def verify_and_correct_predictions(sentences,law, model_name="gpt-4", max_retries=5):
    corrected_sentences = []
    for idx, sentence in enumerate(sentences):
        tokens = sentence['tokens']
        predicted_labels = sentence['labels']
        prompt = construct_verification_prompt(tokens, predicted_labels,law)
        print(f"Processing sentence {idx + 1}/{len(sentences)}")
        for attempt in range(max_retries):
            try:
                response = openai.ChatCompletion.create(
                    model=model_name,
                    messages=[
                        {"role": "system",
                         "content": "You are a helpful assistant."},
                        {"role": "user",
                         "content": prompt}
                    ],
                )
                corrected_text = response.choices[0].message.content.strip()

                corrected_tokens = []
                corrected_labels = []
                for line in corrected_text.split('\n'):
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        token = parts[0]
                        label = parts[-1]
                        corrected_tokens.append(token)
                        corrected_labels.append(label)

                if corrected_tokens == tokens:
                    corrected_sentences.append({
                        'tokens': corrected_tokens,
                        'labels': corrected_labels
                    })
                    break
                else:
                    print("Token mismatch. Retrying...")
                    time.sleep(1)
            except openai.error.RateLimitError:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"Rate limit exceeded. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            except openai.error.OpenAIError as e:
                print(f"An error occurred: {e}")
                time.sleep(1)
        else:
            print("Failed to process sentence after multiple attempts. Using original predictions.")
            corrected_sentences.append(sentence)
    return corrected_sentences


