import numpy as np
import pandas as pd
from collections import Counter
from scipy.stats import linregress
import math

'''

Law	Expression	Parameter definition	Main insight
Zipf’s law	f(r)∝1/r^s	f(r): word frequency; r: rank	Principle of least effort
Heaps' Law	V(N)∝N^β	V(N): vocabulary size; N: corpus size	Redundancy in language
Taylor’s law	σ^2∝μ^b	σ2: f(r) variance; μ: f(r) mean	Predictability of word usage
Hilberg's law	H(L)∝L^α	H(L): entropy of text; L: word count	Redundancy of information density
Ebeling’s law	C(L)∝L^d	C(L): variability of text; L: character count	Order and randomness in language
Menzerath's Law	Φ(L)∝L^(-γ)	Φ(L): character length; L: word count	Larger the whole, smaller the parts
Benford's law	P(d)=〖log〗_10 (1+1/d)	P(d): first digit frequency; d: first digit	Naturalness of information
Mandelbrot’s law	F(q,τ)∝L^(qH(q))	F(q, τ): long-range correlation; L: text scales; q: order	Self-similarity of language

Here are three demonstration examples
'''

def zipfs_law(text):
    words = text.lower().split()
    freq = Counter(words)
    freq_values = np.array(sorted(freq.values(), reverse=True))
    ranks = np.arange(1, len(freq_values) + 1)
    log_freq = np.log(freq_values)
    log_rank = np.log(ranks)

    slope, intercept, r_value, p_value, std_err = linregress(log_rank, log_freq)
    r_squared = r_value ** 2
    return r_squared

def heaps_law(text):
    words = text.lower().split()
    vocab = set()
    N_values = []
    V_values = []
    total_words = 0
    step_size = max(1, len(words) // 1000)
    for i in range(0, len(words), step_size):
        chunk = words[i:i + step_size]
        total_words += len(chunk)
        vocab.update(chunk)
        N_values.append(total_words)
        V_values.append(len(vocab))

    N_values = np.array(N_values)
    V_values = np.array(V_values)
    nonzero_indices = N_values > 0
    N_values = N_values[nonzero_indices]
    V_values = V_values[nonzero_indices]

    log_N = np.log(N_values)
    log_V = np.log(V_values)

    slope, intercept, r_value, p_value, std_err = linregress(log_N, log_V)
    r_squared = r_value ** 2
    return r_squared

def taylors_law(text):
    words = text.lower().split()
    segment_size = max(1, len(words) // 10)
    means = []
    variances = []
    for i in range(0, len(words), segment_size):
        segment = words[i:i + segment_size]
        freq = Counter(segment)
        counts = np.array(list(freq.values()))
        if len(counts) > 1:
            mean = np.mean(counts)
            variance = np.var(counts)
            if mean > 0 and variance > 0:
                means.append(mean)
                variances.append(variance)

    means = np.array(means)
    variances = np.array(variances)
    log_mean = np.log(means)
    log_variance = np.log(variances)
    slope, intercept, r_value, p_value, std_err = linregress(log_mean, log_variance)
    r_squared = r_value ** 2
    return r_squared

def dynamic_law_selector(text):
    r_squared_values = {}
    r_squared_values["Zipf's Law"] = zipfs_law(text)
    r_squared_values["Heaps' Law"] = heaps_law(text)
    r_squared_values["Taylor's Law"] = taylors_law(text)


    applicable_laws = [law for law, r2 in r_squared_values.items() if r2 >= 0.9]
    if applicable_laws:
        selected_laws = applicable_laws
    else:

        max_r2 = max(r_squared_values.values())
        selected_laws = [law for law, r2 in r_squared_values.items() if r2 == max_r2]
    return selected_laws, r_squared_values

def main():
    text = ""
    selected_laws, r2_values = dynamic_law_selector(text)
    print("Selected Law(s):")
    for law in selected_laws:
        print(f"{law} (R² = {r2_values[law]:.4f})")
    print("\nAll R² Values:")
    for law, r2 in r2_values.items():
        print(f"{law}: R² = {r2:.4f}")

if __name__ == "__main__":
    main()
