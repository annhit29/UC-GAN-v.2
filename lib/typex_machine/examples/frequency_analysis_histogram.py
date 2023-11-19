#!/usr/bin/env python3
import sys
import matplotlib.pyplot as plt
import numpy as np
from typex.encryptor import Encryptor

plt.rcdefaults()
input_text = sys.stdin.read()
letter_appearances = {}

for char in Encryptor.ALPHABET:
    letter_appearances[char] = 0

for char in input_text:
    if char.upper() in Encryptor.ALPHABET:
            letter_appearances[char.upper()] += 1

sorted_letters = sorted(
        letter_appearances.keys(),
        key=lambda letter: letter_appearances[letter], reverse=True)
sorted_appearances = sorted(letter_appearances.values(), reverse=True)

y_pos = np.arange(len(sorted_appearances))

plt.bar(y_pos, sorted_appearances, align='center', alpha=1)
plt.xticks(y_pos, sorted_letters)
plt.ylabel('Number of occurances')
plt.xlabel('Letter')
plt.title('Letter Frequency')
plt.show()
