# Developer：Fazzie
# Time: 2021/11/2221:01
# File name: data analyze.py.py
# Development environment: Anaconda Python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

train_df = pd.read_csv('./train_set.csv/train_set.csv', sep='\t', nrows=100)
train_df['text_len'] = train_df['text'].apply(lambda x: len(x.split(' ')))
print(train_df['text_len'].describe())

plt.hist(train_df['text_len'], bins=200)
plt.xlabel('Text char count')
plt.title("Histogram of char count")

train_df['label'].value_counts().plot(kind='bar')
plt.title('News class count')
plt.xlabel("category")

all_lines = ' '.join(list(train_df['text']))
word_count = Counter(all_lines.split(" "))
word_count = sorted(word_count.items(), key=lambda d:d[1], reverse = True)

print(len(word_count))

print(word_count[0])

print(word_count[-1])

train_df['text_unique'] = train_df['text'].apply(lambda x: ' '.join(list(set(x.split(' ')))))
all_lines = ' '.join(list(train_df['text_unique']))
word_count = Counter(all_lines.split(" "))
word_count = sorted(word_count.items(), key=lambda d:int(d[1]), reverse = True)

print(word_count[0])

print(word_count[1])

print(word_count[2])
plt.show()

# home work
# for i in range(1, 100):
#     sentence = train_df['text'][i].count('900') + train_df['text'][i].count('3750') +train_df['text'][i].count('648')
#     print(sentence)

train_df1 = train_df.loc[train_df["label"] == 1]
all_lines = ' '.join(list(train_df1['text']))
word_count = Counter(all_lines.split(" "))
word_count = sorted(word_count.items(), key=lambda d:d[1], reverse = True)

print(len(word_count))

print(word_count[0])
