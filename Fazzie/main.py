import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_df = pd.read_csv('./train_set.csv/train_set.csv', sep='\t', nrows=100)
train_df['text_len'] = train_df['text'].apply(lambda x: len(x.split(' ')))
print(train_df['text_len'].describe())

plt.hist(train_df['text_len'], bins=200)
plt.xlabel('Text char count')
plt.title("Histogram of char count")

train_df['label'].value_counts().plot(kind='bar')
plt.title('News class count')
plt.xlabel("category")
plt.show()