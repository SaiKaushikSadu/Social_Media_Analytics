from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import re
from wordcloud import WordCloud
import nltk

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

df = pd.read_csv('./x_sentiment.csv')

def extract_hashtags(text):
    text = text.lower()
    hashtags = re.findall(r"#\w+", text)
    return ' '.join(hashtags)

df['hashtags'] = df['tweet'].apply(extract_hashtags)

all_hashtags = ' '.join(df['hashtags']).split()
hashtag_counts = Counter(all_hashtags)

# Convert to DataFrame for easy viewing and plotting - Important
hashtag_freq_df = pd.DataFrame(hashtag_counts.items(), columns=['Hashtag', 'Frequency'])
hashtag_freq_df = hashtag_freq_df.sort_values(by='Frequency', ascending=False).reset_index(drop=True)[:5]

hashtags_text = ' '.join(df['hashtags'])

wordcloud = WordCloud(width=800, height=400, background_color='white', regexp=r'#\w+').generate(hashtags_text)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Hashtag Word Cloud', fontsize=18)
plt.show()

plt.figure(figsize=(12, 6))
plt.bar(hashtag_freq_df['Hashtag'], hashtag_freq_df['Frequency'])
plt.title('Hashtag Frequency', fontsize=18)
plt.show()

print(df.head())