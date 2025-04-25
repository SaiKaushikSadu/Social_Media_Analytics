import pandas as pd
import numpy as np

# For preprocessing
import nltk
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# For Keyword Extraction
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from wordcloud import WordCloud
from matplotlib import pyplot as plt

from textblob import TextBlob

df = pd.read_csv('./yt_womens_safety.csv')

# Data cleaning
# Convert the createdAt/published_At to date time format
df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')

# Handle missing values
# Drop top comments column
df = df.drop(columns=['top_comments']) # assign to df or use inplace = True

# Drop important rows where the value is empty
df = df .dropna(subset=['title','published_at','channel_id','query','country','transcript'])

# Add 0 to missing counts like likes, etc.
df['likes'] = df['likes'].fillna(0)
df['views'] = df['views'].fillna(0)
df['comments_count'] = df['comments_count'].fillna(0)

#Preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    #Remove URL's, mentions, hastags, symbols, numbers, special char etc...
    text = re.sub(r"http\S+|https\S+|www\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"\d+", "", text)

    # Tokenize
    tokens = word_tokenize(text)
    # Remove Stopwords
    preprocessed_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word)>2]
    return ' '.join(preprocessed_tokens)

def polarisation_score(text):
    return TextBlob(text).sentiment.polarity

def sentiment_analysis(polarity):
    if polarity > 0:
        return 'positive'
    elif polarity < 0:
        return 'negative'
    else:
        return 'neutral'

df['preprocessed_text'] = df['transcript'].apply(preprocess)
df ['polarity'] = df['preprocessed_text'].apply(polarisation_score)
df ['sentiment'] = df['polarity'].apply(sentiment_analysis)

top_positive = df[df['sentiment'] == 'positive'].sort_values(by = 'polarity', ascending = False).head(10)
top_negative = df[df['sentiment'] == 'negative'].sort_values(by = 'polarity', ascending = False).head(10)

positive_text = " ".join(df[df['sentiment'] == 'positive']['preprocessed_text'])
negative_text = " ".join(df[df['sentiment'] == 'negative']['preprocessed_text'])

positive_word_cloud = WordCloud(width=800, height=400).generate(positive_text)
negative_word_cloud = WordCloud(width=800, height=400).generate(negative_text)

plt.figure(figsize=(100,100))
plt.imshow(positive_word_cloud)
plt.show()

plt.figure(figsize=(100,100))
plt.imshow(negative_word_cloud)
plt.show()

df.head().to_html('Exp08.html')