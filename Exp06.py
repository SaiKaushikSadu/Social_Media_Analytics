import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import seaborn as sns
from nltk import WordNetLemmatizer, word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob

df = pd.read_csv('./yt_womens_safety.csv')
df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
df = df.drop(columns=['top_comments']) # assign to df or use inplace = True
df = df .dropna(subset=['title','published_at','channel_id','query','country','transcript'])
df['likes'] = df['likes'].fillna(0)
df['views'] = df['views'].fillna(0)
df['comments_count'] = df['comments_count'].fillna(0)

#Preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|https\S+|www\S+","",text)
    text = re.sub(r"[^a-z\s]","",text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"\d+","",text)

    tokens = word_tokenize(text)
    preprocessed_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word)>2]

    return ' '.join(preprocessed_tokens)

def polarity(text):
    return TextBlob(text).sentiment.polarity

def sentiment_classify(polarity_score):
    if polarity_score > 0:
        return 'positive'
    elif polarity_score < 0:
        return 'negative'
    else:
        return 'neutral'

df ['preprocessed_text'] = df['transcript'].apply(preprocess)
df ['polarity'] = df['preprocessed_text'].apply(polarity)
df ['sentiment'] = df ['polarity'].apply(sentiment_classify)

# likes, comments, views - heatmap / corr
sns.heatmap(df[['likes','comments_count','views']].corr(), annot=True)
plt.show()

# likes vs sentiment -  boxplot
sns.boxplot(x="sentiment", y="likes", data=df)
plt.show()

# total length vs likes
df ['text_length'] = df['preprocessed_text'].str.len()
sns.scatterplot(x='text_length',y='likes',data=df)
plt.show()

# country vs likes
sns.barplot(x='country',y='likes',data=df)
plt.show()

# likes - histogram
sns.histplot(x='likes',data=df)
plt.show()

# Engagement Rate - Likes+Comments+Tweets / Followers
# engagement_rate = (df['likes']+df['comments']+df['tweets']) / df['followers']

df.head().to_html('Exp06.html')