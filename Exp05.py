import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import seaborn as sns
from nltk import WordNetLemmatizer, word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
from wordcloud import WordCloud

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

counts = df['sentiment'].value_counts()
percent = counts / counts.sum() * 100
print(f"Positive : {percent.get('positive'):.2f}%")
print(f"Negative : {percent.get('negative'):.2f}%")
print(f"Neutral : {percent.get('neutral'):.2f}%")

# Top sentiments
top_positive = df[df['sentiment']=='positive'].sort_values(by='polarity', ascending=False).head(10)
top_negative = df[df['sentiment']=='negative'].sort_values(by='polarity', ascending=False).head(10)

print(top_positive)
print(top_negative)

positive_text = ' '.join(df[df['sentiment']=='positive']['preprocessed_text'])
negative_text = ' '.join(df[df['sentiment']=='negative']['preprocessed_text'])

positive_word_cloud = WordCloud(width=800, height=400, background_color='white',colormap='Greens',max_words=50).generate(positive_text)
negative_word_cloud = WordCloud(width=800, height=400, background_color='white',colormap='Reds',max_words=50).generate(negative_text)

plt.imshow(positive_word_cloud)
plt.show()

plt.imshow(negative_word_cloud)
plt.show()

plt.figure(figsize = (10,10))
plt.pie(counts,labels=counts.index, autopct='%1.1f%%')
plt.show()

plt.figure(figsize=(8, 6))
sns.barplot(x=counts.index, y=counts.values)
plt.title('Sentiment Analysis Bar Chart')
plt.xlabel('Sentiment')
plt.ylabel('Number of Transcripts')
plt.show()

df.head().to_html('Exp05.html')