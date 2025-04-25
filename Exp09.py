# competitor analysis
import pandas as pd
from skimage.color.rgb_colors import green
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')

sample_data = [
    {"brand": "BrandA", "followers": 50000, "posts": 120, "likes": 1500, "comments": 100, "text": "Absolutely love their latest campaign!"},
    {"brand": "BrandB", "followers": 80000, "posts": 200, "likes": 1100, "comments": 85, "text": "Not impressed with the new collection."},
    {"brand": "BrandC", "followers": 60000, "posts": 150, "likes": 1700, "comments": 130, "text": "They really care about sustainability."},
    {"brand": "BrandA", "followers": 50000, "posts": 120, "likes": 1500, "comments": 100, "text": "Their ads are everywhere, kind of annoying."},
    {"brand": "BrandB", "followers": 80000, "posts": 200, "likes": 1100, "comments": 85, "text": "Great customer service experience!"},
    {"brand": "BrandC", "followers": 60000, "posts": 150, "likes": 1700, "comments": 130, "text": "Too pricey for what they offer."},
    {"brand": "BrandA", "followers": 50000, "posts": 120, "likes": 1500, "comments": 100, "text": "Innovative and fresh ideas every time."},
    {"brand": "BrandB", "followers": 80000, "posts": 200, "likes": 1100, "comments": 85, "text": "I'm neutral about their latest product."},
    {"brand": "BrandC", "followers": 60000, "posts": 150, "likes": 1700, "comments": 130, "text": "Packaging was amazing and eco-friendly."},
    {"brand": "BrandA", "followers": 50000, "posts": 120, "likes": 1500, "comments": 100, "text": "Not worth the hype."}
]

df = pd.DataFrame(sample_data)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocessing(text):
    text = text.lower()

    tokens = word_tokenize(text)
    preprocessed_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word)>2]
    return ' '.join(preprocessed_tokens)

def polarity_scores(text):
    return TextBlob(text).sentiment.polarity

def sentiment_analysis(score):
    if score >= 0:
        return 'positive'
    elif score < 0:
        return 'negative'
    else:
        return 'neutral'
df ['preprocessed_text'] = df['text'].apply(preprocessing)
df ['polarity'] = df['text'].apply(polarity_scores)
df ['sentiments'] = df['polarity'].apply(sentiment_analysis)

# compare followers
df ['avg_likes'] = df['likes'].mean()
df ['avg_comments'] = df['comments'].mean()
df ['avg_posts'] = df['posts'].mean()
df ['total_followers'] = df['followers'].sum()

df ['engagement_rate'] = (df['likes'] + df['comments'] +df['posts']) / df['followers']

sentiment_dist = df.groupby(['brand','sentiments']).size().unstack(fill_value=0).fillna(0)
print(sentiment_dist)

sns.barplot(x='brand', y='total_followers', data=df)
plt.show()

sns.barplot(x='brand', y='avg_comments', data=df)
plt.show()

sns.barplot(x='brand', y='avg_likes', data=df)
plt.show()

sns.barplot(x='brand', y='avg_posts', data=df)
plt.show()

sns.lineplot(x='brand', y='engagement_rate', data=df)
plt.show()

sns.countplot(x='brand', hue='sentiments', palette={'positive': 'green', 'negative': 'red'}, data=df)
plt.show()

df.head(10).to_html('Exp09.html')