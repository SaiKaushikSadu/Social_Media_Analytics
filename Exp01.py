import pandas as pd

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

# Topic Modelling
from sklearn.decomposition import LatentDirichletAllocation
import seaborn as sns

df = pd.read_csv('./yt_womens_safety.csv')

# Analyze dataset
# print(df.shape)
# print(df.columns)
# print(df.isnull().sum())
# print(df.info())
# print(df.describe())

# sample_data = df.head().to_dict(orient='records')
# print(sample_data)

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

df['preprocessed_text'] = df['transcript'].apply(preprocess)

# Keyword extraction
tfidf_score = dict(zip(
    TfidfVectorizer(max_features=50).fit(df['preprocessed_text']).get_feature_names_out(),
    TfidfVectorizer(max_features=50).fit_transform(df['preprocessed_text']).sum(axis=0).A1
))

print("----------- Keyword Extraction -----------")
for word, score in sorted(tfidf_score.items(),key=lambda x:x[1], reverse=True)[:10]:
    print(f"{word} : {score:.2f}")

# Word Cloud for keywords
wordcloud = WordCloud(width=800, height=500, background_color='white').generate_from_frequencies(tfidf_score)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Keyword Extraction Word Cloud")
plt.show()

# Topic Modelling (LDA)
count_vector = CountVectorizer(max_features=500, stop_words='english')
X_count = count_vector.fit_transform(df['preprocessed_text'])

lda = LatentDirichletAllocation(random_state=42, n_components=5)
lda.fit(X_count)

keywords = count_vector.get_feature_names_out()

for i, topic in enumerate(lda.components_):
    print(f"Topic {i+1}")
    print([keywords[key] for key in topic.argsort()[-10:]])

for i, topic in enumerate(lda.components_):
    plt.figure()
    terms = [keywords[index] for index in topic.argsort()[-10:]]
    scores = topic[topic.argsort()[-10:]]
    sns.barplot(x=terms, y=scores, palette='viridis')
    plt.show()

sample_data_html = df.head().to_html('Exp01.html')