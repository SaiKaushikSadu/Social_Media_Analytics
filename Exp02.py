import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('./yt_womens_safety.csv')

df = df.drop(columns=['top_comments'])
df = df.dropna(subset=['title','published_at','channel_id','country','transcript'])
df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')

df ['country'] = df['country'].str.lower() # convert to lower case

df ['likes'] = df['likes'].astype(int)
df ['views'] = df['views'].astype(int)
df ['comments_count'] = df['comments_count'].astype(int)

location_engagement = df.groupby('country').agg(
    total_likes = ('likes','sum'),
    total_views = ('views','sum'),
    total_comment = ('comments_count','sum'),
    avg_likes = ('likes','mean'),
    avg_views = ('views','mean'),
    avg_comment = ('comments_count','mean'),
    max_likes = ('likes','max'),
    max_views = ('views','max'),
    max_comment = ('comments_count','max'),
    min_likes = ('likes','min'),
    min_views = ('views','min'),
    min_comment = ('comments_count','min'),
)

# Alternative
# df['sum_likes'] = df.groupby('country')['likes'].transform('sum')

location_engagement.reset_index(drop=False, inplace=True)

print(location_engagement.to_string(index=False))

# Frequency
total = df['country'].value_counts()
top_locs = total.head(5) # top 5 locs

plt.figure(figsize = (10,10))
top_locs.plot(kind='bar')
plt.show()

# Barplot
plt.figure(figsize = (10,10))
sns.barplot(x='total_likes',y='country',data = location_engagement)
plt.xlabel('Total Likes')
plt.ylabel('Country')
plt.show()

# Alternative
# plt.figure(figsize = (10,10))
# sns.barplot(x='sum_likes',y='country',data = df)
# plt.xlabel('Total Likes')
# plt.ylabel('Country')
# plt.show()

# Stacked Bar Plot
location_engagement.set_index('country')[['avg_views','max_views','min_views']].plot(kind='bar',stacked=True,figsize=(10,10))
plt.xlabel('Country')
plt.ylabel('Views Metrics')
plt.show()

# Scatter plot between avg and max likes
plt.figure(figsize=(10, 6))
sns.scatterplot(x='avg_likes', y='max_likes', data=location_engagement)
plt.title('Average Likes vs Max Likes per Country')
plt.xlabel('Average Likes')
plt.ylabel('Max Likes')
plt.show()

# Correlation heatmap between Avg, Max, and Min values
correlation_matrix = location_engagement[['avg_likes', 'max_likes', 'min_likes']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation between Avg, Max, and Min Likes per Country')
plt.show()

df.head().to_html('Exp02.html')