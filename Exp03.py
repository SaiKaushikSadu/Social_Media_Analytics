import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('./yt_womens_safety.csv')

df = df.drop(columns=['top_comments'])
df = df.dropna(subset=['title', 'published_at', 'channel_id', 'country', 'transcript'])
df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
df['country'] = df['country'].str.lower()
df['likes'] = df['likes'].astype(int)
df['views'] = df['views'].astype(int)
df['comments_count'] = df['comments_count'].astype(int)

df['published_date'] = df['published_at'].dt.date
df['month'] = df['published_at'].dt.to_period('M').astype(str)
df['week'] = df['published_at'].dt.to_period('W').astype(str)
df['day_of_week'] = df['published_at'].dt.day_name()

daily_trend = df.groupby('published_date')[['likes', 'views', 'comments_count']].sum().reset_index()

weekly_trend = df.groupby('week')[['likes', 'views', 'comments_count']].sum().reset_index()

monthly_trend = df.groupby('month')[['likes', 'views', 'comments_count']].sum().reset_index()

dayofweek_trend = df.groupby('day_of_week')[['likes', 'views', 'comments_count']].mean().reindex(
    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']).reset_index()

# Plot function
def plot_trend(data, x_col, title):
    melted = data.melt(id_vars=x_col, var_name='Metric', value_name='Count')
    sns.lineplot(data=melted, x=x_col, y='Count', hue='Metric', marker='o')
    plt.show()

plot_trend(daily_trend, 'published_date', 'Daily')
plot_trend(weekly_trend, 'week', 'Weekly')
plot_trend(monthly_trend, 'month', 'Monthly')
plot_trend(dayofweek_trend, 'day_of_week', 'Average per Day of Week')

# sns.lineplot(x='published_date', y='likes', data=df)
# plt.show()
# sns.lineplot(x='published_date', y='comments_count', data=df)
# plt.show()
# sns.lineplot(x='published_date', y='views', data=df)
# plt.show()

df.head().to_html('Exp03.html')