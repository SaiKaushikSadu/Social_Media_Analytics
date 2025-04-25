import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import re
from networkx.algorithms.community import girvan_newman

# Load dataset
df = pd.read_csv("./exp10_csv.csv")

# Step 1: Extract brand mentions
def extract_brands(text):
    return re.findall(r'brand\w+', str(text).lower())  # Looks for words like 'brandA', 'brandB'

df['brands_mentioned'] = df['text'].apply(extract_brands)

# Step 2: Create user-brand matrix
user_brand_pairs = df.explode('brands_mentioned')[['username', 'brands_mentioned']].dropna()

# Step 3: Build user-user edges if they mentioned the same brand
edges = []
brand_groups = user_brand_pairs.groupby('brands_mentioned')['username'].apply(list)

for users in brand_groups:
    for i in range(len(users)):
        for j in range(i + 1, len(users)):
            edges.append((users[i], users[j]))

# Step 4: Create Graph
G = nx.Graph(edges)
G.add_edges_from(edges)

# Step 5: Community Detection using Girvan-Newman
communities = girvan_newman(G)
top_level_communities = next(communities)
community_list = [list(c) for c in top_level_communities]

print("Detected Communities:")
for i, community in enumerate(community_list, 1):
    print(f"Community {i}: {community}")

# Step 6: Visualization
pos = nx.spring_layout(G, seed=42)
colors = ['red', 'blue', 'green', 'purple']

plt.figure(figsize=(10, 6))
for i, community in enumerate(community_list):
    nx.draw_networkx_nodes(G, pos, nodelist=community, node_color=colors[i % len(colors)], label=f'Community {i+1}')
nx.draw_networkx_edges(G, pos)
nx.draw_networkx_labels(G, pos)
plt.show()

# Step 7: Influencer Analysis
centrality = nx.degree_centrality(G)
top_influencers = sorted(centrality.items(), key=lambda x: x[1], reverse=True)

print("\nTop Influencers by Degree Centrality:")
for user, score in top_influencers[:5]:
    print(f"{user}: {score:.3f}")

df.head().to_html('Exp10.html')