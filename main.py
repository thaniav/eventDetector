import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import adjusted_rand_score
from gensim import corpora, models
import matplotlib.pyplot as plt

# Load
events_df = pd.read_csv('3465974/events.csv')
tweets_df = pd.read_csv('3465974/tweets.csv')


data = pd.merge(tweets_df, events_df, on='event_ID')

# Preprocessing
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(data['tweets'])


num_topics = 10
corpus = corpora.MmCorpus(vectorizer.transform(data['tweets']))
lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=vectorizer.get_feature_names())


topic_distributions = [lda_model.get_document_topics(doc) for doc in corpus]


topic_significance_scores = [max(topic_dist, key=lambda x: x[1])[1] for topic_dist in topic_distributions]

# K - Means
kmeans = KMeans(n_clusters=20, random_state=0)
kmeans_labels = kmeans.fit_predict(np.array(topic_significance_scores).reshape(-1, 1))


# DBSCAN Clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X)


cluster_labels = np.loadtxt('cluster_labels.txt', dtype=int)
time_resolutions = np.loadtxt('time_resolutions.txt')

ari_kmeans = adjusted_rand_score(cluster_labels, kmeans_labels)
ari_dbscan = adjusted_rand_score(cluster_labels, dbscan_labels)

print(f"Adjusted Rand Index for K-Means: {ari_kmeans}")
print(f"Adjusted Rand Index for DBSCAN: {ari_dbscan}")

event_dates = pd.to_datetime(events_df['date'])
plt.figure(figsize=(10, 6))
for topic in range(num_topics):
    topic_evolution = [topic_dist[topic][1] for topic_dist in topic_distributions]
    plt.plot(event_dates, topic_evolution, label=f"Topic {topic}")
plt.xlabel("Date")
plt.ylabel("Topic Significance")
plt.legend()
plt.title("Temporal Topic Evolution")
plt.show()
