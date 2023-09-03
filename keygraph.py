from keybert import KeyBERT
from nltk import word_tokenize, PorterStemmer
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
import pandas as pd

events_df = pd.read_csv('3465974/events.csv')
tweets_df = pd.read_csv('3465974/tweets.csv')


def preprocess_text(word):
    # Lowercase the text
    word = word.lower()

    # Tokenization
    tokens = word_tokenize(word)

    # Remove punctuation and non-alphabetic characters
    tokens = [token for token in tokens if token.isalpha()]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]

    # Join tokens back into a preprocessed text
    preprocessed = ' '.join(stemmed_tokens)

    return preprocessed


# Load your dataset and preprocess text data (replace with actual data loading)
text_data = pd.read_csv('your_text_data.csv')
preprocessed_text = preprocess_text(text_data['text_column'])

# Initialize the KeyBERT model
model = KeyBERT()

# Extract keywords for each text document
document_keywords = []
for text in preprocessed_text:
    keywords = model.extract_keywords(text, keyphrase_ngram_range=(1, 2), top_n=5)
    document_keywords.append(keywords)

# Perform K-means clustering on extracted keywords
num_clusters = 10  # Number of clusters to create
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
clusters = kmeans.fit_predict(document_keywords)

# Associate clusters with events based on event IDs
events_df['cluster'] = clusters

# Print results or further analyze clusters
print(events_df[['event_ID', 'cluster']])
