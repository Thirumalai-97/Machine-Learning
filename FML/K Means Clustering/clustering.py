from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans

# Sample documents and their manually selected keywords
articles = [ "n1.txt", "n2.txt"," n3.txt", "n4.txt", "n5.txt"]

# Manually selected keywords for each document

keywords=["Parliment","India","Pakistan", "September", "Congress"]

# Create a vocabulary from all the keywords
vocabulary = set(word for keywords_doc in keywords for word in keywords_doc)

# Create a CountVectorizer for BoW representation
vectorizer = CountVectorizer(vocabulary=vocabulary)

# Transform documents into BoW representations
X = vectorizer.transform(articles)

# Cluster the documents using K-Means
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)

# Assign documents to clusters
clusters = kmeans.labels_

# Print the results
for i, doc in enumerate(clusters):
    print(f"Articles {i + 1} belongs to Cluster {clusters[i] + 1}:")
    print(doc)
    print()

# Observations and insights can be drawn from the cluster assignments
