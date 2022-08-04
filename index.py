import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import _stop_words
from nltk.corpus import names
from nltk.stem import WordNetLemmatizer
from sklearn.manifold import TSNE


#Importing the data
groups = fetch_20newsgroups()
groups.keys()
groups['target_names']

#Plotting the distribution of topics
sns.displot(groups.target, kde=True)
plt.title('Topics Distribution')
plt.xlabel('Topic')
plt.ylabel('Count')
plt.show()


groups.data[0]
groups.target_names[groups.target[0]] #name of category to which the first document belongs to


#Get the top 500 features and dropping stop words
cv_stop = CountVectorizer(stop_words='english', max_features=500)


#Retaining letter-only words
def is_letter_only(word):
    for char in word:
        if not char.isalpha():
            return False
    return True

data_cleaned = []
names_list = set(names.words())
lemmatizer = WordNetLemmatizer()

#Stemming and Lemmatizing
for doc in groups.data:
    data_cleaned.append(' '.join([lemmatizer.lemmatize(word.lower())
                             for word in doc.split()
                             if is_letter_only(word)
                             and word not in names_list]))

data_cleaned_count = cv_stop.fit_transform(data_cleaned)
print(cv_stop.get_feature_names_out())

#Dimensionality Reduction with t-SNE
tsne_model = TSNE(n_components=2, perplexity=40, random_state=42, learning_rate=500)
categories_5 = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware','comp.windows.x']
groups_5 = fetch_20newsgroups(categories=categories_5)
count_vector_sw = CountVectorizer(stop_words="english", max_features=500)
data_cleaned = []

#Reducing to 5 related topics
for doc in groups_5.data:
    doc = doc.lower()
    doc_cleaned = ' '.join(lemmatizer.lemmatize(word) for word in doc.split() if is_letter_only(word) and word not in names_list)
    data_cleaned.append(doc_cleaned)

data_cleaned_count_5 = count_vector_sw.fit_transform(data_cleaned)
data_tsne = tsne_model.fit_transform(data_cleaned_count_5.toarray())
plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=groups_5.target)
plt.title('Scatter Plot')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.colorbar()
plt.show()