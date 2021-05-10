import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

dataset = pd.read_csv('restaurant_reviews.tsv', delimiter='\t', quoting=3)

nltk.download('stopwords')
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')

# cleaning the text
corpus = []
for reviews in dataset['Review']:
    review = re.sub('[^a-zA-Z]', ' ', reviews)  # removing all special characters except a-z & A-Z
    review = review.lower().split()
    review = [ps.stem(j) for j in review if not j in set(all_stopwords)]       # stemming the verbs only
    corpus.append(' '.join(review))

# creating a bag of words
cv = CountVectorizer(max_features=1500)
x = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.18, random_state=0)

classifier = GaussianNB()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix is:', cm)

acc = accuracy_score(y_test, y_pred)
print('Precision score of the model is:', acc)