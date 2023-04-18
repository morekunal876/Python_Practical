# 6.a Implement the different Distance methods (Euclidean) with Prediction, Test Score and Confusion Matrix


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
#Load the dataset
url = "https://raw.githubusercontent.com/SharmaNatasha/Machine-Learning-using-Python/master/Datasets/IRIS.csv"
df = pd.read_csv(url)
#quick look into the data
df.head(5)
#Separate data and label
x = df.iloc[:,1:4]
y = df.iloc[:,4]
#Prepare data for classification process
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

#Create a model
KNN_Classifier = KNeighborsClassifier(n_neighbors = 6, p = 2, metric='minkowski')

#Train the model
KNN_Classifier.fit(x_train, y_train)
#Let's predict the classes for test data
pred_test = KNN_Classifier.predict(x_test)

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
#Load the dataset
url = "https://raw.githubusercontent.com/SharmaNatasha/Machine-Learning-using-Python/master/Datasets/IRIS.csv"
df = pd.read_csv(url)
#quick look into the data
df.head(5)
#Separate data and label
x = df.iloc[:,1:4].values
#Creating the kmeans classifier
KMeans_Cluster = KMeans(n_clusters = 3)
y_class = KMeans_Cluster.fit_predict(x)

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
corpus = [
 'the brown fox jumped over the brown dog',
 'the quick brown fox',
 'the brown brown dog',
 'the fox ate the dog'
]
query = ["brown"]
X = vectorizer.fit_transform(corpus)
Y = vectorizer.transform(query)

cosine_similarity(Y, X.toarray())
