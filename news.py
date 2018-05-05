# ## Import Libraries
# Loading all libraries to be used
import copy
import numpy as np
import re
import nltk
nltk.download('stopwords')
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import SMOTE
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import accuracy_score
#from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# # Data preparation
# ## Load data
titles = []
categories = []
print("\nLoading data")
with open('dsjVoxArticles.tsv','r') as tsv:
    count = 0;
    for line in tsv:
        a = line.strip().split('\t')[:3]
        if a[2] in ['Business & Finance', 'Health Care', 'Science & Health', 'Politics & Policy', 'Criminal Justice']:
            title = a[0].lower()
            title = re.sub('\s\W',' ',title)
            title = re.sub('\W\s',' ',title)
            titles.append(title)
            categories.append(a[2])

# ## Split data
print("\nSplitting data")
title_tr, title_te, category_tr, category_te = train_test_split(titles,categories)
title_tr, title_de, category_tr, category_de = train_test_split(title_tr,category_tr)
print("Training: ",len(title_tr))
print("Developement: ",len(title_de),)
print("Testing: ",len(title_te))




# # Data Preprocessing
# ## Vectorization of data
# Vectorize the data using Bag of words (BOW)
print("\nVectorizing data")
tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")
stop_words = nltk.corpus.stopwords.words("english")
vectorizer = CountVectorizer(tokenizer=tokenizer.tokenize, stop_words=stop_words)

vectorizer.fit(iter(title_tr))
Xtr = vectorizer.transform(iter(title_tr))
Xde = vectorizer.transform(iter(title_de))
Xte = vectorizer.transform(iter(title_te))

encoder = LabelEncoder()
encoder.fit(category_tr)
Ytr = encoder.transform(category_tr)
Yde = encoder.transform(category_de)
Yte = encoder.transform(category_te)

# ## Feature Reduction
# We can check the variance of the feature and drop them based on a threshold
print("\nApplyting Feature Reduction")
print("Number of features before reduction : ", Xtr.shape[1])
selection = VarianceThreshold(threshold=0.001)
Xtr_whole = copy.deepcopy(Xtr)
Ytr_whole = copy.deepcopy(Ytr)
selection.fit(Xtr)
Xtr = selection.transform(Xtr)
Xde = selection.transform(Xde)
Xte = selection.transform(Xte)
print("Number of features after reduction : ", Xtr.shape[1])

# ## Sampling data
sm = SMOTE(random_state=42)
Xtr, Ytr = sm.fit_sample(Xtr, Ytr)




# # Train Models
# ### Baseline Model
# “stratified”: generates predictions by respecting the training set’s class distribution.
print("\n\nTraining baseline classifier")
dc = DummyClassifier(strategy="stratified")
dc.fit(Xtr, Ytr)
pred = dc.predict(Xde)
print(classification_report(Yde, pred, target_names=encoder.classes_))

# ### Decision Tree
print("Training Decision tree")
dt = DecisionTreeClassifier()
dt.fit(Xtr, Ytr)
pred = dt.predict(Xde)
print(classification_report(Yde, pred, target_names=encoder.classes_))

# ### Random Forest
print("Training Random Forest")
rf = RandomForestClassifier(n_estimators=40)
rf.fit(Xtr, Ytr)
pred = rf.predict(Xde)
print(classification_report(Yde, pred, target_names=encoder.classes_))

# ### Multinomial Naive Bayesian
print("Training Multinomial Naive Bayesian")
nb = MultinomialNB()
nb.fit(Xtr, Ytr)
pred_nb = nb.predict(Xde)
print(classification_report(Yde, pred_nb, target_names=encoder.classes_))

# ### Support Vector Classification
print("Training Support Vector Classification")
from sklearn.svm import SVC
svc = SVC()
svc.fit(Xtr, Ytr)
pred = svc.predict(Xde)
print(classification_report(Yde, pred, target_names=encoder.classes_))

# ### Multilayered Perceptron
print("Training Multilayered Perceptron")
mlp = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(100, 20), random_state=1, max_iter=400)
mlp.fit(Xtr, Ytr)
pred = mlp.predict(Xde)
print(classification_report(Yde, pred, target_names=encoder.classes_))




# # Final Model: Multinomial Naive Bayesian
# **Multinomial Naive Bayesian** works the best. Lets run NB on our test data and get the confusion matrix and its heat map.
# ## Predict test data
print("\n\nPredicting test data using Multinomial Naive Bayesian")
pred_final = nb.predict(Xte)
print(classification_report(Yte, pred_final, target_names=encoder.classes_))


# get incorrectly classified data
print("\n\nIncorrectly classified")
incorrect = np.not_equal(pred_nb, Yde).nonzero()[0]
print(
    "\nTitle: ",titles[incorrect[6]],
    "\nTrue Category: ",categories[incorrect[6]],
    "\nPredicted Category: ", encoder.inverse_transform([pred[incorrect[6]]])[0]
)
