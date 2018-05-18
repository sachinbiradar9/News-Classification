# News Classification
Classify news into categories based on their headline.  
Various classifiers were tried - Decision Tree, Support Vector Classifier, Multinomial Naive Bayesian Classifier,
Multilayered Perceptron, Random Forest. Multinomial Naive Bayesian Classifier worked the best. It is logical
for Multinomial Naive Bayesian to work the best as even we as humans classify based on keywords. We are
likely to predict “Politics” is we see keywords like Obama, election, republic and we are likely to predict
“Criminal” if we see keywords like drugs, jail and so on. Naive bayesian scans whole dataset and finds the
probabilities of each word in headline being associated with a class and then find the probability for whole
headline hence it works good.

## Installation
`pip install numpy`  
`pip install scikit-learn`   
`pip install imblearn`  
`pip install seaborn`  
Dataset: https://data.world/elenadata/vox-articles

## Usage
You can look for detailed explanation with graphs at - [news.ipynb](news.ipynb)  
You can also run the python file using -  `python3 news.py`

## Credits
[Dr. Elena Zeleva](https://www.cs.uic.edu/~elena/)
