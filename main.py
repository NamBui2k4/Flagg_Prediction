import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# prepare data
df = pd.read_csv('Data.csv', dtype=str)

# preprocessing
df = df.dropna(axis=1)
df = df.drop('response_id', axis=1)

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')

lemma = nltk.WordNetLemmatizer()

# Use loc to avoid chained assignment warnings
for i in range(len(df)):
    # replace non-alphabetic characters by ' ' and then lowercase
    df.loc[i, 'response_text'] = re.sub(r'[^a-zA-Z]', ' ', df.loc[i, 'response_text']).lower()

    # tokenize
    df.loc[i, 'response_text'] = nltk.word_tokenize(df.loc[i, 'response_text'])
    
    # remove stopwords
    df.loc[i, 'response_text'] = [word for word in df.loc[i, 'response_text']
                                   if word not in set(stopwords.words('english'))]
    
    # lemmatize
    df.loc[i, 'response_text'] = [lemma.lemmatize(word) for word in df.loc[i, 'response_text']]
    
    # join them back into a single string
    df.loc[i, 'response_text'] = " ".join(df.loc[i, 'response_text'])

# encoder  
cv = TfidfVectorizer(stop_words='english')

# split data
X = df['response_text'].tolist()
y = df['class'].tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# fit the vectorizer on the training data and transform the training data
X_train = cv.fit_transform(X_train)

# transform the test data using the already fitted vectorizer
X_test = cv.transform(X_test)

# training with model
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# evaluation
with open('result.csv', 'w') as f:
    for i, j in zip(y_pred, y_test):
        f.write('predicted: {} , ground truth: {} \n'.format(i, j))
         
with open('Evaluate.txt', 'w') as f:
    f.write(classification_report(y_test, y_pred,zero_division=0))

# save
import joblib
joblib.dump(model, "model_saved.pkl") 

