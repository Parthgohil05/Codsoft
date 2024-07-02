import string
import numpy as np
import pandas as pd
import streamlit as st

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')

"""# **Importing library**"""

# Streamlit code to display the Python script
st.code("""
import string
import numpy as np
import pandas as pd
import streamlit as st

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

nltk.download('stopwords')
nltk.download('punkt')
""")

"""# **Read CSV File**"""
df = pd.read_csv("spam.csv", encoding='ISO-8859-1')

st.code("""
df = pd.read_csv("spam.csv", encoding='ISO-8859-1')
""")
st.write(df.head())

"""**✅ Removed Unnamed columns from the dataset**"""
df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)

st.code("""
df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
""")
st.write(df.head())

"""**✅ Handling the missing and duplicate data**"""
df = df.drop_duplicates(keep='first')

st.code("""
df = df.drop_duplicates(keep='first')
""")
st.write(df)

"""# **Label Encoding**"""
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['v1'] = encoder.fit_transform(df['v1'])

st.code("""
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['v1'] = encoder.fit_transform(df['v1'])
""")
st.write(df.head())

"""**✅ Text Preprocessing: Remove special characters, punctuation, convert to lowercase, remove stopwords, and perform stemming.**"""
stemmer = PorterStemmer()
stopwords_set = set(stopwords.words('english'))
corpus = []

for text in df['v2']:
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = word_tokenize(text)
    text = [stemmer.stem(word) for word in text if word not in stopwords_set]
    corpus.append(' '.join(text))

df['v2'] = corpus

st.code("""
stemmer = PorterStemmer()
stopwords_set = set(stopwords.words('english'))
corpus = []

for text in df['v2']:
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = word_tokenize(text)
    text = [stemmer.stem(word) for word in text if word not in stopwords_set]
    corpus.append(' '.join(text))

df['v2'] = corpus
""")
st.write(df.head())

X = df['v2']
y = df['v1']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.code("""
X = df['v2']
y = df['v1']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
""")
st.text("Shapes of X_train, X_test, y_train, y_test:")
st.write(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

st.code("""
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)
""")
st.write(X_train.shape, X_test.shape)

"""# **Naive Bayes Classifier**"""
clf = MultinomialNB()
clf.fit(X_train, y_train)

y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)

train_accuracy = accuracy_score(y_train, y_pred_train)
train_precision = precision_score(y_train, y_pred_train, average='weighted')
train_recall = recall_score(y_train, y_pred_train, average='weighted')
train_f1 = f1_score(y_train, y_pred_train, average='weighted')
train_confusion = confusion_matrix(y_train, y_pred_train)

test_accuracy = accuracy_score(y_test, y_pred_test)
test_precision = precision_score(y_test, y_pred_test, average='weighted')
test_recall = recall_score(y_test, y_pred_test, average='weighted')
test_f1 = f1_score(y_test, y_pred_test, average='weighted')
test_confusion = confusion_matrix(y_test, y_pred_test)

st.code("""
clf = MultinomialNB()
clf.fit(X_train, y_train)

y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)

train_accuracy = accuracy_score(y_train, y_pred_train)
train_precision = precision_score(y_train, y_pred_train, average='weighted')
train_recall = recall_score(y_train, y_pred_train, average='weighted')
train_f1 = f1_score(y_train, y_pred_train, average='weighted')
train_confusion = confusion_matrix(y_train, y_pred_train)

test_accuracy = accuracy_score(y_test, y_pred_test)
test_precision = precision_score(y_test, y_pred_test, average='weighted')
test_recall = recall_score(y_test, y_pred_test, average='weighted')
test_f1 = f1_score(y_test, y_pred_test, average='weighted')
test_confusion = confusion_matrix(y_test, y_pred_test)
""")
st.text("Training and Test Metrics:")
st.write(f"Training Accuracy: {train_accuracy}")
st.write(f"Test Accuracy: {test_accuracy}")
st.write(f"Training Precision: {train_precision}")
st.write(f"Test Precision: {test_precision}")
st.write(f"Training Recall: {train_recall}")
st.write(f"Test Recall: {test_recall}")
st.write(f"Training F1 Score: {train_f1}")
st.write(f"Test F1 Score: {test_f1}")

"""# **Output** """
selected_sms = st.selectbox("Select an SMS from the dataset:", df['v2'])

if st.button("Check"):
    processed_input = selected_sms.lower()
    processed_input = processed_input.translate(str.maketrans('', '', string.punctuation))
    processed_input = word_tokenize(processed_input)
    processed_input = [stemmer.stem(word) for word in processed_input if word not in stopwords_set]
    processed_input = ' '.join(processed_input)
    
    input_vectorized = vectorizer.transform([processed_input])
    prediction = clf.predict(input_vectorized)
    
    if prediction[0] == 1:
        st.success("The message is classified as: **Spam**")
    else:
        st.success("The message is classified as: **Not Spam**")
