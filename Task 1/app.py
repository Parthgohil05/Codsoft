# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

# Load the dataset
columns = ['Movie Name', 'Genre', 'Description']
df_train = pd.read_csv('train_data.txt', delimiter=':::', engine='python', names=columns, index_col=0)
df_test = pd.read_csv('test_data_solution.txt', delimiter=':::', engine='python', names=columns, index_col=0)

# Preprocess the data
df_train['Description'] = df_train['Description'].astype(str).str.lower()
df_test['Description'] = df_test['Description'].astype(str).str.lower()

# Remove duplicates
df_train = df_train.drop_duplicates(keep='first')
df_test = df_test.drop_duplicates(keep='first')

# Separate features and labels
X_train_full = df_train['Description']
y_train_full = df_train['Genre']

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = tfidf.fit_transform(X_train)
X_val_vec = tfidf.transform(X_val)

# Train the Multinomial Naive Bayes Classifier
nb = MultinomialNB()
nb.fit(X_train_vec, y_train)

# Streamlit Interface
st.title("Movie Genre Prediction")

# Create dropdown options for movie plots
plot_options = [f"Plot Summary {i+1}" for i in range(len(df_train))]

# Dropdown for selecting plot summary
plot_selection = st.selectbox("Select a movie plot summary:", plot_options)

# Get the selected plot summary
if plot_selection:
    plot_index = plot_options.index(plot_selection)
    selected_plot = df_train.iloc[plot_index]['Description']
    
    # Display the plot summary description
    st.write("Plot Summary Description:", selected_plot)

    # Transform the selected plot using the TF-IDF vectorizer
    plot_transformed = tfidf.transform([selected_plot]).toarray()

    # Predict the genre
    genre_prediction = nb.predict(plot_transformed)
    st.write("Predicted Genre:", genre_prediction[0])
