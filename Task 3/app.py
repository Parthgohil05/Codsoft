import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score

# Set the title for the Streamlit app
st.title("Bank Customer Churn Modeling")

# Subheading for importing libraries
st.subheader("🔧 Importing Libraries")
st.code("""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
""")

# Subheading for loading data
st.subheader("📂 Loading Dataset")
data = pd.read_csv('Churn_Modelling.csv').dropna(axis=1)
data.drop(columns=['Surname', 'RowNumber', 'CustomerId'], inplace=True)
data = pd.get_dummies(data, columns=['Geography', 'Gender'])
st.write(data.info())
st.write(data.head())

# Subheading for data visualization
st.subheader("📊 Data Visualization: Churn Distribution")
churn_count = data['Exited'].value_counts()
temp_df = pd.DataFrame({
    'Exited': churn_count.index,
    'Counts': churn_count.values
})

plt.figure(figsize=(10, 6))
sns.barplot(x='Exited', y='Counts', data=temp_df)
plt.xticks(rotation=90)
st.pyplot(plt)

# Subheading for data splitting
st.subheader("✂️ Splitting Data")
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

st.write("Training set shape:", X_train.shape)
st.write("Validation set shape:", X_val.shape)
st.write("Test set shape:", X_test.shape)

# Function for cross-validation scoring
def cv_scoring(estimator, X, y):
    return accuracy_score(y, estimator.predict(X))

# Subheading for model comparison
st.subheader("⚖️ Model Comparison Using Cross-Validation")
models = {
    "SVC": SVC(),
    "Gaussian NB": GaussianNB(),
    "Random Forest": RandomForestClassifier(random_state=18)
}

for model_name in models:
    model = models[model_name]
    scores = cross_val_score(model, X, y, cv=10, n_jobs=-1, scoring=cv_scoring)
    st.write(f"{model_name}: Mean Score = {np.mean(scores)}")

# Subheading for training the best model
st.subheader("🏆 Training the Best Model: Random Forest Classifier")
rf_model = RandomForestClassifier(random_state=18)
rf_model.fit(X_train, y_train)
train_accuracy = accuracy_score(y_train, rf_model.predict(X_train))
val_accuracy = accuracy_score(y_val, rf_model.predict(X_val))

st.write(f"Training Accuracy: {train_accuracy * 100:.2f}%")
st.write(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# Subheading for confusion matrix
st.subheader("🔢 Confusion Matrix on Validation Data")
cf_matrix = confusion_matrix(y_val, rf_model.predict(X_val))
plt.figure(figsize=(12, 8))
sns.heatmap(cf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix for Random Forest Classifier on Validation Data")
st.pyplot(plt)

# Subheading for test data prediction
st.subheader("🧪 Test Data Prediction")
y_pred = rf_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
st.write(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Confusion Matrix for Test Data
st.subheader("📉 Confusion Matrix for Test Data")
cm = confusion_matrix(y_test, y_pred)
st.write(cm)

# Classification Report for Test Data
st.subheader("📋 Classification Report for Test Data")
report = classification_report(y_test, y_pred, output_dict=True)
st.write(report)

# ROC Curve and AUC for Test Data
st.subheader("📈 ROC Curve and AUC for Test Data")
y_proba = rf_model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_proba)
fpr, tpr, _ = roc_curve(y_test, y_proba)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {auc:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
st.pyplot(plt)
