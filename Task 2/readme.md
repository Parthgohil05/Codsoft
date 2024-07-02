```markdown
# Spam Message Classification with Streamlit

This project demonstrates how to build a spam message classification model using Naive Bayes and display the results in a Streamlit web app. The data used for this project is a CSV file containing SMS messages labeled as "spam" or "ham".

## Installation

To run this project, ensure you have Python installed on your machine. Then, install the required libraries using pip:

```sh
pip install numpy pandas streamlit nltk scikit-learn
```

Additionally, download the necessary NLTK data:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

## Usage

1. **Run the Streamlit App**:
   Save the provided code in a Python file (e.g., `spam_classifier.py`). Run the Streamlit app with the following command:

   ```sh
   streamlit run spam_classifier.py
   ```

2. **Interact with the App**:
   Open the local Streamlit server in your web browser. The app allows you to select an SMS from the dataset and classify it as spam or not spam using the trained Naive Bayes model.

## Features

- **Text Preprocessing**: Remove special characters, punctuation, convert to lowercase, remove stopwords, and perform stemming.
- **Model Training**: Train a Naive Bayes classifier using the preprocessed text data.
- **Model Evaluation**: Evaluate the model using accuracy, precision, recall, F1 score, and confusion matrix.
- **Streamlit Interface**: Interactive web interface to classify SMS messages.

## Dataset

The dataset used is `spam.csv`, which contains the SMS messages and their labels ("spam" or "ham"). The file should be placed in the same directory as the Python script.

## Example Output

The Streamlit app will display whether the selected SMS is classified as "Spam" or "Not Spam" based on the trained model.

## Credits

This project uses the following libraries:
- NumPy
- Pandas
- Streamlit
- NLTK
- Scikit-learn

## License

This project is licensed under the MIT License.