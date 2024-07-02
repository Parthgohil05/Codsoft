# Bank Customer Churn Modeling

This project demonstrates how to build a machine learning model to predict bank customer churn and display the results using a Streamlit web app. The dataset used for this project contains various customer details and whether they exited (churned) or not.

## Installation

To run this project, ensure you have Python installed on your machine. Then, install the required libraries using pip:

```sh
pip install numpy pandas matplotlib seaborn streamlit scikit-learn
```

## Usage

1. **Run the Streamlit App**:
   Save the provided code in a Python file (e.g., `churn_modeling.py`). Run the Streamlit app with the following command:

   ```sh
   streamlit run churn_modeling.py
   ```

2. **Interact with the App**:
   Open the local Streamlit server in your web browser. The app will guide you through the following steps:
   - Importing Libraries
   - Loading and displaying the dataset
   - Visualizing the churn distribution
   - Splitting the data into training, validation, and test sets
   - Comparing different models using cross-validation
   - Training the best model (Random Forest Classifier)
   - Displaying the confusion matrix and classification report for the validation and test data
   - Plotting the ROC curve and calculating the AUC for the test data

## Features

- **Data Preprocessing**: Load the dataset, handle missing values, and perform one-hot encoding for categorical variables.
- **Data Visualization**: Visualize the churn distribution using a bar plot.
- **Model Training and Evaluation**: Train multiple models (SVC, GaussianNB, RandomForestClassifier), compare their performance using cross-validation, and select the best model for further evaluation.
- **Confusion Matrix and Classification Report**: Display the confusion matrix and classification report for the validation and test data.
- **ROC Curve and AUC**: Plot the ROC curve and calculate the AUC for the test data.

## Dataset

The dataset used is `Churn_Modelling.csv`, which contains various customer details and their churn status. The file should be placed in the same directory as the Python script.

## Example Output

The Streamlit app will display various metrics and visualizations to help understand the model's performance, including:

- Churn distribution bar plot
- Confusion matrix heatmap
- Classification report
- ROC curve and AUC

## Credits

This project uses the following libraries:
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Streamlit
- Scikit-learn

## License

This project is licensed under the MIT License.