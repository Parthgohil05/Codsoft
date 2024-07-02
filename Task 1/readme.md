# Movie Genre Prediction

This project is a web application that predicts the genre of a movie based on its plot summary using a Multinomial Naive Bayes classifier. The application is built with Streamlit, a framework for creating interactive web applications in Python.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model Training](#model-training)
- [Web Application](#web-application)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Introduction

The Movie Genre Prediction application allows users to select a movie plot summary from a dropdown menu, view its description, and predict the genre of the movie. The prediction is made using a Naive Bayes classifier trained on TF-IDF vectorized plot summaries.

## Dataset

The dataset used in this project consists of two files:
- `train_data.txt`: Contains training data with movie names, genres, and plot descriptions.
- `test_data_solution.txt`: Contains test data with movie names, genres, and plot descriptions.

## Preprocessing

1. The plot descriptions are converted to lowercase.
2. Duplicate entries are removed.
3. The data is split into training and validation sets.
4. TF-IDF vectorization is applied to convert text data into numerical vectors.

## Model Training

A Multinomial Naive Bayes classifier is trained using the TF-IDF vectors of the plot descriptions. The model is evaluated on the validation set.

## Web Application

The web application is built using Streamlit. It allows users to:
- Select a plot summary from a dropdown menu.
- View the plot summary description.
- Predict the genre of the selected plot summary.

## Installation

To run this application locally, follow these steps:

1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/movie-genre-prediction.git
   cd movie-genre-prediction
   ```

2. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

3. Place the `train_data.txt` and `test_data_solution.txt` files in the project directory.

## Usage

To start the Streamlit application, run the following command:
```sh
streamlit run app.py
```

This will launch the web application in your default web browser. You can then select a plot summary from the dropdown menu and see the predicted genre.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

