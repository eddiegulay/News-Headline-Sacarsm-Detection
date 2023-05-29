# Sentiment Classification

This project aims to develop models for news headline sarcasm classification and feedback rating level prediction from restaurant feedbacks. The goal is to build accurate and efficient models using TensorFlow and Keras.

## Objective
- Classify news headlines as sarcastic or non-sarcastic.
- Predict the rating level of restaurant feedbacks.

## Approach
The project follows the following approach:
1. Data Collection: Gather a dataset of news headlines labeled with sarcasm and a dataset of restaurant feedbacks with associated ratings.
2. Data Preprocessing: Clean and preprocess the text data, including tasks such as tokenization.
3. Model Development:
    - News Headline Sarcasm Classification: Build a deep learning model using TensorFlow and Keras to classify news headlines as sarcastic or non-sarcastic.
    - Feedback Rating Level Prediction: Develop another deep learning model to predict the rating level (1 - 5 rating) of restaurant feedbacks.
4. Model Training: Train the developed models using the prepared datasets.
5. Model Evaluation: Assess the performance of the trained models using appropriate evaluation metrics.
6. Model Deployment: Deploy the trained models to make predictions on new data.
7. Documentation and Reporting: Create comprehensive documentation detailing the project's methodology, data preprocessing steps, model architecture, and evaluation results.

## Repository Structure
The repository contains the following files and directories:
- `datasets`: Directory containing the collected datasets.
- `model`: Directory containing trained models or model-related files.
- `tokenizers`: Directory containing tokenizer files or related components.
- `feedback.py`: Python file for feedback-related functions and utilities.
- `feedback_rating_inference.ipynb`: Jupyter Notebook for feedback rating level prediction inference.
- `feedback_rating_training.ipynb`: Jupyter Notebook for feedback rating level prediction model training.
- `README.md`: Markdown file providing an overview of the project.
- `sarcasm.py`: Python file for sarcasm-related functions and utilities.
- `sarcasm_detection.ipynb`: Jupyter Notebook for news headline sarcasm detection.
- `sarcasm_inference.ipynb`: Jupyter Notebook for news headline sarcasm detection inference.

## Prerequisites
- Python 3.9 or above
- TensorFlow 2.0 or above
- scikit-learn (sklearn) library
- NumPy library
- JSON library

## Setup Instructions
1. Clone the GitHub repository:
   ```bash
   git clone https://github.com/eddiegulay/Sentiment-Classification.git
   cd Sentiment-Classification
   ```
2. Create Virtual Environment
```bash
    python3 -m venv env
    source env/bin/activate  # On Windows: .\env\Scripts\activate
```

3. install dependencies
```bash
    pip install tensorflow scikit-learn numpy
```

## Datasets:

- News Headline Sarcasm Dataset: (available in dataset dir or you could use your own)
- Restaurant Feedbacks Dataset: (available in dataset dir or you could use your own)


## Open the Jupyter Notebooks:

### For Feedback Rating Level Prediction:

1. Open `feedback_rating_training.ipynb` in Jupyter Notebook.
2. Follow the instructions in the notebook to run the code, preprocess the data, train the model, and evaluate its performance.
3. Use `feedback_rating_inference.ipynb` to make predictions on new data.

## Customize and Experiment:

- Feel free to modify the code, parameters, or architectures according to your needs.
- Explore different preprocessing techniques, feature engineering approaches, or model architectures to improve performance.

~ same applies for sarcasm classification


## Notes:

- Ensure that you have enough computational resources (CPU/GPU) to train the models, as deep learning models can be computationally intensive if you use higher hyperparameters and larger datasets.
- For large datasets, consider using data generators or mini-batch training to manage memory constraints.


## Future Enhancements
In the future, the project can be expanded by incorporating additional features, such as sentiment analysis for feedbacks, incorporating contextual embeddings (e.g., BERT), or exploring other NLP techniques to improve model performance.

---

## Contributors

- [Eddie Gulay](https://github.com/eddiegulay)

Thank you for your interest in this project!