# Student Academic Success Classification

## 1. Problem Statement

The objective of this project is to develop a machine learning classification system to predict students' academic outcomes based on demographic, academic, socioeconomic, and other relevant attributes.

The project implements five classification algorithms and compares their performance using six evaluation metrics. An interactive Streamlit application allows users to upload test data, select a trained model, and view its evaluation results.

The target variable has three possible outcomes:
- Dropout
- Enrolled
- Graduate

## 2. Dataset Description

**Dataset Name:** Predict Students' Dropout and Academic Success

**Source:** UCI Machine Learning Repository  
**Dataset Link:** https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success

Dataset characteristics:
- Total instances: 4,424
- Input features: 35
- Target variable: Target
- Classification type: Multiclass classification
- Target classes: Dropout, Enrolled, Graduate

The dataset contains student-related information, including demographic details, academic background, application information, and socioeconomic indicators.

The dataset was divided into training and testing sets using an 80:20 stratified split with a random state of 42.

Categorical features were encoded using OneHotEncoder, and numerical features were standardized using StandardScaler. These transformations were fitted only on the training data to avoid data leakage.

## 3. GitHub Repository Link

**GitHub Repository:** [Insert your GitHub repository URL here]

The repository contains:
- Streamlit application (`app.py`)
- Model training source code (`train_models.py`)
- Data preprocessing source code (`preprocess_data.py`)
- Dataset and test dataset
- Five saved trained model pipelines
- Model comparison results
- Requirements file
- Project documentation

## 4. Models Used and Performance Comparison

The following five classification algorithms were implemented and evaluated using the same dataset and test split:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

### Model Evaluation Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7243 | 0.8786 | 0.6956 | 0.7000 | 0.6876 | 0.5729 |
| Decision Tree | 0.6847 | 0.7635 | 0.6404 | 0.6363 | 0.6337 | 0.4989 |
| KNN | 0.7028 | 0.7977 | 0.6320 | 0.6131 | 0.6182 | 0.5049 |
| Gaussian Naive Bayes | 0.2441 | 0.6725 | 0.4798 | 0.3922 | 0.2171 | 0.1242 |
| Support Vector Machine | 0.7299 | 0.8831 | 0.6979 | 0.6988 | 0.6904 | 0.5760 |

*Precision, Recall and F1 Score are calculated using macro averaging. AUC is calculated using the one-vs-rest multiclass approach.*

### Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|---|---|
| Logistic Regression | Achieved 72.43% accuracy and 0.8786 AUC. It demonstrated strong overall classification performance and achieved the highest recall of 0.7000. Its results were close to those of SVM. |
| Decision Tree | Achieved 68.47% accuracy and 0.7635 AUC. The model offers an interpretable classification approach but performed below SVM and Logistic Regression across the main evaluation metrics. |
| KNN | Achieved 70.28% accuracy and 0.7977 AUC. It performed better than Decision Tree in accuracy and AUC, although its precision, recall and F1 score were comparatively lower. |
| Gaussian Naive Bayes | Achieved 24.41% accuracy and 0.6725 AUC. Its low F1 and MCC scores indicate comparatively weak classification performance on this dataset. The model's assumptions about feature independence may limit its performance. |
| Support Vector Machine | Achieved the highest accuracy (0.7299), AUC (0.8831), precision (0.6979), F1 score (0.6904), and MCC (0.5760). It demonstrated the strongest overall classification performance among the five models. |

### Overall Winner

**Support Vector Machine (SVM)** is the overall winner for this dataset.

SVM achieved the best results across five of the six evaluation metrics, including accuracy, AUC, precision, F1 score, and MCC. Logistic Regression achieved marginally higher recall. However, SVM demonstrated the strongest overall balance of classification performance.

## 5. Streamlit Application

The interactive application is developed using Streamlit.

**Live Application:** [Insert your deployed Streamlit application URL here]

Application features:
- CSV test dataset upload.
- Dropdown selection of the five trained classification models.
- Loading of saved model pipelines.
- Display of Accuracy, AUC, Precision, Recall, F1 Score, and MCC.
- Confusion matrix visualization.
- Classification report.
- Test data preview.

The application uses the uploaded test dataset for evaluation and does not retrain the models. The preprocessing transformations are included in the saved model pipelines, ensuring consistent processing during prediction.

## 6. Technologies Used

- Python 3.9
- Pandas and NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn
- Joblib
- GitHub
- Streamlit Community Cloud