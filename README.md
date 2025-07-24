# Heart Disease Prediction from Medical Data
![image](https://github.com/user-attachments/assets/dfe9cb78-7596-404e-b8f8-a8e1c638d5ce)

## Project Overview

This project aims to predict the possibility of heart disease in patients based on various medical data points. Utilizing a structured medical dataset, we apply several classification techniques to build predictive models that can identify individuals at risk.

## Objective

The primary objective is to develop and evaluate machine learning models capable of accurately predicting the presence or absence of heart disease using patient health metrics.

## Approach

The approach involves a standard machine learning workflow:
1.  **Data Loading & Initial Exploration**: Loading the dataset and performing initial checks.
2.  **Exploratory Data Analysis (EDA)**: Visualizing data distributions and correlations to gain insights.
3.  **Data Preprocessing**: Handling missing values, encoding categorical features, and scaling numerical features.
4.  **Data Splitting**: Dividing the dataset into training and testing sets to ensure robust model evaluation.
5.  **Model Training & Evaluation**: Applying various classification algorithms and assessing their performance using key metrics.

## Dataset Details & Features

* **Dataset**: The `heart.csv` dataset.
* **Dimensions**: The dataset contains **303 entries** and **14 columns**.
* **Missing Values**: All columns have **303 non-null** entries, indicating that there are no missing values in this dataset.
* **Features Used**: The models leverage a variety of patient attributes including:
    * `age`: Age of the patient
    * `sex`: Sex (1 = male; 0 = female)
    * `cp`: Chest pain type (0-3)
    * `trestbps`: Resting blood pressure
    * `chol`: Serum cholestoral in mg/dl
    * `fbs`: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
    * `restecg`: Resting electrocardiographic results (0-2)
    * `thalach`: Maximum heart rate achieved
    * `exang`: Exercise induced angina (1 = yes; 0 = no)
    * `oldpeak`: ST depression induced by exercise relative to rest
    * `slope`: The slope of the peak exercise ST segment
    * `ca`: Number of major vessels (0-3) colored by flourosopy
    * `thal`: Thallium stress test result (1, 2, 3)
* **Target Variable**: `target` (1 = presence of heart disease; 0 = absence of heart disease).

## Algorithms Applied

The following classification algorithms were implemented and evaluated:
* **Logistic Regression**
* **Support Vector Machine (SVM)**
* **Random Forest Classifier**
* **XGBoost Classifier**

## Setup and How to Run

This project was developed in a Google Colaboratory (Colab) environment, which provides free access to GPUs and simplified setup.

1.  **Clone the Repository**:
    ```bash
    git clone [https://github.com/YourGitHubUsername/YourRepositoryName.git](https://github.com/YourGitHubUsername/YourRepositoryName.git)
    ```
    (Replace `YourGitHubUsername` and `YourRepositoryName`)
2.  **Open in Google Colab**: Upload the `Heart_Disease_Prediction_from_Medical_Data.ipynb` file to your Google Drive, then open it with Google Colab.
3.  **Connect to Google Drive**: In the Colab notebook, ensure your `heart.csv` dataset is accessible, typically by mounting Google Drive and placing the file at `/content/drive/MyDrive/heart.csv` or adjusting the path in the notebook accordingly.
4.  **Run All Cells**: Execute all cells sequentially in the Colab notebook. This will perform data loading, preprocessing, model training, and evaluation.

## Results & Performance Summary

| Model               | Accuracy (Test Set) | Precision (Test Set) | Recall (Test Set) | F1-Score (Test Set) | ROC AUC Score (Test Set) |
| :------------------ | :------------------ | :------------------- | :---------------- | :------------------ | :----------------------- |
| Logistic Regression | 0.8033              | 0.7692               | 0.9091            | 0.8333              | 0.8690                   |
| SVM                 | 0.8361              | 0.7949               | 0.9394            | 0.8611              | 0.8864                   |
| Random Forest       | 0.8689              | 0.8421               | 0.9697            | 0.9014              | 0.9427                   |
| XGBoost             | 0.8033              | 0.7561               | 0.9394            | 0.8378              | 0.8561                   |

The **Random Forest Classifier** generally shows the strongest performance among the models evaluated, achieving an accuracy of 0.8689 and an ROC AUC score of 0.9427 on the test set.

## Visualizations

The notebook includes various visualizations to aid understanding:
* **Correlation Matrix Heatmap**: To visualize the relationships between all features.
* **Distribution Plots**: Histograms and KDE plots for continuous variables.
* **Count Plots for Categorical Variables vs. Target**: To show how different categories relate to the target variable.
* **Model Performance Comparison Chart**: A bar chart summarizing the accuracy of all trained models.

## Dependencies

The project relies on the following Python libraries:
* `pandas`
* `numpy`
* `scikit-learn` (specifically `train_test_split`, `StandardScaler`, `LogisticRegression`, `SVC`, `RandomForestClassifier`, and various `sklearn.metrics` for evaluation like `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `roc_auc_score`, `confusion_matrix`, `classification_report`)
* `xgboost`
* `matplotlib.pyplot`
* `seaborn`
* `google.colab`

## Future Work

* **Hyperparameter Tuning**: Optimize model performance further using techniques like GridSearchCV or RandomizedSearchCV.
* **Feature Engineering**: Create new features from existing ones to potentially improve model accuracy.
* **Ensemble Methods**: Explore advanced ensemble techniques to combine the strengths of multiple models.
* **Deep Learning**: Investigate neural networks for this classification task.
