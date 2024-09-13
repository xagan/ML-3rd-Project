# Student Performance Prediction: Machine Learning Project 3

## Project Overview
This project, developed as part of a university machine learning course, focuses on predicting student performance using various classification algorithms. The project is implemented in two main Jupyter notebooks: `main.ipynb` and `main2.ipynb`.

## Dataset
The project uses a dataset named 'corrected.csv', which contains various features related to student information and academic performance.

## Dependencies
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Code Structure

### main.ipynb

1. Data Loading and Preprocessing
   - Load the dataset
   - Handle missing values using forward fill
   - Encode categorical variables using LabelEncoder
   - Scale numerical features using StandardScaler

2. Dimensionality Reduction
   - Apply Principal Component Analysis (PCA) to reduce features to 10 components

3. Model Training and Evaluation
   - Split data into training and test sets
   - Train and evaluate multiple classifiers:
     - Gaussian Naive Bayes
     - Logistic Regression
     - Decision Tree
     - Random Forest
     - Support Vector Machine (SVM)

4. Performance Visualization
   - Display confusion matrices for each model
   - Plot ROC curves for all models
   - Create a bar plot comparing accuracy and F1 scores across models

### main2.ipynb

1. Data Loading and Preprocessing
   - Similar to main.ipynb, but uses ColumnTransformer for preprocessing
   - Separates categorical and numerical columns

2. Dimensionality Reduction
   - Applies PCA to reduce features to 10 components

3. Model Training and Evaluation
   - Uses the same set of classifiers as main.ipynb

4. Performance Visualization
   - Similar visualizations to main.ipynb

## Key Differences between main.ipynb and main2.ipynb

1. Preprocessing Approach:
   - main.ipynb uses separate steps for encoding and scaling
   - main2.ipynb uses ColumnTransformer for a more streamlined preprocessing pipeline

2. Feature Handling:
   - main2.ipynb explicitly separates categorical and numerical features

3. Implementation Details:
   - main2.ipynb uses slightly different variable names and structures

## Methods Used

1. Data Preprocessing:
   - Label Encoding: Convert categorical variables to numerical
   - Standard Scaling: Normalize numerical features
   - One-Hot Encoding (in main2.ipynb): Transform categorical variables

2. Dimensionality Reduction:
   - Principal Component Analysis (PCA): Reduce feature space to 10 components

3. Classification Algorithms:
   - Gaussian Naive Bayes
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - Support Vector Machine (SVM)

4. Model Evaluation:
   - Accuracy Score
   - F1 Score
   - Confusion Matrix
   - ROC Curve and AUC

5. Visualization:
   - Heatmaps for confusion matrices
   - ROC curves for all models
   - Bar plots for comparing model performances

## How to Run
1. Ensure all dependencies are installed
2. Place the 'corrected.csv' file in the same directory as the notebooks
3. Run the Jupyter notebooks cell by cell

## Future Improvements
- Experiment with other dimensionality reduction techniques
- Implement cross-validation for more robust results
- Try ensemble methods or other advanced algorithms
- Perform feature importance analysis
- Optimize hyperparameters for each model

## Note

This project is part of a learning process. The code and approach may not be optimal or suitable for production use.
