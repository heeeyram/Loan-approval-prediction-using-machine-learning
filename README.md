# Loan-approval-prediction-using-machine-learning

---

## 📌 Objective

- Understand the key features that affect loan approvals.
- Clean and preprocess real-world data.
- Visualize trends and correlations in data.
- Train a machine learning model to predict loan approval.
- Evaluate the model using accuracy metrics.

---

## 📊 Dataset Overview

The dataset contains the following features:

- **Loan_ID**: Unique Loan Identifier (dropped in preprocessing)
- **Gender**
- **Married**
- **Education**
- **Self_Employed**
- **ApplicantIncome**
- **CoapplicantIncome**
- **LoanAmount**
- **Loan_Amount_Term**
- **Credit_History**
- **Property_Area**
- **Loan_Status**: Target Variable (Y/N)

---

## ⚙️ Technologies Used

- **Python** 🐍
- **Jupyter Notebook**
- **Pandas**
- **Matplotlib**
- **Seaborn**
- **Missingno**
- **Scikit-learn**

---

## 📈 Exploratory Data Analysis (EDA)

- **Correlation Heatmap** to understand inter-feature relationships.
- **Countplots** for categorical variables vs Loan Status.
- **Boxplots** to compare income and loan amount by approval status.
- **Pairplots** for multivariate visualization.
- **Pie Charts** for target distribution.
- **Distribution Plots** to analyze numerical spreads.
- **Missing Value Matrix** using `missingno`.

---

## 🧹 Data Preprocessing

- Forward-filled missing values.
- Removed `Loan_ID` as it's non-predictive.
- Used `LabelEncoder` for categorical to numerical conversion.
- Split dataset into features (X) and target (y).
- Train-test split using 70/30 ratio.

---

## 🤖 Machine Learning Model

- **RandomForestClassifier** from `sklearn.ensemble`
- Trained on `X_train`, predicted on `X_test`.
- Plotted **feature importances** to see what drives predictions.

---

## 📊 Model Evaluation

- Calculated **Accuracy Score**.
- Evaluated classification performance using visual outputs and accuracy percentage.

---

## 🔮 Final Output

The notebook prints the final model accuracy and shows key graphs and visualizations. This can help banks or loan officers better understand applicant risk levels.

---

## 🚀 How to Run

1. Clone this repo:
   ```bash
   git clone https://github.com/heeeyram/Loan-approval-prediction-using-machine-learning.git
   cd Loan-approval-prediction-using-machine-learning
