# 🚀 Machine Learning Basics & Supervised Learning Projects

Welcome to my Machine Learning practice repository! This project tracks my journey of learning core Machine Learning concepts from the ground up, focusing on end-to-end workflows including **data cleaning**, **preprocessing**, and building various **supervised learning models**.

---

## 📌 Project Highlights

### 1. 🧹 Data Cleaning & Preprocessing
Real-world datasets require proper preprocessing before feeding into ML models:
* **Missing Value Imputation**: Handling numerical missing values (mean imputation) and categorical nulls.
* **Feature Dropping**: Removing redundant/noisy columns.
* **Categorical Encoding**: Converting text categories using `LabelEncoder` and data type casting (e.g., boolean to integer).
* **Feature Scaling**: Applying `StandardScaler` to prevent feature dominance and ensure no data leakage (fitting on train set, transforming test set).
* **Train/Test Splitting**: Stratified / random splitting (`train_test_split`) for fair model evaluation.

---

## 🤖 Supervised Learning Models Implemented (Practice)

### 🔹 Regression Models
* **Linear Regression**: Predicts continuous numerical targets (e.g., Ford car prices, insurance charges, health metrics).

### 🔹 Classification Models (Titanic Dataset Survival Prediction)
We tested and compared multiple classification algorithms to predict passenger survival:

| Model | Technique | Key Metric (Accuracy) |
| :--- | :--- | :---: |
| **Logistic Regression** | Linear probability estimation | ~81.63% |
| **K-Nearest Neighbors (KNN)** | Distance-based classification ($k=5$ with standard scaling) | ~81.29% |
| **Gaussian Naive Bayes** | Probabilistic classification | ~77.89% |
| **Decision Tree** | Rule-based hierarchical splitting | ~74.49% |
| **Support Vector Classifier (SVC)** | Non-linear boundary with RBF kernel | **~81.97%** 🏆 |

---

## 📊 Model Evaluation & Metrics
To thoroughly assess performance beyond just accuracy:
* **Confusion Matrix**: True Positives, False Positives, True Negatives, False Negatives.
* **Classification Report**: Precision, Recall, and F1-score for each class.
* **R² Score & MSE**: For regression model error analysis.

---

## 🛠️ Tech Stack & Libraries
* **Python 3**
* **Data Manipulation**: `pandas`, `numpy`
* **Data Visualization**: `matplotlib`, `seaborn`
* **Machine Learning**: `scikit-learn`
* **Web App Framework**: `streamlit`
* **Model Persistence**: `joblib`
* **Environment**: Jupyter Notebooks (`.ipynb`) & Python scripts

---

## 🌟 Final Project: Heart Disease Predictor & Streamlit Web App (My 1st End-to-End ML Model! 🎉)

As the capstone milestone of my ML journey, **I built and deployed my very first complete end-to-end Machine Learning model and interactive web application** for **Heart Disease Risk Prediction**.

### 1. Dataset & Problem Overview (`heart.csv`)
* **918 patient observations** with 11 clinical features (`Age`, `Sex`, `ChestPainType`, `RestingBP`, `Cholesterol`, `FastingBS`, `RestingECG`, `MaxHR`, `ExerciseAngina`, `Oldpeak`, `ST_Slope`).
* **Target**: `HeartDisease` (`1` = Heart Disease Detected, `0` = Normal).

### 2. Preprocessing & Feature Engineering
* Handled binary categorical mappings (`is_female`, `is_ExerciseAngina`, `is_up`).
* One-Hot Encoded multi-class features (`ChestPainType`, `RestingECG`).
* Grouped blood pressure into clinical categories (`RestingBP_cat`: `Normal`, `Elevated`, `High`, `Very High`).
* Applied `StandardScaler` strictly on `X_train` to eliminate data leakage.

### 3. Model Benchmark on Heart Disease Dataset
Evaluated on an 80/20 train-test split (184 test samples):

| Model | Accuracy | F1 Score | Performance Summary |
| :--- | :---: | :---: | :--- |
| **Support Vector Machine (SVC)** 🥇 | **~89.13%** | **0.9029** | **Best Model** — Selected for production deployment. |
| **K-Nearest Neighbors (KNN)** 🥈 | **~86.96%** | **0.8812** | Reaches **>90%** when tuned with Manhattan metric. |
| **Logistic Regression** | **~85.33%** | **0.8696** | Solid linear probability baseline. |
| **Gaussian Naive Bayes** | **~85.33%** | **0.8670** | Strong probabilistic benchmark. |
| **Decision Tree** | **~79.89%** | **0.8177** | Rule-based baseline. |

### 4. Interactive Streamlit Web Application (`app.py`)
Built a web app allowing users and clinicians to input patient metrics and get real-time health risk assessments:
* **Inputs**: Two-column layout with intuitive sliders and dropdowns.
* **Automated Inference Pipeline**: Formats inputs, applies one-hot encoding, scales via saved `scaler.pkl`, and predicts with `svm_heart.pkl`.
* **Output**: Clear, color-coded health risk alert cards (⚠️ High Risk vs ✅ Low Risk / Normal).

#### How to run the web app locally:
```bash
streamlit run app.py
```

---

## 📂 Repository Structure
```text
├── final project .ipynb # Capstone: Complete Heart Disease EDA, training & model export
├── app.py               # Streamlit web app for real-time heart disease predictions
├── svm_heart.pkl        # Serialized production SVM model
├── scaler.pkl           # Saved StandardScaler parameters
├── col.pkl              # Expected feature column order for inference
├── heart.csv            # Heart disease clinical dataset
├── project.ipynb        # Practice: Titanic survival classification (Logistic Regression, KNN, NB, DT, SVM)
├── mlproject.ipynb      # Practice: Linear Regression on Ford car dataset
├── dc1.ipynb            # Practice: Exploratory Data Analysis & Data Cleaning
├── ford.csv             # Ford car pricing dataset
├── insurance.csv        # Medical insurance charges dataset
├── README.md            # Project documentation
└── .gitignore           # Git ignore configuration
```
