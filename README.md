# ❤️ Heart Disease Prediction — My 1st Machine Learning Model & Web App 🚀

Welcome to my Machine Learning repository! This project marks an exciting milestone: **I built and deployed my very first end-to-end Machine Learning model** — a **Heart Disease Risk Classifier** complete with data preprocessing, model selection, tuning, and an interactive **Streamlit web application**.

---

## 🌟 Project Milestone & Overview
Cardiovascular diseases (CVDs) are the number one cause of death globally. Early detection of individuals at high risk allows for timely lifestyle and clinical interventions. 

In this project, I took raw clinical patient data through the entire machine learning lifecycle:
1. **Exploratory Data Analysis (EDA)** & Statistical testing.
2. **Feature Engineering** (One-Hot Encoding, domain-specific categorization).
3. **Leakage-Free Feature Scaling** using `StandardScaler`.
4. **Model Exploration & Benchmarking** across 5 different supervised algorithms.
5. **Model Serialization** using `joblib`.
6. **Web App Deployment** via an interactive UI built with **Streamlit**.

---

## 📊 Dataset Information (`heart.csv`)
* **Total Samples**: 918 patient observations
* **Target Variable**: `HeartDisease` (1 = Heart Disease Detected, 0 = Normal)
* **Clinical Features**:
  * `Age`: Age of the patient (years)
  * `Sex`: Sex of the patient (`M`: Male, `F`: Female)
  * `ChestPainType`: Type of chest pain (`TA`: Typical Angina, `ATA`: Atypical Angina, `NAP`: Non-Anginal Pain, `ASY`: Asymptomatic)
  * `RestingBP`: Resting blood pressure (mm Hg)
  * `Cholesterol`: Serum cholesterol (mm/dl)
  * `FastingBS`: Fasting blood sugar (`1` if > 120 mg/dl, `0` otherwise)
  * `RestingECG`: Resting electrocardiogram results (`Normal`, `ST`, `LVH`)
  * `MaxHR`: Maximum heart rate achieved (60 to 202 bpm)
  * `ExerciseAngina`: Exercise-induced angina (`Y`: Yes, `N`: No)
  * `Oldpeak`: ST depression induced by exercise relative to rest
  * `ST_Slope`: The slope of the peak exercise ST segment (`Up`, `Flat`, `Down`)

---

## ⚙️ Data Preprocessing & Feature Engineering
* **Categorical Encoding**:
  * Binary flags: `Sex` $\rightarrow$ `is_female`, `ExerciseAngina` $\rightarrow$ `is_ExerciseAngina`, `ST_Slope` $\rightarrow$ `is_up`.
  * Multi-class features: One-hot encoded dummy variables for `ChestPainType` and `RestingECG`.
* **Clinical Feature Binning**:
  * Grouped `RestingBP` into standard medical categories (`Normal`, `Elevated`, `High`, `Very High`).
* **Feature Scaling (No Data Leakage)**:
  * Applied `StandardScaler` strictly **after** the train/test split.
  * Fitted on `X_train` and applied `transform` to `X_test` to keep test data completely unseen.

---

## 🏆 Model Benchmarking & Evaluation

We evaluated 5 supervised learning algorithms using an 80/20 train-test split:

| Model | Accuracy | F1 Score | Notes |
| :--- | :---: | :---: | :--- |
| **Support Vector Classifier (SVC)** 🥇 | **89.13%** | **0.9029** | **Top Performer**. Maximizes geometric margin with non-linear RBF kernel. |
| **K-Nearest Neighbors (KNN)** 🥈 | **86.96%** | **0.8812** | Tuned with Manhattan distance achieves **> 90%** accuracy. |
| **Logistic Regression** | **85.33%** | **0.8696** | Strong linear baseline with high interpretability. |
| **Gaussian Naive Bayes** | **85.33%** | **0.8670** | Fast, robust probabilistic classifier. |
| **Decision Tree** | **79.89%** | **0.8177** | Intuitive rule-based splits. |

> 🏆 **Champion Model**: **Support Vector Machine (SVM)** was selected as the production model due to its highest generalization accuracy (~89.13%) and strong balance between precision and recall (F1 score of 0.9029).

---

## 🖥️ Streamlit Web Application (`app.py`)

The application allows users and clinicians to input patient parameters and receive an instantaneous risk assessment:

* **Clean Two-Column UI**: Sliders and dropdowns tailored to medical ranges.
* **Automated Feature Engineering Pipeline**: Raw user inputs are automatically transformed, one-hot encoded, and standardized behind the scenes.
* **Instant Risk Feedback**:
  * ⚠️ **High Risk of Heart Disease Detected** (Red alert with medical consultation advice).
  * ✅ **Low Risk / Normal** (Green success card).

### 🚀 How to Run the App Locally:
```bash
# 1. Clone the repository
git clone https://github.com/vaibhav1874/ml-basic-.git
cd ml-basic-

# 2. Install dependencies
pip install streamlit pandas scikit-learn joblib

# 3. Launch Streamlit
streamlit run app.py
```

---

## 📂 Repository Structure
```text
├── final project .ipynb # Complete EDA, preprocessing, scaling & model training notebook
├── app.py               # Streamlit web application interface & inference pipeline
├── svm_heart.pkl        # Serialized trained SVM model
├── scaler.pkl           # Saved StandardScaler parameters (mean & scale)
├── col.pkl              # Saved feature order for consistent inference
├── heart.csv            # Heart disease clinical dataset
├── project.ipynb        # Earlier classification practice (Titanic survival dataset)
├── mlproject.ipynb      # Linear Regression practice (Ford vehicle prices)
├── ford.csv             # Ford car pricing dataset
├── insurance.csv        # Medical insurance cost dataset
├── README.md            # Comprehensive project documentation
└── .gitignore           # Git ignore configuration
```

---

## 🛠️ Tech Stack
* **Language**: Python 3
* **Libraries**: `pandas`, `numpy`, `scikit-learn`, `joblib`
* **Visualization**: `matplotlib`, `seaborn`
* **App Framework**: `Streamlit`
* **Version Control**: Git & GitHub
