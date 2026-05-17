**----- Research Overview: Tool Wear Failure Prediction (Type H Filtered) -----**

This repository contains the source code, data pipeline, and machine learning models implemented for my research paper: **"Tool Wear Failure Prediction in High-Precision Industrial Milling: A Type H-Filtered Machine Learning Approach Using the AI4I 2020 Predictive Maintenance Dataset"**.

### 🔬 Problem Statement & Methodology
Most predictive maintenance approaches treat sensor data from different machine variants as a homogeneous population, which can dilute crucial failure signatures.In high-precision environments—such as **Type H (High-quality variant)** machines—even minor tool wear results in parts falling outside strict dimensional tolerances, leading to expensive batch rejections.

To address this, this project implements a strict **Type H filtering strategy** to isolate high-precision manufacturing data before training a supervised multi-classifier pipeline:
1. **Targeted Filtering:** Extracts and isolates the `Type H` subset from the AI4I 2020 Predictive Maintenance Dataset (reducing the data from 10,000 to 961 records).
2. **Preprocessing:** Scales the 5 primary operational sensor features using `StandardScaler`.
3. **Imbalance Mitigation:** Addresses the severe 12:1 class imbalance (only 7.7% True Failure events) by applying **SMOTE (Synthetic Minority Over-sampling Technique)** exclusively to the training set.
4. **Benchmarking:** Trains and compares Random Forest, Support Vector Classification (SVC), and a Multilayer Perceptron (MLP) neural network against a Logistic Regression baseline].

---

### ⚙️ Pipeline Architecture
---

📊 Experimental Setup & Features
The models ingest **5 key operational sensor features**:
* **Tool Wear [min]** (Primary temporal degradation feature).
* **Air Temperature [K]** & **Process Temperature [K]**
* **Rotational Speed [rpm]** & **Torque [Nm]** 

**Target Variable:** Binary Tool Wear Failure (`TWF`: 0 = Nominal, 1 = Failure).

---

📈 Key Results & Metrics
Evaluated on a completely unmanipulated, realistic testing distribution, the benchmarks yielded the following performance:

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **92.75%** | **52.17%** | 80.00% | **63.16%** | **96.39%** |
| Logistic Regression | 91.19% | 46.88% | **100.00%** | 63.83% | 95.58% |
| SVC (RBF Kernel) | 90.67% | 44.83% | 86.67% | 59.09% | 95.47% |
| MLP Neural Network | 90.67% | 44.83% | 86.67% | 59.09% | 94.94% |

🔑 Key Takeaways:
* **Top Performer:** **Random Forest** achieved the best overall discriminative capacity (**96.39% AUC-ROC**), significantly reducing false alarms (lowest false positives).
* **Dominant Predictor:** Feature importance analysis revealed that **Tool Wear Time** holds **81.70%** of the discriminative weight in predicting failure, aligning perfectly with industrial mechanical fatigue thresholds.

---

🛠️ Built With
* **Python 3**
* **scikit-learn** (StandardScaler, RandomForestClassifier, SVC, MLPClassifier, LogisticRegression)
* **imbalanced-learn** (SMOTE)
* **ucimlrepo** (Automated dataset acquisition)
* **Matplotlib & Seaborn** (Automated manufacturing visualization suite)
  
---
### 📜 Citation / Author
**Rodrigo Jr. C. Balayo** **College of Electronics Engineering, Technological University of the Philippines, Manila** 📧 rodrigojr.balayo@tup.edu.ph 
