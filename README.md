Project Architecture & Methodology

This project provides a targeted, data-driven framework for predicting Tool Wear Failure (TWF) specifically inside high-precision industrial milling environments. By utilizing a subset-isolated machine learning pipeline on the AI4I 2020 Predictive Maintenance Dataset, the models specialize in high-quality manufacturing constraints where tighter dimensional tolerances make degradation critical.

The Data & Modelling Pipeline
The repository's code steps sequentially through the following experimental stages:

1. Type H Filtering: Drops heterogeneous machine data, reducing noise by narrowing down the operational profiles to the 961 specific high-quality variant records.
2. Preprocessing: Handles stratified 80/20 train-test splits and relies on strict `StandardScaler` transformations fitted exclusively on the training partition to avoid evaluation bias.
3. SMOTE Augmentation: Balances the skewed 12:1 non-failure to failure training distribution, expanding training records from 768 to 1,418 instances.
4. Classifier Evaluation: Benchmarks four separate structural architectures on precision, recall, F1-score, and AUC-ROC curves.

Performance Summary
The pipeline scripts output the following test-set performance metrics[cite: 600]:

| Classifier | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Random Forest (Best) | 92.75% | 52.17% | 80.00% | 0.6316 | 0.9639 |
| Logistic Regression | 91.19% | 46.88% | 100.0% | 0.6383 | 0.9558 |
| Support Vector Machine (SVC) | 90.67% | 44.83% | 86.67% | 0.5909 | 0.9547 |
| Multilayer Perceptron (MLP) | 90.67% | 44.83% | 86.67% | 0.5909 | 0.9494 |

*Note: While Logistic Regression yields a 100% recall rate, it comes at the cost of high false-alarm rates (lowest precision), establishing Random Forest as the most stable, operational system model for precision deployment*
