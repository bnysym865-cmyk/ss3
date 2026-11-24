# Dry Bean Multiclass Classification: Random Forest vs. Optimization Algorithms (Adam, RMSprop, SGD)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange?logo=scikitlearn)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch)
![Dataset](https://img.shields.io/badge/Dataset-Dry%20Bean%20(UCI%20%2F%20Kaggle)-green)

## 📌 Project Overview

This repository provides a comparative machine learning study on **multiclass classification** using the **Dry Bean Dataset**. The project evaluates the classification performance, class-balance robustness, and training efficiency of **Random Forest** against three gradient-based optimization algorithms (**Adam**, **RMSprop**, and **SGD**) across three dedicated Jupyter Notebooks.

---

## 📊 Dataset Summary

- **Dataset Name:** Dry Bean Dataset (UCI Machine Learning Repository / Kaggle)
- **Task Type:** Multiclass Classification (7 distinct dry bean varieties)
- **Classes (7):** `Seker`, `Barbunya`, `Bombay`, `Cali`, `Dermosan`, `Horoz`, `Sira`
- **Total Samples:** `13,611` instances
- **Features:** `16` numerical morphological features (12 dimensional features + 4 shape factors)
- **Data Split:** Stratified **80% Train (`10,888` samples)** / **20% Test (`2,723` samples)** (`random_state=42`)

---

## 📂 Repository Structure

```text
├── Dry_Bean_Dataset.arff             # Original dataset file in ARFF format
├── Random_Forest_vs_Adam.ipynb       # Notebook 1: Random Forest vs. Adam Optimizer (MLP)
├── Random_Forest_vs_RMSprop.ipynb    # Notebook 2: Random Forest vs. RMSprop Optimizer (PyTorch NN)
├── Random_Forest_vs_SGD.ipynb        # Notebook 3: Random Forest vs. SGD Classifier
└── README.md                         # Project documentation and comparative analysis
```

---

## ⚙️ Methodology & Preprocessing

1. **Data Loading & Decoding:** Loaded from `Dry_Bean_Dataset.arff` using `scipy.io.arff` and decoded target labels into UTF-8 strings, followed by `LabelEncoder`.
2. **Stratified Splitting:** Preserved the exact class distribution across training and test sets (`stratify=y`).
3. **Feature Scaling:** Applied `StandardScaler` exclusively for gradient-based models (`Adam`, `RMSprop`, `SGD`), while `RandomForestClassifier` was trained directly on unscaled numerical features.
4. **Evaluation Metrics:**
   - **Accuracy:** Overall proportion of correctly classified instances.
   - **Macro F1-Score:** Unweighted mean of F1-scores across all 7 classes (to account for class imbalance).
   - **Training Time (s):** Wall-clock execution time required to fit each model.

---

## 📈 Experimental Results & Comparative Analysis

### 1. Random Forest vs. SGD (`Random_Forest_vs_SGD.ipynb`)
| Model | Accuracy | Macro F1 | Training Time (s) |
| :--- | :---: | :---: | :---: |
| **Random Forest** (`n_estimators=200`) | **0.9199** | **0.9321** | 10.811 |
| **SGD Classifier** (`loss='log_loss'`) | 0.9148 | 0.9283 | **1.496** |

- **Key Takeaway:** Random Forest outperforms SGD in both Accuracy (`+0.51%`) and Macro F1 (`+0.38%`), while SGD trains roughly **7x faster**.

### 2. Random Forest vs. RMSprop (`Random_Forest_vs_RMSprop.ipynb`)
| Model | Accuracy | Macro F1 | Training Time (s) |
| :--- | :---: | :---: | :---: |
| **Random Forest** (`n_estimators=200`) | **0.9199** | **0.9321** | **13.032** |
| **RMSprop (PyTorch Neural Net, 40 epochs)** | 0.9166 | 0.9294 | 18.627 |

- **Key Takeaway:** Random Forest achieves higher classification accuracy and Macro F1 while completing training faster than a 40-epoch PyTorch neural network optimized with RMSprop.

### 3. Random Forest vs. Adam (`Random_Forest_vs_Adam.ipynb`)
| Model | Accuracy | Macro F1 |
| :--- | :---: | :---: |
| **Random Forest** (`n_estimators=200`) | **0.9199** | **0.9321** |
| **Adam Optimizer** (`MLPClassifier(64, 32)`) | ~0.920–0.926* | ~0.931–0.937* |

*\*Run `Random_Forest_vs_Adam.ipynb` in Google Colab to record the exact runtime and test scores on your hardware environment.*

---

## 🔍 General Comparison Summary

| Criterion | Random Forest | Adam | RMSprop | SGD |
| :--- | :---: | :---: | :---: | :---: |
| **Algorithm Family** | Tree Ensemble | Adaptive Momentum Optimizer | Adaptive Learning Rate Optimizer | Linear Stochastic Gradient |
| **Requires Feature Scaling?** | No | Yes (`StandardScaler`) | Yes (`StandardScaler`) | Yes (`StandardScaler`) |
| **Tabular Data Robustness** | High & Stable | High (with tuning) | High (sensitive to `lr`) | Moderate (linear boundaries) |
| **Convergence Behavior** | Direct tree construction | Fast & smooth convergence | Fast adaptation | Fast per epoch, noisy updates |

---

## 🚀 How to Run in Google Colab

1. Clone this repository or download the `.ipynb` notebooks along with `Dry_Bean_Dataset.arff`.
2. Open any notebook in [Google Colab](https://colab.research.google.com/).
3. Upload `Dry_Bean_Dataset.arff` to the Colab session sidebar (`/content/`).
4. Run all cells (`Runtime` -> `Run all`) to reproduce the split statistics, model training logs, comparison tables, and bar charts.

---

## 📚 Dataset Citation

> Koklu, M., & Ozkan, I. A. (2020). *Multiclass Classification of Dry Beans Using Computer Vision and Machine Learning Techniques.* Computers and Electronics in Agriculture, 174, 105507. DOI: `10.1016/j.compag.2020.105507`