# Random Forest vs. Gradient-Based Optimizers

This repository contains independent Google Colab notebooks comparing a **Random Forest** classifier (tree-based ensemble baseline) against a parametric **Binary Logistic Regression** model trained using five gradient-based optimization algorithms on the same dataset and train/test split (`80% / 20%`, `random_state=42`).

## Methodology Note
- **Random Forest** constructs an ensemble of decision trees using impurity-based splits (Gini/Entropy) and bootstrap aggregation (bagging), so it does not rely on gradient descent or feature scaling.
- **Gradient-Based Optimizers** minimize Binary Cross-Entropy (BCE) loss iteratively over standardized features (`StandardScaler`).

## Repository Structure
| File | Algorithm / Model | Key Characteristics |
| :--- | :--- | :--- |
| `01_Random_Forest.ipynb` | Random Forest (`n_estimators=100`) | Non-parametric tree ensemble baseline |
| `02_GD.ipynb` | Batch Gradient Descent (GD) | Full-dataset gradient update per epoch |
| `03_SGD.ipynb` | Stochastic Gradient Descent (SGD) | Single-sample noisy updates per step |
| `04_Mini_Batch_GD.ipynb` | Mini-Batch Gradient Descent | Balanced updates (`batch_size=64`) |
| `05_RMSprop_AdaGrad.ipynb` | RMSprop / AdaGrad | Adaptive per-parameter learning rates |
| `06_Adam_Momentum.ipynb` | Adam with Momentum | First & second moment adaptive updates |

## Experimental Results
*(Update the values below with your actual Colab outputs)*

| Model / Optimizer | Final Train Loss | Test Accuracy | Test F1-Score | Training Time (s) |
| :--- | :---: | :---: | :---: | :---: |
| Random Forest | N/A | — | — | — |
| GD | — | — | — | — |
| SGD | — | — | — | — |
| Mini-Batch GD | — | — | — | — |
| RMSprop / AdaGrad | — | — | — | — |
| Adam with Momentum | — | — | — | — |

## Key Takeaways
- **Accuracy & Non-Linearity:** Random Forest captures non-linear feature interactions without scaling, whereas linear decision boundaries trained via optimizers depend on feature separability.
- **Convergence Speed:** Adaptive optimizers (**Adam with Momentum** and **RMSprop/AdaGrad**) converge in fewer epochs than vanilla **GD**, while **Mini-Batch GD** balances computation speed and update stability compared to single-sample **SGD**.

## How to Run
1. Open any `.ipynb` file in **Google Colab**.
2. Ensure the same dataset file (`dataset.csv`) is loaded across all six notebooks.
3. Run all cells sequentially to view training time, evaluation metrics, and the loss curve.
