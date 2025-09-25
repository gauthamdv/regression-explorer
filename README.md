# 🏠 Linear Regression & Feature Engineering Analysis

## 📖 Introduction

This project demonstrates **linear regression** on a dataset with multiple features.
The focus is on **feature engineering**, applying **transformations** to stabilize variance, and evaluating the model using various metrics.

---

## 📂 Project Structure

```
project-root/
│
├── datasets/                # Raw and processed datasets
│   ├── Housing.csv
│   └── Housing-Clean.csv
│
├── models/                  # Saved models
│   └── model.pkl
│
├── plots/                   # Visualizations
│   ├── pred_vs_actual.png
│   ├── residuals.png
│   └── r2_vs_transforms.png
│
├── requirements.txt         # Python dependencies
├── regression-explorer.ipynb
└── README.md
```

---

## 🧹 Data Preprocessing

* Features split into **numeric** and **binary** types.
* New features created to improve model performance:

  * `total_rooms`
  * `area_per_bedroom`
  * `total_features`
* **Standardization** applied to numeric features for better regression performance.

---

## 📈 Single Feature Regression

* Initial regression using **only the `area` feature**.
* Limited predictive power: **R² ≈ 0.2729**.

---

## 📊 Multiple Feature Regression

* Regression using **all features**.
* Significant performance improvement: **R² ≈ 0.6529**.
* **Feature engineering** further enhanced model performance.

---

## 🔄 Transformations

* Tested transformations on the **target variable**:

  * Cube-root
  * Log
  * Yeo-Johnson
* **R² scores** compared across transformations.
* **Cube-root transformation** provided the highest R² score.

---

## 🛠 Regularization

* Methods tested: **Ridge, Lasso, ElasticNet**.
* Ridge regression provided slightly **better coefficient stability**, but minimal R² improvement.

---

## 🏆 Results

* Cube-root transformation + feature engineering = **best results**.
* Ridge regularization can be optionally applied for **more stable coefficients**.

---

## 🔚 Conclusion

* Linear regression assumes **linear relationships**, limiting predictive power for this dataset.
* **Feature engineering** and **transformations** improve performance but **perfect prediction is not achievable** with this approach.
* The **final model** is saved and ready for future predictions.

---

## 💾 Model Storage

* Trained model saved using `joblib`:

```python
import joblib
joblib.dump(model, 'models/model.pkl')
```

---

## ⚡ Dependencies

Install required packages using:

```bash
pip install -r requirements.txt
```

---

✨ **Happy Data Science!**
