# 🧱 Concrete Compressive Strength Predictor

A machine learning model that predicts the compressive strength (MPa) of concrete based on its mix composition and curing age — built as part of a research project in **Transportation Systems Engineering**, focused on pavement/material strength prediction.

🔗 **Live App:** [concrete-strength-predictor-wa3t.onrender.com](https://concrete-strength-predictor-wa3t.onrender.com)

---

## 📌 Overview

Concrete compressive strength is a highly nonlinear function of its ingredients and age (Yeh, 1998). This project builds, evaluates, and deploys a tuned **XGBoost regression model** to predict strength from 8 input features, and compares it against several baseline ML models.

## 📊 Dataset

- **Source:** [UCI / Kaggle — Concrete Compressive Strength Data Set](https://www.kaggle.com/datasets/elikplim/concrete-compressive-strength-data-set)
- **Original paper:** I-Cheng Yeh, *"Modeling of strength of high performance concrete using artificial neural networks,"* Cement and Concrete Research, Vol. 28, No. 12, pp. 1797-1808 (1998)
- **Size:** 1030 samples, 8 input features, 1 target variable

| Feature | Unit |
|---|---|
| Cement | kg/m³ |
| Blast Furnace Slag | kg/m³ |
| Fly Ash | kg/m³ |
| Water | kg/m³ |
| Superplasticizer | kg/m³ |
| Coarse Aggregate | kg/m³ |
| Fine Aggregate | kg/m³ |
| Age | days |
| **Target: Compressive Strength** | **MPa** |

## 🧠 Methodology

1. **EDA** — distribution analysis, correlation heatmap, duplicate removal
2. **Preprocessing** — feature scaling (for linear/SVR models), train/test split (80/20)
3. **Model comparison** — Linear Regression, SVR, Random Forest, XGBoost
4. **Hyperparameter tuning** — GridSearchCV on XGBoost (54 combinations × 5-fold CV)
5. **Validation** — 5-fold cross-validation (shuffled) to confirm result stability
6. **Interpretability** — SHAP analysis to verify predictions align with known concrete engineering principles
7. **Deployment** — Streamlit web app hosted on Render

## 📈 Results

| Model | R² | RMSE (MPa) | MAE (MPa) |
|---|---|---|---|
| Linear Regression | 0.580 | 11.19 | 8.90 |
| SVR | 0.600 | 10.92 | 8.34 |
| Random Forest | 0.910 | 5.19 | 3.53 |
| XGBoost (untuned) | 0.925 | 4.73 | 2.88 |
| **XGBoost (tuned)** | **0.9375** | **4.32** | **2.73** |
| XGBoost (5-fold CV mean) | 0.9253 | — | — |

**Best model:** Tuned XGBoost Regressor (`learning_rate=0.1, max_depth=5, n_estimators=300, subsample=0.8`)

### Key findings (via SHAP analysis)
- **Age** and **Cement** are the strongest positive predictors of strength
- **Water** content negatively impacts strength — confirming the well-established water-cement ratio principle in concrete engineering
- **Blast Furnace Slag** and **Superplasticizer** contribute positively as strength-enhancing additives
- Model predictions align with established civil engineering domain knowledge, not just statistical fit

## 🗂️ Repository Structure

```
concrete-strength-app/
├── app.py                          # Streamlit web app
├── concrete_strength_model.pkl     # Trained & tuned XGBoost model
├── requirements.txt                # Python dependencies
├── render.yaml                     # Render deployment config
├── notebooks/
│   └── concrete_strength_model.ipynb   # Full analysis notebook (EDA → modeling → SHAP)
├── data/
│   └── concrete_data.csv           # Dataset
└── README.md
```

## 🚀 Running Locally

```bash
# Clone the repo
git clone https://github.com/todipriyal3101-alt/concrete-strength-app.git
cd concrete-strength-app

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## 🛠️ Tech Stack

- **Modeling:** scikit-learn, XGBoost, SHAP
- **Data handling:** pandas, NumPy
- **Visualization:** matplotlib, seaborn
- **Deployment:** Streamlit, Render

## 📄 Citation

If referencing the dataset:
> I-Cheng Yeh, "Modeling of strength of high performance concrete using artificial neural networks," *Cement and Concrete Research*, Vol. 28, No. 12, pp. 1797-1808 (1998).

## 👤 Author

Priyal Todi — Research Project, Transportation Systems Engineering (7th Semester)
