# 🤖 Project: Titanic Survival Prediction (Machine Learning)

## ✨ Summary

Successfully developed, tuned, and deployed a predictive model for passenger survival using the 668-passenger Titanic dataset. This project involved a rigorous, head-to-head comparison of multiple classification algorithms (**Decision Tree** vs. **Random Forest**) and advanced validation techniques.

The primary achievement was building and optimizing a robust classifier, culminating in a **Random Forest model that achieved a 0.84 (84%) mean accuracy** during K-fold cross-validation. This project demonstrates a full-cycle machine learning workflow, from data preprocessing and feature selection to hyperparameter tuning and final model evaluation.

## 🏆 Core Achievements & Contributions

* **Comparative Model Development:** Built and trained two distinct supervised learning models (**Decision Trees** and **Random Forest**) to identify the most accurate and robust algorithm for this classification problem.
* **Advanced Hyperparameter Tuning:** Implemented a **grid search** methodology to systematically tune key hyperparameters for both models (e.g., `maxdepth`, `minsplit` for trees; `ntree` and `mtry` for forests), moving beyond default settings to maximize predictive performance.
* **Robust Validation:** Employed **K-fold Cross-Validation (k=5)** to validate model performance, ensuring the final accuracy score was reliable and that the model was not overfitting to the training data.
* **Data-Driven Model Selection:** Conclusively identified the **Random Forest** model as the superior solution, achieving a **mean accuracy (meanacc) of 0.8396** with its optimized parameters (`ntree=225`, `mtry=3`).
* **Feature Importance Analysis:** Used the models to extract and quantify the most critical predictors of survival. The analysis confirmed that **`Sex`** was the most dominant predictive feature, followed by `Fare`, `Pclass`, `Cabin`, and `Age`.
* **Clean & Reusable Code:** Wrote a clean `source_code.R` script that includes a final `my_model` function, as required by the project specs, capable of taking new test data and returning predictions.

## 🛠️ Technology Stack

* **Language:** **R**
* **Core Libraries:**
    * **`randomForest`** (for Random Forest model)
    * **`rpart`** & **`rpart.plot`** (for Decision Trees)
    * **`ggplot2`** (for visualizing hyperparameter tuning results)
* **Techniques:** Supervised Machine Learning, Classification, Hyperparameter Tuning, Grid Search, K-fold Cross-Validation, Feature Importance.

## 📄 Repository Contents

* **`report.pdf`:** The final 6-page report detailing the ML methodology, model comparison, and results.
* **`source_code.R`:** The complete R source code for data preprocessing, model training, tuning, and the final `my_model` function.
* **`my_model.RData`:** The final, trained, and saved Random Forest classifier object, ready for prediction.

## 👥 Authors

* **Tomy Liu**
* **Jiawei Xu**
