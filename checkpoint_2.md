# Checkpoint 2

## Part B

### 2. Briefly explain your baseline model and your plans for improving the model.

**Baseline model**  
A **linear regression** predicts `calories` from `n_steps` and `n_ingredients` (always available at submission). The pipeline uses `StandardScaler` and `LinearRegression`, evaluated with **RMSE** (kcal) and **R²**. With only two linear predictors, the baseline is interpretable but leaves much variance unexplained and yields fairly large errors.

**Plans for improving the model**  

- **Richer features**: Add `log_minutes`, `n_tags`, and binary flags like `is_dessert` and `is_quick`.  
- **Nonlinear model**: Switch to a **Random Forest Regressor** to capture nonlinearity and interactions.  
- **Hyperparameter tuning**: **GridSearchCV** over `n_estimators`, `max_depth`, `min_samples_leaf`, with RMSE and cross-validation.  
- **Evaluation**: Compare tuned model RMSE and R² to the baseline on a held-out test set and inspect feature importances.

