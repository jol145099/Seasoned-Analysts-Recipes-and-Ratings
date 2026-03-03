# Checkpoint 2

## Part A

### 1. For step 4, what are two hypotheses you have tested, and what were the result?
**Null hypothesis**: Recipes above and below the median ingredient count come from the same calorie distribution (any observed difference in median calories is due to random chance).
**Alternative Hypothesis**: Recipes with more ingredients have a different median calorie count than recipes with fewer ingredients.
**Result**: Yielded p-value as 0.00, and conclude to reject the null hypothesis and favor toward the alternative hypothesis that recipes with more ingredients have a different median calorie count than recipes with fewer ingredients.

## Part B

### 2. Briefly explain your baseline model and your plans for improving the model.

**Baseline model**  
A **linear regression** predicts `calories` from `n_steps` and `n_ingredients` (always available at submission). The pipeline uses `StandardScaler` and `LinearRegression`, evaluated with **RMSE** (kcal) and **R²**. With only two linear predictors, the baseline is interpretable but leaves much variance unexplained and yields fairly large errors.

**Plans for improving the model**  

- **Richer features**: Add `log_minutes`, `n_tags`, and binary flags like `is_dessert` and `is_quick`.  
- **Nonlinear model**: Switch to a **Random Forest Regressor** to capture nonlinearity and interactions.  
- **Hyperparameter tuning**: **GridSearchCV** over `n_estimators`, `max_depth`, `min_samples_leaf`, with RMSE and cross-validation.  
- **Evaluation**: Compare tuned model RMSE and R² to the baseline on a held-out test set and inspect feature importances.

## Part C

### 3. Submit a working GitHub page webpage URL for the project. On the webpage, you need to at least include a project title.

**Link**: https://jol145099.github.io/Seasoned-Analysts-Recipes-and-Ratings/