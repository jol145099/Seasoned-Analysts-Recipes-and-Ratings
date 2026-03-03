# Checkpoint 2

## Part A

### 1. For step 4, what are two hypotheses you have tested, and what were the result?

- Hypothesis 1
  
(1) **Null hypothesis**: Recipes above and below the median ingredient count come from the SAME calorie distribution (any observed difference in median calories is due to random chance).

(2) **Alternative Hypothesis**: Recipes with more ingredients have a DIFFERENT median calorie count than recipes with fewer ingredients.

(3) **Result**: From the result, the observed statistics are 113.30 KCals. Since the p-value is less than the significance level of 0.05, we concluded that there is statistical evidence that the recipes with more ingredients have a different median calorie count than recipes with fewer ingredients.

- Hypothesis 2

(1) **Null hypothesis**: Recipes with quick (<=30 mins) and slow (> 30 mins) preparation time have the SAME average rating.

(2) **Alternative hypothesis**: Recipes with quick (<=30 mins) preparation time have a HIGHER average rating than recipes with slow (> 30 mins) preparation time.

(3) **Result**: From the result, the observed test statistic is 0.03499. Since the p-value is less than the significance level of 0.05, we concluded that there is statistical evidence that the quick recipes have higher average ratings.

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
