# What Makes a Recipe High-Calorie? 
# Predicting Nutrition from Recipe Metadata

**Authors:** Baixi Guo, Chiu-Chiu (JoJo) Lin, Ryan Liao

**Project website:** [https://jol145099.github.io/Seasoned-Analysts-Recipes-and-Ratings/](https://jol145099.github.io/Seasoned-Analysts-Recipes-and-Ratings/)

---

## Overview

This project looks at the Food.com Recipes and Ratings dataset from a few different angles. We explore recipe metadata and user ratings, look at missingness, run hypothesis tests, and then build a regression model to predict calorie content using information available when a recipe is submitted. The README summarizes what is currently in the notebook.

---

## Introduction

We worked with the [Food.com Recipes and Ratings dataset](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions), which contains over 80,000 recipes along with a large number of user ratings and reviews.

**Central question:** Can we predict a recipe's calorie content from information available at the time it is submitted, before any user reviews it? We also look at two related questions: do quick recipes (≤30 minutes) tend to get higher average ratings, and do recipes with more ingredients tend to have more calories?

The merged dataset has **234,429 rows** (one per recipe–review pair) and **83,782 unique recipes**. Relevant columns:

| Column | Type | Description |
| :--- | :--- | :--- |
| `minutes` | Quantitative | Stated preparation time (minutes) |
| `n_steps` | Quantitative | Number of recipe steps |
| `n_ingredients` | Quantitative | Number of unique ingredients |
| `rating_average` | Quantitative | Mean star rating across reviews |
| `nutrition_calories` | Quantitative | Total calorie count (kcal) |
| `nutrition_total_fat` | Quantitative | Total fat (% daily value) |
| `nutrition_sugar` | Quantitative | Sugar (% daily value) |
| `nutrition_protein` | Quantitative | Protein (% daily value) |
| `nutrition_carbs` | Quantitative | Carbohydrates (% daily value) |

---

## Data Cleaning and Exploratory Data Analysis

Before doing any analysis, we cleaned the data in a few important ways:

1. **Left-merged recipes and interactions** on `recipe_id`, keeping all recipes (including those with no reviews).
2. **Replaced 0-star ratings with `NaN`** because on Food.com, 0 means the user left a review without selecting a star, not a true zero-star rating.
3. **Computed `rating_average` per recipe** by grouping on `recipe_id`, taking the mean of non-null ratings, and merging back.
4. **Parsed string-encoded list columns** (`tags`, `nutrition`, `steps`, `ingredients`) into Python lists.
5. **Expanded the 7-element nutrition list** into separate numeric columns: `nutrition_calories`, `nutrition_total_fat`, `nutrition_sugar`, `nutrition_sodium`, `nutrition_protein`, `nutrition_sat_fat`, `nutrition_carbs`.

Sample of the cleaned DataFrame (selected columns):

| name | minutes | n_steps | n_ingredients | nutrition_calories | rating_average |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 1 brownies in the world best ever | 40 | 10 | 9 | 138.4 | 4.0 |
| 1 in canada chocolate chip cookies | 45 | 12 | 11 | 595.1 | 5.0 |
| 412 broccoli casserole | 40 | 6 | 9 | 194.8 | 5.0 |
| millionaire pound cake | 120 | 7 | 7 | 878.3 | 5.0 |
| 2000 meatloaf | 90 | 17 | 13 | 267.0 | 5.0 |

---

## Univariate Analysis

We looked at the distributions of key variables.

**Calories:** The distribution is clearly right-skewed. Most recipes fall somewhere around 100 to 800 kcal, but there is a long tail that stretches far past 2,000 kcal. In the notebook, we cap the histogram at 4,000 kcal so the overall shape is easier to read.

**Preparation time (`minutes`):** This variable is also right-skewed, so we restricted the view to recipes under 720 minutes. Looking at these one-variable distributions first helped us get a better sense of the scale of the data before moving into comparisons and modeling.

---

## Bivariate Analysis

Next, we looked at how pairs of variables relate to each other.

**Prep time vs. average rating:** The main bivariate plot in the notebook is a scatter plot of `minutes` vs. `rating_average`. It does not show a strong linear relationship. Ratings stay clustered around 4 to 5 stars no matter how long the recipe takes, which suggests people may care more about whether the recipe turns out well than how fast it is.

More broadly, this plot helped motivate our later hypothesis test comparing quick and slow recipes.

---

## Assessment of Missingness

**NMAR:** The column with the most notable missingness in the notebook analysis is `rating_average`, but the missingness rate is still fairly small at about **1.18%** of rows. A reasonable interpretation is that missing ratings happen when a recipe was never rated by users, which could depend on factors like visibility or popularity that are not directly observed in the data.

**Missingness dependency:** In the notebook, we do a simple comparison of missingness rates for quick versus slow recipes. Quick recipes have a missing-rate of **0.0088**, while slow recipes have a missing-rate of **0.0145**, for a difference of **0.0057**. Based on that comparison, the notebook concludes that slow recipes have a higher proportion of missing ratings.

---

## Hypothesis Testing

**Test 1: Do quick recipes (≤30 min) get higher average ratings?**

- **Null:** Quick and slow recipes have the same mean average rating; any difference is due to chance.
- **Alternative:** Quick recipes have a *higher* mean average rating.
- **Test statistic:** Difference in mean `rating_average` (Quick − Slow). **Significance level:** α = 0.05.
- **Result:** Using a one-sided permutation test with 3,000 repetitions, we got a p-value of **0.00000** and rejected the null hypothesis. In the notebook, this is interpreted as statistical evidence that quick recipes have higher average ratings.

**Test 2: Do recipes with more ingredients (above median) have different median calories?**

- **Null:** Recipes above and below the median ingredient count come from the same calorie distribution (any observed difference in median calories is due to random chance).
- **Alternative:** Recipes with more ingredients have a different median calorie count than recipes with fewer ingredients.
- **Test statistic:** Absolute difference in median `nutrition_calories` (high-ingredient − low-ingredient). **Significance level:** α = 0.05.
- **Result:** The p-value was approximately 0.00, so we rejected the null hypothesis. This suggests that recipes with more ingredients tend to have a meaningfully different, and generally higher, median calorie count.

---

## Framing a Prediction Problem

**Problem:** Predict the **calorie content** (`nutrition_calories`) of a recipe.

**Type:** Regression.

**Response:** `nutrition_calories`. We chose calories because they are one of the easiest nutrition measures for people to understand and compare, and they are useful to know before deciding whether to make a recipe.

**Metric:** RMSE (kcal). We used RMSE because it penalizes large errors more heavily than MAE, which matters here since being very wrong on a high-calorie dish is worse than being slightly off on a lighter recipe.

**Features at time of prediction:** In the notebook, the framing section lists these features as available at submission time: `minutes`, `time_group`, and the nutrition fields `fat`, `sugar`, `sodium`, `protein`, and `carbs`. The main idea is that the model should only use information known before users leave reviews or ratings.

---

## Baseline Model

**Baseline model:** In the notebook, the baseline model is a **Ridge regression** that uses only one feature, `minutes`, to predict `nutrition_calories`. It is trained on an 80/20 train-test split and evaluated using **RMSE**. The reported baseline test RMSE is **588.83 kcal**, which means the model is off by about 589 calories on average.

Even though this is a very simple baseline, it gives us a reference point for seeing whether adding nutritional features improves performance.

---

## Final Model

The final model in the notebook is still a **Ridge regression**, but it uses a much richer set of predictors:

- `minutes`
- `nutrition_total_fat`
- `nutrition_sugar`
- `nutrition_protein`
- `nutrition_carbs`

The notebook tunes the Ridge `alpha` parameter using **GridSearchCV** over `[0.01, 0.1, 1, 10, 100]`. The best value found is **0.01**.

This model performs much better than the baseline:

- **Baseline RMSE:** 588.83 kcal
- **Final RMSE:** 39.79 kcal
- **Improvement:** 549.04 kcal
- **Percent improvement:** 93.24%

This big improvement makes sense because the final model includes nutrition-related predictors that are much more directly connected to calorie content than prep time alone.

---

## Fairness Analysis

**Groups:** Group X = Quick recipes (`minutes ≤ 30`), Group Y = Slow recipes (`minutes > 30`).

**Metric used in the notebook:** Mean Squared Error (MSE), compared across groups.

The notebook's fairness analysis is a simple group comparison rather than a full permutation test. It splits recipes into:

- **Quick:** `minutes ≤ 30`
- **Slow:** `minutes > 30`

Then it compares the model's average squared error across the two groups:

- **Quick recipe MSE:** 2015.13
- **Slow recipe MSE:** 986.82
- **Difference (Quick - Slow):** 1028.30

Based on this result, the notebook concludes that the model performs worse on quick recipes than on slow recipes.
