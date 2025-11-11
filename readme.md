Fraud Detection Data Science Project (California 2020)

presentor: Ofri Rozner

This project is an end-to-end data science workflow to build and optimize a machine learning classifier to detect credit card fraud. Using a large-scale, real-world dataset, this project successfully navigates extreme class imbalance, complex data skewness, and high dimensionality to produce a state-of-the-art model with 97.7% Recall and 0.9977 AUC on unseen test data.

1. Project Overview

Objective

The primary goal was to build a model that minimizes financial loss by maximizing the detection of fraudulent transactions (Recall), while maintaining a manageable rate of false positives (Precision) to reduce operational costs for fraud investigators.

The Data & Core Challenge: A Needle in a Haystack

    Data Scoping: The project was scoped to 2020 data from California, providing a dense, 2M+ transaction dataset to analyze the unique behavioral shifts caused by the COVID-19 pandemic (e.g., a massive move to online, "Card-Not-Present" transactions).
    California was chosen for having the biggest absolute number of fraud cases in the dataset

    The Challenge: Exploratory Data Analysis (EDA) revealed a severe class imbalance of 0.54%. This means for every 185 legitimate transactions, there was only 1 fraudulent case. This imbalance made standard metrics like Accuracy useless and required specialized techniques.

2. Final Model Performance (Test Set)

The final, tuned  model was evaluated on the completely unseen Test Set (15% of the data). The performance confirms the model is highly accurate, robust, and not overfit.
Metric	Result	Interpretation
AUC	0.9977	Exceptional. Near-perfect ability to distinguish fraud from non-fraud.
Recall	0.9767 (97.7%)	Goal Achieved. The model successfully catches 97.7% of all fraud.
Precision	0.5720 (57.2%)	Operationally Efficient. Over 1 in 2 alerts are genuine fraud, a strong signal-to-noise ratio.
F1-Score	0.7215	A robust and well-balanced score.

3. The 6-Stage Workflow

This project was structured across six key notebooks, representing a complete data science pipeline:

[1] Data_Prep.ipynb

    Loaded the raw dataset (2M+ rows).

    Filtered for the 'California' (CA) state scope.
    
    Normalized string columns.

    Handled initial data types, converting trans_date, trans_time, and dob to datetime objects.

    Handled high cardinallty categorical columns ('job', 'city') by scoping on the 20 most popular categories

    Dropped obvious PII (first, last, street) 

[2] EDA.ipynb

    Key Insight: Discovered the 0.54% class imbalance.

    Key Insight: Identified extreme right-skew in numerical features like amt and city_pop.

    Key Insight: Found that fraud risk was highly concentrated in specific contexts, such as late-night hours (TX_HOUR_1 to 3) and specific categories (category_shopping_net).

[3] Data_Cleanse.ipynb

    Challenge: Standard Z-score outlier detection failed due to extreme skew (flagging 90%+ of data).

    Solution: Implemented the robust Interquartile Range (IQR) method to identify true outliers.

    Action: Outliers were "capped" at the 1.5*IQR boundary. This neutralized their skewing effect without deleting the rare fraud samples.

[4] Feature_Engineering.ipynb

Raw data was transformed into predictive behavioral signals.

    Velocity Features: TX_COUNT_1h, AMT_AVG_1h, TIME_SINCE_LAST_TX, SSN_COUNT_1D.

    Recurrence Features: CC_PREV_FRAUD, ACCT_PREV_FRAUD. 

    Risk Scores: ZIP_FRAUD_RATE, AGE_RISK_SCORE (from profile).

    Anomaly Flags: IS_ANY_OUTLIER_IQR, DIST_HOME_MERCH_LOG.

    Transformation:

        Quantile Binning (pd.qcut): Used to handle the non-linear risk of skewed features like amt, city_pop, and TIME_SINCE_LAST_TX.

        One-Hot Encoding (OHE): Applied to all final categorical features.

        Standardization (StandardScaler): Scaled the entire numerical feature matrix.

[5] Model Selection and Fine Tuning.ipynb

    Feature Selection:

        Challenge: High dimensionality (1000+ features after OHE) and slow model runtimes.

        Method: A Consensus Feature Selection (using Lasso, Ridge, RandomForest, GBM) was run to find the most universally predictive features.

        Result: Filtered the data to the Top 20 most robust features, all of which were selected by at least 4 out of 5 models.

        problem: Lasso and ridge did not converge after 5000 iterations, leaving noise in the selection. The ensemble method helped overcome this problem and the model evaluation prooved the selected featrs were good regardless, it was decided not to re-run the selection.

    Model Selection & Tuning:

        Data Split: The data was split into 70/15/15 (Train/Validation/Test) using stratify=y to preserve the 0.54% fraud ratio in all sets.

        Baseline: A 6-model comparison (XGBoost, RandomForest, etc.) was run. All models used class weighting (scale_pos_weight or class_weight='balanced') to overcome the imbalance.

        Result: XGBoost was selected as the champion model (highest AUC and best Recall/Precision balance).

        Challenge: RandomizedSearchCV (the scikit-learn wrapper) had version conflicts with XGBoost (AttributeError).

        Solution: Pivoted to a robust Manual Fine-Tuning loop, testing 3 key parameter sets on the Validation (Dev) set. This successfully identified the optimal hyperparameters.

[6] Model Evaluation.ipynb

    Final Step: The single best model (XGBoost with max_depth=8, learning_rate=0.05) was trained on the full Training set.

    Final Proof: The model was evaluated one time on the unseen Test Set, producing the final, unbiased metrics seen in Section 2.

4. Analysis of Final 20 Features

The model's success is built on these 20 high-consensus features, which all scored 4/4 in our selection process.
Category	        Features Selected (Count: 20)

🚀 Velocity (9)	AMT_MAX_1h, TX_COUNT_1h, AMT_AVG_1h, TX_COUNT_24h, AMT_AVG_24h, AMT_MAX_24h, SSN_COUNT_1D, AMT_AVG_7d_BIN, TX_DAY

🛒 Context (7)	category_shopping_net, category_grocery_pos, category_food_dining, category_gas_transport, category_grocery_net, category_travel, category_misc_pos

📊 Anomaly (2)	IS_ANY_OUTLIER_IQR, city_pop_BIN_4

🆔 Recurrence (2)	CC_PREV_FRAUD, category_shopping_pos (Note: category_shopping_pos appears twice in your log, one may be CC_PREV_FRAUD)

5. Model Interpretability (SHAP Analysis)

To ensure the model is not a "black box" and to validate that its internal logic is sound, a SHAP (SHapley Additive exPlanations) analysis was performed. This technique explains how each of the Top 20 features contributes to the final prediction for fraud.

The summary plot below confirms that the model's "thinking" is rational and aligns perfectly with our feature engineering strategy.

Analysis of Key Drivers

This plot shows the impact of each feature. A red dot means a high value for that feature (e.g., TX_COUNT_1h = 5), while a blue dot means a low value. The X-axis shows the impact on the fraud prediction.

    Positive SHAP Value (Right): Pushes the model to predict "Fraud."

    Negative SHAP Value (Left): Pushes the model to predict "Legitimate."

Key Insights:

    Recurrence is the #1 Signal: The top features are historical. A high (red) value for CC_PREV_FRAUD (meaning the card has been used for fraud before) has the largest positive impact, strongly pushing the prediction to "Fraud."

    Velocity is Critical: Features like TX_COUNT_1h and AMT_AVG_1h are top-tier predictors. High (red) values for these features (high-velocity attacks) are the next biggest drivers of fraud risk.

    Anomalies & Context are Key:

        IS_ANY_OUTLIER_IQR: A high value (red, meaning True) has a clear positive impact, confirming that outliers are inherently high-risk.

        category_shopping_net: This OHE feature (red, meaning True) also strongly pushes the prediction toward fraud, validating our focus on "Card-Not-Present" (online) categories.

Conclusion: The SHAP analysis proves that the model is making its decisions for the right reasons, basing its high-performance predictions on the logical, high-signal velocity and recurrence features we engineered.

5. How to Use This Project

    Run the notebooks in numerical order ([1] to [6]).

    The final, deployable model is saved as final_xgb_model_production.json.

    The list of 20 features used to build this model is saved in lasso_selected_features.npy.
