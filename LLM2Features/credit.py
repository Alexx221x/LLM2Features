# (Feature: `credit_history_flag` - Indicates if the credit history is critical or not)
# Usefulness: This flag can help identify risky applicants with critical credit history, which is an important indicator for creditworthiness.
# Input samples: 'credit_history': ['critical/other existing credit', 'existing paid', 'no credits/all paid']
df['credit_history_flag'] = df['credit_history'].apply(lambda x: 1 if 'critical' in x else 0)

# (Feature: `savings_status_flag` - Indicates if the savings status is low)
# Usefulness: Applicants with lower savings may be more likely to default on loans, making this a useful feature.
# Input samples: 'savings_status': ['no checking', '<100', '>=1000']
df['savings_status_flag'] = df['savings_status'].apply(lambda x: 1 if '<100' in x else 0)

# (Feature: `employment_duration_flag` - Indicates if the employment duration is less than 1 year)
# Usefulness: Short employment duration may indicate job instability, which is a risk factor for loan default.
# Input samples: 'employment': ['unemployed', '<1', '>=7']
df['employment_duration_flag'] = df['employment'].apply(lambda x: 1 if '<1' in x else 0)

# (Feature: `high_installment_commitment_flag` - Indicates if the installment commitment is high)
# Usefulness: High installment commitments can indicate financial strain, which is important for predicting loan defaults.
# Input samples: 'installment_commitment': [1.0, 2.0, 4.0]
df['high_installment_commitment_flag'] = df['installment_commitment'].apply(lambda x: 1 if x > 3 else 0)

# (Feature: `age_group` - Categorizes age into groups)
# Usefulness: Different age groups may have different risk profiles. Younger and older applicants might have different default probabilities.
# Input samples: 'age': [24.0, 45.0, 62.0]
df['age_group'] = pd.cut(df['age'], bins=[0, 25, 45, 65, 100], labels=['young', 'adult', 'middle_aged', 'senior'])

# (Dropping `foreign_worker` column - deemed not useful for prediction)
# Explanation: The 'foreign_worker' column might not add significant value to the prediction model and can be dropped to reduce complexity.
df.drop(columns=['foreign_worker'], inplace=True)

# (Feature: `credit_amount_log` - Log transformation of credit amount)
# Usefulness: Log transformation can help normalize the distribution of credit amount, which often has a long tail.
# Input samples: 'credit_amount': [6187.0, 8000.0, 3000.0]
df['credit_amount_log'] = np.log1p(df['credit_amount'])

# (Feature: `duration_group` - Categorizes duration into groups)
# Usefulness: Different loan durations might have different default risks. Categorizing can help in better segmentation.
# Input samples: 'duration': [30.0, 15.0, 40.0]
df['duration_group'] = pd.cut(df['duration'], bins=[0, 12, 24, 36, 48, 60, 120], labels=['very_short', 'short', 'medium', 'long', 'very_long', 'extra_long'])

# (Feature: `property_value` - Assigns a numeric value to property magnitude)
# Usefulness: Different property values can impact the credit risk. Assigning numeric values helps in quantitative analysis.
# Input samples: 'property_magnitude': ['car', 'real estate', 'life insurance']
property_value_mapping = {'real estate': 3, 'life insurance': 2, 'car': 1, 'no known property': 0}
df['property_value'] = df['property_magnitude'].map(property_value_mapping)

# (Feature: `housing_own_flag` - Indicates if the housing status is 'own')
# Usefulness: Owning a house can be a positive indicator of financial stability.
# Input samples: 'housing': ['own', 'rent', 'for free']
df['housing_own_flag'] = df['housing'].apply(lambda x: 1 if x == 'own' else 0)


# (Feature: `existing_credits_flag` - Indicates if there are multiple existing credits)
# Usefulness: Having multiple existing credits can indicate higher financial burden.
# Input samples: 'existing_credits': [1.0, 2.0, 3.0]
df['existing_credits_flag'] = df['existing_credits'].apply(lambda x: 1 if x > 1 else 0)


# (Feature: `num_dependents_flag` - Indicates if there are multiple dependents)
# Usefulness: Having multiple dependents can indicate higher financial responsibilities.
# Input samples: 'num_dependents': [1.0, 2.0, 3.0]
df['num_dependents_flag'] = df['num_dependents'].apply(lambda x: 1 if x > 1 else 0)


# (Feature: `own_telephone_flag` - Indicates if the applicant has a telephone)
# Usefulness: Having a telephone can indicate better accessibility and stability.
# Input samples: 'own_telephone': ['none', 'yes']
df['own_telephone_flag'] = df['own_telephone'].apply(lambda x: 1 if x == 'yes' else 0)
