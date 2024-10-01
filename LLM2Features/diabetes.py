# (HighRiskPregnanciesFlag: Flag indicating high risk pregnancies (more than 5 pregnancies))
# Usefulness: Multiple pregnancies may indicate high risk, which could correlate with diabetes outcomes.
# Input samples: 'Pregnancies': [13, 4, 3]
df['HighRiskPregnanciesFlag'] = df['Pregnancies'] > 5

# (GlucoseLevelCategory: Categorize glucose levels into Low, Normal, High)
# Usefulness: Different glucose level categories can help in identifying diabetes status.
# Input samples: 'Glucose': [129, 105, 85]
df['GlucoseLevelCategory'] = pd.cut(df['Glucose'], bins=[0, 70, 140, 200], labels=['Low', 'Normal', 'High'])

# (BloodPressureFlag: Flag indicating high blood pressure (above 80))
# Usefulness: High blood pressure is a common comorbidity with diabetes.
# Input samples: 'BloodPressure': [90, 70, 85]
df['BloodPressureFlag'] = df['BloodPressure'] > 80

# (BMI_Category: Categorize BMI into Underweight, Normal, Overweight, and Obese)
# Usefulness: BMI categories provide insight into the patient's weight status which is crucial for diabetes risk.
# Input samples: 'BMI': [21.8, 25.3, 30.5]
df['BMI_Category'] = pd.cut(df['BMI'], bins=[0, 18.5, 24.9, 29.9, 50], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])

# (YoungFlag: Flag indicating if the person is below 30 years old)
# Usefulness: Age can be a significant factor in the prevalence of diabetes.
# Input samples: 'Age': [22, 35, 28]
df['YoungFlag'] = df['Age'] < 30

# (Drop SkinThickness and Insulin columns: Due to potential unreliability and irrelevance in this context)
df.drop(columns=['SkinThickness', 'Insulin'], inplace=True)

# (DiabetesPedigreeFlag: Flag indicating a high diabetes pedigree function (>0.5))
# Usefulness: High diabetes pedigree function indicates a higher likelihood of diabetes based on family history.
# Input samples: 'DiabetesPedigreeFunction': [0.155, 0.672, 0.501]
df['DiabetesPedigreeFlag'] = df['DiabetesPedigreeFunction'] > 0.5

# (AgeGroup: Categorize age into Young, Middle-aged, Senior)
# Usefulness: Different age groups have different risk profiles for diabetes.
# Input samples: 'Age': [22, 45, 67]
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 30, 60, 100], labels=['Young', 'Middle-aged', 'Senior'])

# (Normalize Glucose levels for better model performance)
# Usefulness: Normalized values can improve the performance of some algorithms.
# Input samples: 'Glucose': [129, 105, 85]
df['GlucoseNormalized'] = (df['Glucose'] - df['Glucose'].min()) / (df['Glucose'].max() - df['Glucose'].min())
