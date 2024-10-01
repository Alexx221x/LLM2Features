# Age Group: Categorize age into groups
# Usefulness: Age groups may correlate with survival rates, capturing different experiences or maturity levels among students.
# Input samples: 'age': [18, 20, 22]
df['age_group'] = pd.cut(df['age'], bins=[17, 19, 21, 23, 25], labels=['18-19', '20-21', '22-23', '24-25'])

# On-Campus Resident Flag: Indicates if the student lives on campus
# Usefulness: Residential status may impact access to resources and social support, affecting survival.
# Input samples: 'residential_status': ['On-Campus', 'Off-Campus', 'Off-Campus']
df['on_campus'] = df['residential_status'] == 'On-Campus'

# High Academic Pressure Flag: Indicates if the student feels high academic pressure
# Usefulness: High academic pressure may negatively impact mental health and survival.
# Input samples: 'academic_pressure': [5, 3, 4]
df['high_academic_pressure'] = df['academic_pressure'] >= 4

# Sleep Deprivation Flag: Indicates if the student sleeps less than 7 hours
# Usefulness: Sleep deprivation can affect mental health and academic performance, impacting survival.
# Input samples: 'average_sleep': ['<5 hrs', '7-8 hrs', '5-6 hrs']
df['sleep_deprived'] = df['average_sleep'].isin(['<5 hrs', '5-6 hrs'])

# Mental Health Concern Flag: Indicates if the student shows signs of mental health concerns
# Usefulness: Mental health issues can significantly affect a student's likelihood to continue their studies.
# Input samples: 'depression': [3, 1, 4], 'anxiety': [3, 2, 5], 'isolation': [2, 1, 4]
df['mental_health_concern'] = ((df['depression'] + df['anxiety'] + df['isolation']) >= 9)

# Year of Study Numeric: Converts academic year to numeric
# Usefulness: Numeric representation of academic year allows for ordinal comparison and may correlate with survival.
# Input samples: 'academic_year': ['1st year', '2nd year', '3rd year']
year_mapping = {'1st year': 1, '2nd year': 2, '3rd year': 3, '4th year': 4}
df['year_of_study'] = df['academic_year'].map(year_mapping)

# Engaged in Sports Flag: Indicates if the student engages in sports activities
# Usefulness: Participation in sports may affect social integration and stress levels, impacting survival.
# Input samples: 'sports_engagement': ['7+ times', 'Never', '1-3 times']
df['engaged_in_sports'] = df['sports_engagement'] != 'Never'

# Financial and Future Insecurity Score: Combines financial concerns and future insecurity
# Usefulness: Financial stress and insecurity about the future may compound to affect survival.
# Input samples: 'financial_concerns': [4, 2, 5], 'future_insecurity': [3, 1, 4]
df['financial_future_stress'] = df['financial_concerns'] + df['future_insecurity']

# The 'campus_discrimination' column is dropped because it has low variance and may not contribute significantly to the model, helping to prevent overfitting.
df.drop(columns=['campus_discrimination'], inplace=True)
