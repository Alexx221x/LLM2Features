# Adding 'trip_hour' feature based on 'pickup_datetime'
# Usefulness: The time of day (hour) can influence travel patterns and passenger survival due to factors like traffic conditions and daylight. Knowing the hour can help the classifier make more informed predictions.
# Input samples: 'pickup_datetime': [numpy.datetime64('2016-06-18T15:30:12.000000000'), numpy.datetime64('2016-06-18T08:45:30.000000000'), numpy.datetime64('2016-06-18T23:10:11.000000000')]
df['trip_hour'] = df['pickup_datetime'].dt.hour

# Adding 'pickup_day_of_week' feature based on 'pickup_datetime'
# Usefulness: The day of the week can affect traffic patterns and trip durations, which can influence passenger survival. This feature can help capture these weekly variations.
# Input samples: 'pickup_datetime': [numpy.datetime64('2016-06-18T15:30:12.000000000'), numpy.datetime64('2016-06-13T08:45:30.000000000'), numpy.datetime64('2016-06-15T23:10:11.000000000')]
df['pickup_day_of_week'] = df['pickup_datetime'].dt.dayofweek

# Adding 'is_weekend' feature based on 'pickup_day_of_week'
# Usefulness: Trips during weekends may have different characteristics compared to weekdays (e.g., less traffic, different passenger behavior). This binary feature can help the classifier identify such patterns.
# Input samples: 'pickup_day_of_week': [5, 0, 2]
df['is_weekend'] = df['pickup_day_of_week'].isin([5, 6]).astype(int)

# Adding 'pickup_hour_category' feature by binning 'trip_hour' into categorical time periods
# Usefulness: Categorizing hours into broader time periods (e.g., morning, afternoon, evening) can help the classifier generalize better by reducing noise and capturing meaningful patterns.
# Input samples: 'trip_hour': [15, 8, 23]
bins = [0, 6, 12, 18, 24]
labels = ['Night', 'Morning', 'Afternoon', 'Evening']
df['pickup_hour_category'] = pd.cut(df['trip_hour'], bins=bins, labels=labels, right=False)

# Dropping 'store_and_fwd_flag' column
# Explanation: The 'store_and_fwd_flag' column likely has limited relevance to predicting survival in this context and might not provide significant information for the classifier.
df.drop(columns=['store_and_fwd_flag'], inplace=True)
