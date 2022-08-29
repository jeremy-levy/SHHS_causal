x_columns = ['age', 'sex', 'race', 'ethnicity', 'bmi', 'smoking_status', 'window_begin']
x_columns_continuous = ['age', 'bmi', 'window_begin']
x_columns_categorical = ['sex', 'race', 'ethnicity', 'smoking_status']

y_column = ['outcome']
event_columns = ['event_name', 'event_start', 'event_duration', 'patient']
t_baseline = ['CA_relative_3']
t_autoencoder = ['autoencoder_treatment']
t_pca = ['pca_treatment']
