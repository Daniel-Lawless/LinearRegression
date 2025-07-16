# Extract the continuous column names from X_train
continuous_columns = X_train.select_dtypes(include = ['float64']).columns

# Allow our scalar to learn the mean and standard deviation from the continuous columns of the X_train set.
scaler = StandardScaler()
scaler.fit(X_train[continuous_columns])

# Apply this scalar to our X_train and X_test sets.
scaled_X_train_features = scaler.transform(X_train[continuous_columns])
scaled_X_test_features = scaler.transform(X_test[continuous_columns])

# Convert back to dataframes.
scaled_X_train_features = pd.DataFrame(scaled_X_train_features, columns = continuous_columns)
scaled_X_test_features = pd.DataFrame(scaled_X_test_features, columns = continuous_columns)

# Get the non_continuous columns of our train and test sets.
X_train_rest = X_train.drop(columns = continuous_columns)
X_test_rest = X_test.drop(columns = continuous_columns)

# Concatenate the two sets together to get the full X_train and X_test scaled sets with the categorical columns included.
X_train_full = pd.concat([scaled_X_train_features, X_train_rest], axis = 1)
X_test_full = pd.concat([scaled_X_test_features, X_test_rest], axis = 1)