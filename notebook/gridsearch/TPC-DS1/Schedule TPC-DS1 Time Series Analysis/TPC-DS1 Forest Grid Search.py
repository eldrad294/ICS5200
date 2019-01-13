print('Test')

# #
# # Test Harness
# iteration = 0
# max_lag_param = 15
# test_harness_param = (.5,.6,.7,.8,.9)
# test_n_estimator_param = (10,100,250,500,1000)
# #
# # Test Multiple Time Splits (Lag)
# for lag in range(1,max_lag_param+1):
#     t0 = time.time()
#     shifted_df = series_to_supervised(df, lag, 1)
#     #
#     # Seperate labels from features
#     y_df_column_names = shifted_df.columns[len(df.columns):len(df.columns) + len(y_label)]
#     y_df = shifted_df[y_df_column_names]
#     X_df = shifted_df.drop(columns=y_df_column_names)
#     #
#     # Delete middle timesteps
#     X_df = remove_n_time_steps(data=X_df, n_in=lag)
#     #
#     # Test Multiple Train/Validation Splits
#     for test_split in test_harness_param:
#         X_train, X_validate, y_train, y_validate = train_test_split(X_df, y_df, test_size=test_split)
#         X_train = X_train.values
#         y_train = y_train.values
#         X_validate, X_test, y_validate, y_test = train_test_split(X_validate, y_validate, test_size=.5)
#         X_validate = X_validate.values
#         y_validate = y_validate.values
#         #
#         # Train Multiple Regression Forest Models using various estimators
#         for n_estimators in test_n_estimator_param:
#             model = RandomForest(mode='regression',
#                                  n_estimators=n_estimators,
#                                  parallel_degree=parallel_degree)
#             model.fit_model(X_train=X_train,
#                             y_train=y_train)
#             rmse = model.predict_and_evaluate(X=X_validate,
#                                               y=y_validate,
#                                               y_labels=y_label,
#                                               plot=False)
#             t1 = time.time()
#             time_total = t1 - t0
#             RandomForest.write_results_to_disk(path="time_series_random_forest_regression_results.csv",
#                                                iteration=iteration,
#                                                lag=lag,
#                                                test_split=test_split,
#                                                estimator=n_estimators,
#                                                score=rmse,
#                                                time_train=time_total)
#             iteration += 1