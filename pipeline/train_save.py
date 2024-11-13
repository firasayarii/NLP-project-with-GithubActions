# Save the model to a specific path
def train_and_save(X,y):
    import joblib
    from sklearn.linear_model import LogisticRegression
    # Initialize and train the model
    model = LogisticRegression(max_iter=1000)
    model.fit(X,y)
    joblib.dump(model, 'models/linear_regression.joblib')

