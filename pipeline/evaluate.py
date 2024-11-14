# Parcourir tous les fichiers .joblib dans le dossier
def evaluate(X_test,y_test):
    from pathlib import Path
    import joblib
    import os
    from sklearn.metrics import accuracy_score, classification_report , precision_score, recall_score , f1_score
    # Chemin du dossier contenant les fichiers
    base_dir = os.path.abspath(os.path.dirname(__file__))  # Dossier actuel
    model_dir = os.path.join(base_dir, '..', 'models')  # Accéder au dossier data
    model_path = os.path.join(model_dir, 'logistic_regression.joblib')
    model = joblib.load(model_path)
    y_pred=model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall =recall_score(y_test, y_pred)
    F1_score=f1_score(y_test, y_pred)
    return accuracy , float(precision) , float(recall) , float(F1_score)

