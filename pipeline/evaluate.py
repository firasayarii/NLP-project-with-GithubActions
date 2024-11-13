# Parcourir tous les fichiers .joblib dans le dossier
def evaluate(X_test,y_test):
    from pathlib import Path
    import joblib
    from sklearn.metrics import accuracy_score, classification_report , precision_score, recall_score , f1_score
    # Chemin du dossier contenant les fichiers
    folder_path = Path('./models')
    for file_path in folder_path.glob('*.joblib') :

        model = joblib.load('models\\logistic_regression.joblib')
        y_pred=model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall =recall_score(y_test, y_pred)
        F1_score=f1_score(y_test, y_pred)
        return accuracy , float(precision) , float(recall) , float(F1_score)

