
import os
import numpy as np
import pytest
from pipeline.evaluate import evaluate

# Fonction pour obtenir le chemin absolu du répertoire courant
def get_data_path(filename):
    # Résout le chemin relatif par rapport au répertoire courant (projet racine)
    project_root = os.path.dirname(os.path.abspath(__file__))  # Répertoire courant du script
    data_path = os.path.join(project_root, 'data', filename)  # Ajoute le chemin du fichier
    return data_path

# Charger les données
X_test = np.load(get_data_path('X_test.npy'), allow_pickle=True)
X_test = X_test.tolist().toarray()
y_test = np.load(get_data_path('y_test.npy'), allow_pickle=True)

# Fonction de test
def test_performance():
    accuracy, precision, recall, F1_score = evaluate(X_test, y_test)
    assert accuracy > 0.8
    assert float(precision) > 0.8
    assert float(recall) > 0.8
    assert float(F1_score) > 0.8
