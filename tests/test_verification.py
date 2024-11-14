
import os
import numpy as np
import pytest
from pipeline.evaluate import evaluate



base_dir = os.path.abspath(os.path.dirname(__file__))  # Dossier actuel
data_dir = os.path.join(base_dir, '..', 'data')  # Accéder au dossier data
X_test_path = os.path.join(data_dir, 'X_test.npy')
y_test_path = os.path.join(data_dir, 'y_test.npy')

# Charger les fichiers
X_test = np.load(X_test_path, allow_pickle=True)
y_test = np.load(y_test_path, allow_pickle=True)

# Transformation (si nécessaire)
X_test = X_test.tolist().toarray()


# Fonction de test
def test_performance():
    accuracy, precision, recall, F1_score = evaluate(X_test, y_test)
    assert accuracy > 0.8
    assert float(precision) > 0.8
    assert float(recall) > 0.8
    assert float(F1_score) > 0.8
