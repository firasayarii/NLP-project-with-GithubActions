from pipeline.evaluate import evaluate
import numpy as np
import os
import pytest

X_test = np.load(os.path.join('c:\\Users\\msi\\Documents\\End_to_end_nlp\\data', 'X_test.npy'),allow_pickle=True)
X_test = X_test.tolist().toarray()
y_test = np.load(os.path.join('c:\\Users\\msi\\Documents\\End_to_end_nlp\\data', 'y_test.npy'),allow_pickle=True)
def test_performance():

    accuracy , precision , recall , F1_score=evaluate(X_test,y_test)
    assert accuracy > 0.8 
    assert float(precision) > 0.8
    assert float(recall) >0.8
    assert float(F1_score)>0.8