def verify(X_test,y_test):
    from pipeline.evaluate import evaluate
    import numpy as np
    import os
    folder_path='./data'
    X_test = np.load(os.path.join(folder_path, 'X_test.npy'))
    y_test = np.load(os.path.join(folder_path, 'y_test.npy'))

    accuracy , precision , recall , F1_score=evaluate(X_test,y_test)
    assert accuracy > 0.8 
    assert float(precision) > 0.8
    assert float(recall) >0.8
    assert float(F1_score)>0.8