import os
import pandas as pd

class Model:
    def __init__(self):
        model_path = os.path.join('myapp', 'SVC.pkl')
        self.model = pd.read_pickle(model_path)
        # your code here

    def predict(self, x):
        pred = self.model.predict(x)
        return pred
        '''
        Parameters
        ----------
        x : np.ndarray
            Входное изображение -- массив размера (28, 28)
        Returns
        -------
        pred : str
            Символ-предсказание 
        '''
        # your code here
