import pandas as pd
import numpy as np
import sklearn
from utils import Utils
from models import Models
import warnings
warnings.simplefilter("ignore")
if __name__ == "__main__":
    utils = Utils()
    models = Models()
    data = utils.load_from_csv('./in/agosto_severidad.csv')
    X, y = utils.features_target(data,['severidad'], ['severidad'])
    models.grid_training(X,y)
    #print(data)

