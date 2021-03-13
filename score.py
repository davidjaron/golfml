import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingRegressor
# from sklearn.ensemble import VotingRegressor
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import TheilSenRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge

def main():
    folds = 10
    test_set_size = int(684/folds)
    test_set_start = 0

    dataset = pd.read_csv("past_players_season_stats.csv", header=None)
    dataset = dataset.sample(frac=1)
    dataset = dataset.dropna()
    # train = train.drop([8], 1)
    dataset = dataset.drop([0], 1)

    models = [KernelRidge(kernel="polynomail"), KNeighborsRegressor(), LinearRegression(), KernelRidge(), ElasticNet(), GradientBoostingRegressor(), AdaBoostRegressor(), HistGradientBoostingRegressor(), RandomForestRegressor(), GaussianProcessRegressor(),
    HuberRegressor(), PassiveAggressiveRegressor(), RANSACRegressor(), TheilSenRegressor(), Ridge(), Lasso(),  
    MLPRegressor(), DecisionTreeRegressor(), SVR()]
    for model in models:
        clf = BaggingRegressor(model, n_estimators=1000, max_samples=.5)
        results = []
        test_set_start = 0
        for i in range(folds):
            test_set = dataset.iloc[test_set_start:(test_set_size + test_set_start)]
            training_set = pd.concat([dataset.iloc[0:test_set_start], dataset.iloc[(test_set_size + test_set_start):len(dataset)]])
            test_results = test_set[8]
            train_results = training_set[8]
            test_set = test_set.drop([8], 1)
            training_set = training_set.drop([8], 1)

            scaler = preprocessing.MinMaxScaler()
            X_train_scaled = scaler.fit_transform(training_set)
            X_test_Scaled = scaler.transform(test_set)

            clf.fit(X_train_scaled, train_results)
            # predict = clf.predict(X_test_Scaled)
            score = clf.score(X_test_Scaled, test_results)
            print(score)
            results.append(score)
            test_set_start += test_set_size
    
        print(model)
        print("Average for model: " + str(sum(results)/len(results)))




main()