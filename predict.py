import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
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
# XGBoost, random forest, 
def main():
    models = [KNeighborsRegressor(), ElasticNet(), KernelRidge(), LinearRegression(), GradientBoostingRegressor(), HuberRegressor(), KernelRidge(kernel="polynomial"), KernelRidge(kernel="laplacian")]
    all_predictions = []
    for model in models:
        clf = bagging = BaggingRegressor(model, n_estimators=1000, max_samples=.5)
        train = pd.read_csv("past_players_season_stats.csv", header=None)
        train = train.sample(frac=1)
        train = train.dropna()
        results = train[8]
        names = train[0]
        train = train.drop([8], 1)
        train = train.drop([0], 1)

        test = pd.read_csv("current_season_stats.csv", header=None)
        test = test.dropna()
        new_names = test[0]
        test = test.drop([0], 1)

        scaler = preprocessing.MinMaxScaler()
        X_train_scaled = scaler.fit_transform(train)
        X_test_Scaled = scaler.transform(test)


        clf.fit(X_train_scaled, results)
        predict = clf.predict(X_test_Scaled)
        

        predictions = test.join(new_names)
        predictions["points"] = np.asarray(predict)
        predictions = predictions.drop(predictions.columns.difference(['points', 0]), 1)
        all_predictions.append(predictions)


    predictions = pd.concat(all_predictions)
    predictions = predictions.groupby([0]).mean()
    predictions.sort_values(by="points", ascending=False).to_csv("final_result.csv")
    predictions = predictions.reset_index()

    salaries = pd.read_csv("DKSalaries.csv")
    salaries = salaries.set_index(['Name']) 
    salaries = salaries.drop(salaries.columns.difference(['Salary']), 1)
    predictions = predictions.set_index([0])

    projections = predictions.join(salaries)
    projections = projections.dropna()
    projections.reset_index()

    projections.to_csv("projections.csv", header=False)



main()