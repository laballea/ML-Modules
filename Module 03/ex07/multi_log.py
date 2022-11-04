from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import yaml
import sys, getopt
from ml42.utils_ml import data_spliter
from ml42.mylogisticregression import MyLogisticRegression as myLR
from ml42.utils_ml import normalize

def display(x, y, y_hat):
    error = np.mean(y != y_hat) * 100
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    colors = {0:"b", 1:"r", 2:"g", 3:"cyan"}
    ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=pd.DataFrame(y, columns=["Origin"])["Origin"].map(colors), label="true value")
    ax.scatter(x[:, 0], x[:, 1], x[:, 2], marker="x", c=pd.DataFrame(y_hat, columns=["Origin"])["Origin"].map(colors), label="predicted value")
    ax.set_xlabel('weight')
    ax.set_ylabel('height')
    ax.set_zlabel('bone_density')
    ax.set_title("{:.2f}%".format(error))
    ax.legend()
    plt.show()

def format(arr, zipcode):
    copy = arr.copy()
    copy[:, 0][copy[:, 0] != zipcode] = -1
    copy[:, 0][copy[:, 0] == zipcode] = 1
    copy[:, 0][copy[:, 0] == -1] = 0  
    return copy

def format_result(arr):
    result = []
    for index, row in arr.iterrows():
        result.append(row.idxmax())
    result = np.array(result).reshape(-1, 1)
    return result

def exec(slr_planets, slr_):
    X = np.array(slr_[["weight", "height", "bone_density"]])
    Y = np.array(slr_planets[["Origin"]])
    result = pd.DataFrame()
    x_n = normalize(X)
    x_train, x_test, y_train, y_test = data_spliter(np.array(X), np.array(Y), 0.7)
    for zipcode in range(4):
        cp_y_train = format(y_train, zipcode)
        x_test = normalize(x_test)
        x_train = normalize(x_train)
        my_lr = myLR(np.ones(X.shape[1] + 1).reshape(-1, 1), 0.01, 50000)
        my_lr.fit_(x_train, cp_y_train)
        y_hat = my_lr.predict_(x_n)
        result[zipcode] = y_hat.reshape(len(y_hat))

    display(x_n, Y, format_result(result))

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "", [])
    except getopt.GetoptError as inst:
        print(inst)
        sys.exit(2)
    with open("solar_system_census_planets.csv", "r") as stream:
        try:
            slr_planets = pd.read_csv(stream)
        except yaml.YAMLError as exc:
            print(exc)
    with open("solar_system_census.csv", "r") as stream:
        try:
            slr_ = pd.read_csv(stream)
        except yaml.YAMLError as exc:
            print(exc)
    exec(slr_planets, slr_)

if __name__ == "__main__":
    main(sys.argv[1:])