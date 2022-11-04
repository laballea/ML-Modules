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
    colors = {1:"b", 0:"r"}
    ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=pd.DataFrame(y, columns=["Origin"])["Origin"].map(colors), label="true value")
    ax.scatter(x[:, 0], x[:, 1], x[:, 2], marker="x", c=pd.DataFrame(y_hat, columns=["Origin"])["Origin"].map(colors), label="predicted value")
    ax.set_xlabel('weight')
    ax.set_ylabel('height')
    ax.set_zlabel('bone_density')
    ax.set_title("{:.2f}%".format(error))
    
    fig, axis = plt.subplots(3, 1)
    fig.suptitle("{:.2f}%".format(error), fontsize=14)
    fig.text(0.04, 0.5, 'planets', va='center', rotation='vertical')
    for idx, i in enumerate(["weight","height","bone_density"]):
        axis[idx].scatter(x[:, idx], y, c="r", label="true value")
        axis[idx].scatter(x[:, idx], y_hat, c="b", label="predicted value")
        axis[idx].set_xlabel(i)

    plt.show()


def exec(slr_planets, slr_, zipcode):
    X = np.array(slr_[["weight", "height", "bone_density"]])
    Y = np.array(slr_planets[["Origin"]])
    Y[:,0][Y[:,0] != zipcode] = -1
    Y[:,0][Y[:,0] == zipcode] = 1
    Y[:,0][Y[:,0] == -1] = 0
    x_train, x_test, y_train, y_test = data_spliter(np.array(X), np.array(Y), 0.7)
    x_test = normalize(x_test)
    x_train = normalize(x_train)
    my_lr = myLR(np.ones(X.shape[1] + 1).reshape(-1, 1), 0.01, 100000)
    my_lr.fit_(x_train, y_train)
    y_hat = np.around(my_lr.predict_(x_test))
    display(x_test, y_test, y_hat)

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "", ["zipcode="])
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
    zipcode = None
    for opt, arg in opts:
        if (opt == "--zipcode"):
            zipcode = int(arg)
    if (zipcode is None or not (0 <= zipcode <= 3)):
        print("usage: mono_log.py --zipcode=[0-3]")
    else:
        exec(slr_planets, slr_, zipcode)


if __name__ == "__main__":
    main(sys.argv[1:])