from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import math
import yaml
import sys
import getopt
from utils.utils_ml import data_spliter
from utils.utils_ml import add_polynomial_features
from utils.mylinearregression import MyLinearRegression as myLR
from utils.utils_ml import normalize


def train_model(yml_models, data, alpha, rate):
    X = np.array(data[yml_models["data"]["x_head"]])
    Y = np.array(data[yml_models["data"]["y_head"]])
    x_train, x_test, y_train, y_test = data_spliter(X, Y, 0.9)
    x_test = normalize(x_test)
    x = normalize(x_train)
    y = y_train
    models = yml_models["models"][yml_models["data"]["best_model"]]
    x_ = add_polynomial_features(x, models["power_x"])
    x_test_ = add_polynomial_features(x_test, models["power_x"])
    theta = np.array(models["theta"]).reshape(-1, 1)

    my_lr = myLR(theta, alpha, rate)
    historic = my_lr.fit_(x_, y, historic_bl=True)
    models["rmse"] = math  .sqrt(historic.pop())
    models["theta"] = [int(tta) for tta in my_lr.theta]
    models["total_it"] = int(models["total_it"]) + rate
    models["historic"] = models["historic"] + historic
    with open(yml_models["data"]["name"], 'w') as outfile:
        yaml.dump(yml_models, outfile, default_flow_style=None)


def display(yml_models, data):
    X = np.array(data[yml_models["data"]["x_head"]])
    Y = np.array(data[yml_models["data"]["y_head"]])
    x_train, x_test, y_train, y_test = data_spliter(X, Y, 0.9)
    x_test = normalize(x_test)
    model = yml_models["models"][yml_models["data"]["best_model"]]
    x_test_ = add_polynomial_features(x_test, model["power_x"])
    theta = np.array(model["theta"]).reshape(-1, 1)
    my_lr = myLR(theta)
    y_hat = my_lr.predict_(x_test_)
    rmse = my_lr.rmse_(y_test, y_hat)
    for idx, title in enumerate(yml_models["data"]["x_head"]):
        plt.figure()
        plt.scatter(x_test[:, idx], y_test, c="b", marker="o", label="price")
        plt.scatter(x_test[:, idx], y_hat, c="r", marker="x", label="predicted price")
        plt.xlabel(title)
        plt.ylabel("price")
        plt.title("{} / rmse: {}".format(yml_models["data"]["best_model"], rmse))
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(np.arange(int(model["total_it"])), np.sqrt(model["historic"]), label=yml_models["data"]["best_model"])
    ax.set_xlabel("number iteration")
    ax.set_ylabel("mse")
    plt.show()


def train(yml_models, data, alpha=0.1, rate=1000):
    train_model(yml_models, data, alpha, rate)


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "a:r:", ["reset", "train", "display"])
    except getopt.GetoptError as inst:
        print(inst)
        sys.exit(2)
    with open("models.yml", "r") as stream:
        try:
            yml_models = yaml.safe_load(stream)
            data = pd.read_csv(yml_models["data"]["path"])
        except yaml.YAMLError as exc:
            print(exc)
    alpha, rate = 0.1, 1000
    for opt, arg in opts:
        if (opt == "-a"):
            alpha = float(arg)
        elif (opt == "-r"):
            rate = int(arg)
    for opt, arg in opts:
        if opt == '--train':
            train(yml_models, data, alpha=alpha, rate=rate)
        elif opt == '--display':
            display(yml_models, data)


if __name__ == "__main__":
    main(sys.argv[1:])
