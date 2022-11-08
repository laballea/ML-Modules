from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import yaml
import sys
import getopt
from ml42.utils_ml import data_spliter
from ml42.utils_ml import add_polynomial_features
from ml42.ridge import MyRidge
from ml42.utils_ml import normalize


def train_model(yml_models, data, alpha, rate):
    X = np.array(data[yml_models["data"]["x_head"]])
    Y = np.array(data[yml_models["data"]["y_head"]])
    x_train, x_test, y_train, y_test = data_spliter(X, Y, 0.7)
    x_test = normalize(x_test)
    x = normalize(x_train)
    y = y_train
    models = yml_models["models"][yml_models["data"]["best_model"]]
    x_ = add_polynomial_features(x, models["power_x"])
    x_test_ = add_polynomial_features(x_test, models["power_x"])
    theta = np.array(models["theta"]).reshape(-1, 1)

    my_lr = MyRidge(theta, alpha, rate)
    historic = my_lr.fit_(x_, y, historic_bl=True)
    end_rmse = my_lr.rmse_(y_test, my_lr.predict_(x_test_))
    models["rmse"] = end_rmse
    models["theta"] = [int(tta) for tta in my_lr.theta]
    models["total_it"] = int(models["total_it"]) + rate
    models["historic"] = models["historic"] + historic
    with open(yml_models["data"]["name"], 'w') as outfile:
        yaml.dump(yml_models, outfile, default_flow_style=None)


def display(yml_models, data):
    X = np.array(data[yml_models["data"]["x_head"]])
    Y = np.array(data[yml_models["data"]["y_head"]])
    x_train, x_test, y_train, y_test = data_spliter(X, Y, 0.7)
    x_test = normalize(x_test)
    model = yml_models["models"][yml_models["data"]["best_model"]]
    x_test_ = add_polynomial_features(x_test, model["power_x"])

    for lambda_ in np.arange(0.2, 1.2, 0.2):
        theta = np.array(model["theta"][str(lambda_)]).reshape(-1, 1)
        my_lr = MyRidge(theta)
        y_hat = my_lr.predict_(x_test_)
        rmse = my_lr.rmse_(y_test, y_hat)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(x_test[:, 0], x_test[:, 1], x_test[:, 2], marker="o", c="red", label="true value")
        ax.scatter(x_test[:, 0], x_test[:, 1], x_test[:, 2], marker="x", c="blue", label="predicted value")
        ax.set_xlabel('weight')
        ax.set_ylabel('prod_distance')
        ax.set_zlabel('time_delivery')
        ax.set_title("{} / rmse: {}".format(yml_models["data"]["best_model"] + " / " + str(lambda_), rmse))

        figure, axis = plt.subplots(3, 1)
        for idx, title in enumerate(yml_models["data"]["x_head"]):
            axis[idx].scatter(x_test[:, idx], y_test, c="b", marker="o", label="price")
            axis[idx].scatter(x_test[:, idx], y_hat, c="r", marker="x", label="predicted price")
            axis[idx].set_xlabel(title)
            axis[idx].set_ylabel("price")
            figure.suptitle("{} / rmse: {}".format(yml_models["data"]["best_model"] + " / " + str(lambda_), rmse))

    fig = plt.figure()
    ax = fig.add_subplot()
    for lambda_ in model["historic"]:
        ax.plot(np.arange(len(model["historic"][str(lambda_)])), np.sqrt(model["historic"][str(lambda_)]), label=yml_models["data"]["best_model"] + " / " + str(lambda_))
    ax.set_xlabel("number iteration")
    ax.set_ylabel("rmse")
    ax.legend()
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
