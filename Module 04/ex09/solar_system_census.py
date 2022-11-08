from matplotlib import pyplot as plt
import random
import numpy as np
import pandas as pd
import yaml
import sys
import getopt
from ml42.utils_ml import data_spliter
from ml42.utils_ml import add_polynomial_features
from ml42.utils_ml import cross_validation
from ml42.mylogisticregression import MyLogisticRegression as myLR
from ml42.utils_ml import normalize
from ml42.metrics import f1_score_
from tqdm import tqdm


def display_3d(x, y, y_hat, title):
    error = np.mean(y != y_hat) * 100
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    colors = {0: "b", 1: "r", 2: "g", 3: "cyan"}
    ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=pd.DataFrame(y, columns=["Origin"])["Origin"].map(colors), label="true value")
    ax.scatter(x[:, 0], x[:, 1], x[:, 2], marker="x", c=pd.DataFrame(y_hat, columns=["Origin"])["Origin"].map(colors), label="predicted value")
    ax.set_xlabel('weight')
    ax.set_ylabel('height')
    ax.set_zlabel('bone_density')
    ax.set_title("{:.2f}%error - {}".format(error, title))
    ax.legend()


def bar_plot(yml_models):
    fig = plt.figure(figsize=(10, 5))
    for models_name in tqdm(yml_models["models"], leave=False):
        model = yml_models["models"][models_name]
        clr = '#%06X' % random.randint(0, 0xFFFFFF)
        for lambda_ in model["f1_score"]:
            plt.bar(models_name[-2:] + " " + lambda_, model["f1_score"][lambda_], color=clr, width=0.4)
    plt.xlabel("models")
    plt.ylabel("f1_score")
    plt.title("f1_score of each models + lambda")


def f1_score_test_set(yml_models, data):
    X = np.array(data[0][["weight", "height", "bone_density"]])
    Y = np.array(data[1][["Origin"]])
    X = normalize(X)

    x_train, x_test, y_train, y_test = data_spliter(X, Y, 0.7)
    result = {}
    for models_name in tqdm(yml_models["models"], leave=False):
        model = yml_models["models"][models_name]
        result[models_name] = {}
        for lambda_ in tqdm(np.arange(0.2, 1.2, 0.2), leave=False):
            X_poly = add_polynomial_features(x_test, model["power_x"])
            pred = pd.DataFrame()
            for zipcode in range(4):
                theta = np.array(model["theta"][str(lambda_)][zipcode]).reshape(-1, 1)
                my_lr = myLR(theta)
                y_hat = my_lr.predict_(X_poly)
                pred[zipcode] = y_hat.reshape(len(y_hat))
            result[models_name][lambda_] = f1_score_(y_test, format_result(pred))
    return result


def previsu(yml_models, data):
    X = np.array(data[0][["weight", "height", "bone_density"]])
    Y = np.array(data[1][["Origin"]])
    X = normalize(X)

    model = yml_models["models"][yml_models["data"]["best_model"]]
    for lambda_ in tqdm(np.arange(0.2, 1.2, 0.2), leave=False):
        X_poly = add_polynomial_features(X, model["power_x"])
        pred = pd.DataFrame()
        for zipcode in range(4):
            theta = np.array(model["theta"][str(lambda_)][zipcode]).reshape(-1, 1)
            my_lr = myLR(theta)
            y_hat = my_lr.predict_(X_poly)
            pred[zipcode] = y_hat.reshape(len(y_hat))
        pred = format_result(pred)
        display_3d(X, Y, pred, yml_models["data"]["best_model"] + " / " + str(lambda_))


def display(yml_models, data):
    X = np.array(data[0][["weight", "height", "bone_density"]])
    Y = np.array(data[1][["Origin"]])

    bar_plot(yml_models)
    print(f1_score_test_set(yml_models, data))
    previsu(yml_models, data)
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


def one_vs_all(k_folds, alpha, rate, lambda_, model):
    result = pd.DataFrame()
    x_train, y_train, x_test, y_test = k_folds
    for zipcode in range(4):
        cp_y_train = format(y_train, zipcode)
        theta = np.array(model["theta"][str(lambda_)][zipcode]).reshape(-1, 1)
        my_lr = myLR(theta, alpha, rate, lambda_=lambda_)
        my_lr.fit_(x_train, cp_y_train)
        y_hat = my_lr.predict_(x_test)
        model["theta"][str(lambda_)][zipcode] = [float(tta) for tta in my_lr.theta]
        result[zipcode] = y_hat.reshape(len(y_hat))
    return f1_score_(y_test, format_result(result))


def train(yml_models, data, alpha, rate):
    X = np.array(data[0][["weight", "height", "bone_density"]])
    Y = np.array(data[1][["Origin"]])
    X = normalize(X)

    result = pd.DataFrame()
    model = yml_models["models"][yml_models["data"]["best_model"]]
    for lambda_ in tqdm(np.arange(0.2, 1.2, 0.2), leave=False):
        X_poly = add_polynomial_features(X, model["power_x"])
        model["f1_score"][str(lambda_)] = 0
        for k_folds in tqdm(cross_validation(X_poly, Y, int(yml_models["data"]["K"])), leave=False):
            f1_score = one_vs_all(k_folds, alpha, rate, lambda_, model)
            model["f1_score"][str(lambda_)] = model["f1_score"][str(lambda_)] + f1_score
        model["f1_score"][str(lambda_)] = model["f1_score"][str(lambda_)] / int(yml_models["data"]["K"])
    with open(yml_models["data"]["name"], 'w') as outfile:
        yaml.dump(yml_models, outfile, default_flow_style=None)


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "a:r:", ["train", "display"])
    except getopt.GetoptError as inst:
        print(inst)
        sys.exit(2)
    with open("models.yml", "r") as stream:
        try:
            yml_models = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    with open("solar_system_census_planets.csv", "r") as stream:
        try:
            data_y = pd.read_csv(stream)
        except yaml.YAMLError as exc:
            print(exc)
    with open("solar_system_census.csv", "r") as stream:
        try:
            data_x = pd.read_csv(stream)
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
            train(yml_models, [data_x, data_y], alpha=alpha, rate=rate)
        elif opt == '--display':
            display(yml_models, [data_x, data_y])


if __name__ == "__main__":
    main(sys.argv[1:])
