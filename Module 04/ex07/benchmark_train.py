from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from ml42.utils_ml import add_polynomial_features
from ml42.ridge import MyRidge
from ml42.utils_ml import normalize
from ml42.utils_ml import cross_validation
import yaml
import math
import sys
import getopt


def init_models(yml_models, data):
    x_head = yml_models["data"]["x_head"]
    yml_models["data"]["quartil"] = [-1, -1, -1]
    yml_models["data"]["best_model"] = None
    pow = range(1, yml_models["data"]["max_poly"] + 1)
    # combinations = np.array(list(itertools.product(list(itertools.product(pow)), repeat=len(x_head)))) # all combinations too long
    combinations = [np.full((len(x_head), 1), po) for po in pow]
    for comb in combinations:
        name = ''.join(["{}**{}_".format(x_name[:3], str(po[0])) for x_name, po in zip(x_head, comb)])
        yml_x = list([int(po[0]) for po in comb])
        if (sum(yml_x) != 0):
            yml_models["models"][name] = {
                "power_x": yml_x,
                "rmse": None,
                "total_it": 0,
                "alpha": 1e-1,
                "historic": {},
                "theta": {}
            }
            for lambda_ in tqdm(np.arange(0.2, 1.2, 0.2), leave=False):
                yml_models["models"][name]["historic"][str(lambda_)] = []
                yml_models["models"][name]["theta"][str(lambda_)] = [1 for _ in range(sum(yml_x) + 1)]
    with open(yml_models["data"]["name"], 'w') as outfile:
        yaml.dump(yml_models, outfile, default_flow_style=None)


def train_models(yml_models, data, alpha, rate):
    X = np.array(data[yml_models["data"]["x_head"]])
    Y = np.array(data[yml_models["data"]["y_head"]])
    X = normalize(X)

    for models_name in tqdm(yml_models["models"], leave=False):
        models = yml_models["models"][models_name]
        for lambda_ in tqdm(np.arange(0.2, 1.2, 0.2), leave=False):
            theta = np.array(models["theta"][str(lambda_)]).reshape(-1, 1)
            X_poly = add_polynomial_features(X, models["power_x"])
            for k_folds in tqdm(cross_validation(X_poly, Y, 10), leave=False):
                my_lr = MyRidge(theta, alpha, rate, lambda_)
                x_train, y_train, x_test, y_test = k_folds
                historic = my_lr.fit_(x_train, y_train, historic_bl=True)
                models["historic"][str(lambda_)] = models["historic"][str(lambda_)] + historic
            models["theta"][str(lambda_)] = [tta for tta in my_lr.theta]
        models["rmse"] = float(math.sqrt(np.mean([models["historic"][lambda_].pop() for lambda_ in models["historic"]])))
        models["total_it"] = int(models["total_it"]) + rate
    with open(yml_models["data"]["name"], 'w') as outfile:
        yaml.dump(yml_models, outfile, default_flow_style=None)


def best_models(yml_models):
    rmse_list = np.array([[str(key), float(value["rmse"])] for key, value in yml_models["models"].items()])
    yml_models["data"]["best_model"] = str(rmse_list[rmse_list[:, 1].astype('float64').argmin()][0])
    yml_models["data"]["quartil"] = [float(np.quantile(rmse_list[:, 1].astype('float64'), quart)) for quart in np.arange(0.25, 1, 0.25)]
    with open(yml_models["data"]["name"], 'w') as outfile:
        yaml.dump(yml_models, outfile, default_flow_style=None)


def train(yml_models, data, alpha=0.1, rate=1000):
    train_models(yml_models, data, alpha, rate)
    best_models(yml_models)


def display(yml_models, data):
    fig = plt.figure()
    ax = fig.add_subplot()
    clr_lst = ["red", "blue", "green", "cyan"]
    for values, clr in zip(yml_models["models"].items(), clr_lst):
        key, model = values
        for lambda_ in model["historic"]:
            ax.plot(np.arange(len(model["historic"][lambda_])), np.sqrt(model["historic"][lambda_]), color=clr, label=key)
            # break
    ax.set_xlabel("number iteration")
    ax.set_ylabel("rmse")
    ax.legend()
    plt.show()


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
        if opt == '--reset':
            init_models(yml_models, data)
        elif opt == '--train':
            train(yml_models, data, alpha=alpha, rate=rate)
        elif opt == '--display':
            display(yml_models, data)


if __name__ == "__main__":
    main(sys.argv[1:])
