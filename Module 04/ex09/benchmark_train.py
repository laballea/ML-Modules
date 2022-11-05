from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import yaml
import sys, getopt
from ml42.utils_ml import data_spliter
from ml42.utils_ml import add_polynomial_features
from ml42.utils_ml import cross_validation
from ml42.mylogisticregression import MyLogisticRegression as myLR
from ml42.utils_ml import normalize
from ml42.metrics import f1_score_
from tqdm import tqdm

def init_models(yml_models, data):
    x_head = yml_models["data"]["x_head"]
    yml_models["data"]["quartil"] = [-1, -1, -1]
    yml_models["data"]["best_model"] = None
    pow = range(1, yml_models["data"]["max_poly"] + 1)
    # combinations = np.array(list(itertools.product(list(itertools.product(pow)), repeat=len(x_head)))) # all combinations too long
    combinations = [np.full((len(x_head), 1), po) for po in pow]
    for comb in combinations:
        name =''.join(["{}**{}_".format(x_name[:3], str(po[0])) for x_name, po in zip(x_head, comb)])
        yml_x = list([int(po[0]) for po in comb])
        if (sum(yml_x) != 0):
            yml_theta = [1 for _ in range(sum(yml_x) + 1)]
            yml_models["models"][name] = {
                "power_x":yml_x,
                "f1_score":{},
                "total_it":0,
                "alpha":1e-1,
                "historic":{},
                "theta": {}
            }
            for lambda_ in tqdm(np.arange(0.2, 1.2, 0.2), leave=False):
                yml_models["models"][name]["f1_score"][str(lambda_)] = 0
                # yml_models["models"][name]["historic"][str(lambda_)] = [[] for i in range(4)]
                yml_models["models"][name]["theta"][str(lambda_)] = [[1 for _ in range(sum(yml_x) + 1)] for i in range(4)]
    with open(yml_models["data"]["name"], 'w') as outfile:
        yaml.dump(yml_models, outfile, default_flow_style=None)

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
    for models_name in tqdm(yml_models["models"], leave=False):
        model = yml_models["models"][models_name]
        for lambda_ in tqdm(np.arange(0.2, 1.2, 0.2), leave=False):
            X_poly = add_polynomial_features(X, model["power_x"])
            model["f1_score"][str(lambda_)] = 0
            for k_folds in tqdm(cross_validation(X_poly, Y, int(yml_models["data"]["K"])), leave=False):
                f1_score = one_vs_all(k_folds, alpha, rate, lambda_, model)
                model["f1_score"][str(lambda_)] = model["f1_score"][str(lambda_)] + f1_score
            model["f1_score"][str(lambda_)] = model["f1_score"][str(lambda_)] / int(yml_models["data"]["K"])
    with open(yml_models["data"]["name"], 'w') as outfile:
        yaml.dump(yml_models, outfile, default_flow_style=None)

def best_models(yml_models):
    rmse_list = np.array([[str(key), np.mean(list(value["f1_score"].values()))] for key, value in yml_models["models"].items()])
    yml_models["data"]["best_model"] = str(rmse_list[rmse_list[:, 1].astype('float64').argmin()][0])
    yml_models["data"]["quartil"] = [float(np.quantile(rmse_list[:, 1].astype('float64'), quart)) for quart in np.arange(0.25, 1, 0.25)]
    with open(yml_models["data"]["name"], 'w') as outfile:
        yaml.dump(yml_models, outfile, default_flow_style=None)


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "a:r:", ["reset", "train", "display"])
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
        if opt == '--reset':
            init_models(yml_models, [data_x, data_y])
        elif opt == '--train':
            train(yml_models, [data_x, data_y], alpha=alpha, rate=rate)
            best_models(yml_models)

if __name__ == "__main__":
    main(sys.argv[1:])