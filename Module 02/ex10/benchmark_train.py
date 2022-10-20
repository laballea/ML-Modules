from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.data_spliter import data_spliter
from utils.polynomial_model import add_polynomial_features
from utils.mylinearregression import MyLinearRegression as myLR
from utils.normalize import normalize
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import yaml
import itertools
import sys, getopt

def plot2d_prediction(act_prices, predicted_prices, title, x_labels, y_labels):
    figure, axis = plt.subplots(len(x_labels), 1, constrained_layout=True)
    figure.suptitle(title, fontsize=16)
    labels = ["real price", "predicted price"]
    for idx, label in enumerate(x_labels):
        axis[idx].scatter(act_prices[0][:,idx], act_prices[1], marker='o', color="b", label=labels[0])
        axis[idx].scatter(predicted_prices[0][:,idx], predicted_prices[1], marker='.', color="orange", label=labels[1])
        axis[idx].set_xlabel(label)
        axis[idx].set_ylabel(y_labels)
        axis[idx].legend()

def init_models(yml_models, data):
    x_head = yml_models["data"]["x_head"]
    pow = range(0, yml_models["data"]["max_poly"] + 1)
    combinations = np.array(list(itertools.product(list(itertools.product(pow)), repeat=len(x_head))))
    for comb in combinations:
        name =''.join(["{}**{}_".format(x_name[:3], str(po[0])) for x_name, po in zip(x_head, comb)])
        yml_x = list([int(po[0]) for po in comb])
        if (sum(yml_x) != 0):
            yml_theta = [1 for _ in range(sum(yml_x) + 1)]
            yml_models["models"][name] = {
                "power_x":yml_x,
                "theta":yml_theta,
                "rmse":None,
                "historic":[],
                "total_it":0,
                "alpha":1e-1,
                "max_iter":1000

            }
    with open('models.yml', 'w') as outfile:
        yaml.dump(yml_models, outfile, default_flow_style=None)

def train_models(yml_models, data):
    X = np.array(data[["weight", "prod_distance", "time_delivery"]])
    Y = np.array(data[["target"]])
    x_train, x_test, y_train, y_test = data_spliter(X, Y, 0.9)
    x_test = normalize(x_test)
    y_test = y_test
    x = normalize(x_train)
    y = y_train

    for models_name in tqdm(yml_models["models"], leave=False):
        models = yml_models["models"][models_name]
        st_quartil = float(yml_models["data"]["1st_quartil"])
        if (st_quartil > 0 and float(models["rmse"]) > st_quartil):
            continue
        x_ = add_polynomial_features(x, models["power_x"])
        x_test_ = add_polynomial_features(x_test, models["power_x"])
        theta = np.array(models["theta"]).reshape(-1, 1)
        my_lr = myLR(theta, alpha=models["alpha"], max_iter=models["max_iter"])
        bg_rmse = my_lr.rmse_(y_test, my_lr.predict_(x_test_))
        while True:
            my_lr = my_lr.fit_(x_, y)
            if my_lr is None:
                print("{} alpha to high".format(models_name))
                models["alpha"] = float(models["alpha"])**2
                continue
            break
        end_rmse = my_lr.rmse_(y_test, my_lr.predict_(x_test_))
        if (abs(bg_rmse - end_rmse) * 100 / bg_rmse < 10):
            models["alpha"] = float(models["alpha"]) * 1e-1
            models["max_iter"] = int(models["max_iter"]) * 2
        models["rmse"] = end_rmse
        models["theta"] = [int(tta) for tta in my_lr.theta]
        models["total_it"] = int(models["total_it"]) + int(models["max_iter"])
        models["historic"].append([{
            "begin_rmse":float(bg_rmse),
            "alpha":float(models["alpha"]),
            "max_iter":int(models["max_iter"]),
            "end_rmse":float(end_rmse),
        }])
    with open('models.yml', 'w') as outfile:
        yaml.dump(yml_models, outfile, default_flow_style=None)

def best_models(yml_models):
    rmse_list = np.array([[str(key), float(value["rmse"])] for key, value in yml_models["models"].items()])
    yml_models["data"]["best_model"] = str(rmse_list[rmse_list[:,1].astype('float64').argmin()][0])
    yml_models["data"]["1st_quartil"] = float(np.quantile(rmse_list[:,1].astype('float64'), .25))
    with open('models.yml', 'w') as outfile:
        yaml.dump(yml_models, outfile, default_flow_style=None)


def main(argv):
    try:
        opts, args = getopt.getopt(argv,"hreset:",["init"])
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-reset':
            with open("models.yml", "r") as stream:
                try:
                    yml_models = yaml.safe_load(stream)
                    data = pd.read_csv(yml_models["data"]["path"])
                except yaml.YAMLError as exc:
                    print(exc)
            init_models(yml_models, data)
    for _ in tqdm(range(0, 5)):
        with open("models.yml", "r") as stream:
            try:
                yml_models = yaml.safe_load(stream)
                data = pd.read_csv(yml_models["data"]["path"])
            except yaml.YAMLError as exc:
                print(exc)  
        train_models(yml_models, data)
        best_models(yml_models)

if __name__ == "__main__":
    print(sys.argv[1:])
    main(sys.argv[1:])

#     # title = "sklearn theta**{} / mse: {} / mse: {}".format(po, mse, mean_squared_error(y_test, y_hat, squared=False))
#     # plot2d_prediction([x_test[:], y_test], [x_test[:], y_hat], title, x_labels=["weight","prod_distance","time_delivery"], y_labels="price")
#     plt.show()
