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
    yml_models["data"]["quartil"] = [-1, -1, -1]
    yml_models["data"]["best_model"] = None
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
    with open(yml_models["data"]["name"], 'w') as outfile:
        yaml.dump(yml_models, outfile, default_flow_style=None)

def train_models(yml_models, data, alpha, rate):
    X = np.array(data[yml_models["data"]["x_head"]])
    Y = np.array(data[yml_models["data"]["y_head"]])
    x_train, x_test, y_train, y_test = data_spliter(X, Y, 0.9)
    x_test = normalize(x_test)
    y_test = y_test
    x = normalize(x_train)
    y = y_train

    for models_name in tqdm(yml_models["models"], leave=False):
        models = yml_models["models"][models_name]
        x_ = add_polynomial_features(x, models["power_x"])
        x_test_ = add_polynomial_features(x_test, models["power_x"])
        theta = np.array(models["theta"]).reshape(-1, 1)
        quartil = yml_models["data"]["quartil"]
        if (quartil[1] != -1 and models["rmse"] > quartil[1]):
            continue
        my_lr = myLR(theta, alpha, rate)
        bg_rmse = my_lr.rmse_(y_test, my_lr.predict_(x_test_))
        while True:
            my_lr = my_lr.fit_(x_, y)
            if my_lr is None:
                print("{} alpha to high".format(models_name))
                alpha = alpha * 1e-1
                continue
            break
        end_rmse = my_lr.rmse_(y_test, my_lr.predict_(x_test_))
        models["rmse"] = end_rmse
        models["theta"] = [int(tta) for tta in my_lr.theta]
        models["total_it"] = int(models["total_it"]) + int(models["max_iter"])
        models["historic"].append([{
            "begin_rmse":float(bg_rmse),
            "alpha":float(alpha),
            "max_iter":int(rate),
            "end_rmse":float(end_rmse),
        }])
    with open(yml_models["data"]["name"], 'w') as outfile:
        yaml.dump(yml_models, outfile, default_flow_style=None)

def best_models(yml_models, delete=False):
    rmse_list = np.array([[str(key), float(value["rmse"])] for key, value in yml_models["models"].items()])
    yml_models["data"]["best_model"] = str(rmse_list[rmse_list[:,1].astype('float64').argmin()][0])
    yml_models["data"]["quartil"] = [float(np.quantile(rmse_list[:,1].astype('float64'), quart)) for quart in np.arange(0.25, 1, 0.25)]
    # if (delete):
    #     for key in list(yml_models["models"]):
    #         if (yml_models["models"][key]["rmse"] > yml_models["data"]["quartil"][1]):
    #             del yml_models["models"][key]
    with open(yml_models["data"]["name"], 'w') as outfile:
        yaml.dump(yml_models, outfile, default_flow_style=None)

def train(yml_models, data, max_range=3, init=False, alpha=0.1, rate=1000):
    if (init):
        for _ in tqdm(range(0, max_range)):
            for _ in tqdm(range(0, max_range), leave=False):
                train_models(yml_models, data, alpha, rate)
            best_models(yml_models, delete=True)
    else:
        for _ in tqdm(range(0, max_range)):
            train_models(yml_models, data, alpha, rate)
        best_models(yml_models, delete=False)

def display(yml_models, data):
    arr = np.array([[int(value["power_x"][0]), int(value["power_x"][1]),int(value["power_x"][2]),int(value["rmse"])] for key, value in yml_models["models"].items()])
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    img = ax.scatter(arr[:,0],arr[:,1],arr[:,2], c=arr[:,3], cmap=plt.hot()) 
    for i in range(len(arr)): #plot each point + it's index as text above
        ax.text(arr[i,0],arr[i,1],arr[i,2],  '%s' % (str(i)), size=5, zorder=1) 
    fig.colorbar(img)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
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
            train(yml_models, data, init=True)
        elif opt == '--train':
            train(yml_models, data, alpha=alpha, rate=rate)
        elif opt == '--display':
            display(yml_models, data)

if __name__ == "__main__":
    main(sys.argv[1:])