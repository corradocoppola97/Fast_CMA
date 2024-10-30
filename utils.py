import torch
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from torchsummaryX import summary
import csv
import ast
import torchvision
from typing import Union
from cmalight import CMA_L
from FCMA import F_CMA
import itertools

#Used to compute the loss function over the entire data set
def closure(data_loader: torch.utils.data.DataLoader,
            model: torchvision.models,
            criterion: torch.nn,
            device: Union[torch.device,str],
            probabilistic = False,
            probabilistic_k = 50):
    
    loss = 0
    model.eval()

    with torch.no_grad():
        if probabilistic == False:
            for x,y in data_loader:
                x,y = x.to(device), y.to(device)
                y_pred = model(x)
                batch_loss = criterion(y_pred, y)
                loss += batch_loss.item() * (len(x)/len(data_loader.dataset))
        else:
            sampled_loader = itertools.islice(data_loader, probabilistic_k)
            for x, y in sampled_loader:
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                batch_loss = criterion(y_pred, y)
                loss += batch_loss.item() * (len(x) / (probabilistic_k * data_loader.batch_size))

    return loss



#Used to compute the accuracy over the entire data set
def accuracy(data_loader: torch.utils.data.DataLoader,
            model: torchvision.models,
            device: Union[torch.device,str]):
    correct_predictions = 0
    total_samples = 0

    model.eval()

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    accuracy = correct_predictions / total_samples
    return accuracy


def set_optimizer(opt:str, model: torchvision.models, momentum_camlV1=0.0, nesterov_cmalV1=False):
    if opt == 'adadelta':
        optimizer = torch.optim.Adadelta(model.parameters())
    elif opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters())
    elif opt == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters())
    elif opt == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters())
    elif opt == 'adamax':
        optimizer = torch.optim.Adamax(model.parameters())
    elif opt == 'asgd':
        optimizer = torch.optim.ASGD(model.parameters())
    elif opt == 'nadam':
        optimizer = torch.optim.NAdam(model.parameters())
    elif opt == 'radam':
        optimizer = torch.optim.RAdam(model.parameters())
    elif opt == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters())
    elif opt == 'rprop':
        optimizer = torch.optim.Rprop(model.parameters())
    elif opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    elif opt == 'cmal':
        optimizer = CMA_L(model.parameters(), zeta=0.05, theta=0.75, delta=0.9, gamma=1e-2, verbose=False, verbose_EDFL=False)
    elif opt == 'fcma':
        optimizer = F_CMA(model.parameters(), zeta=0.01, theta=0.5, delta=0.9, gamma=0.01, verbose=False, verbose_EDFL=False,
                                eta=0.75, tol_zeta=1e-10, momentum=momentum_camlV1, nesterov=nesterov_cmalV1)
    else:
        raise SystemError('Set your optimizer! :)')
    return optimizer

def hardware_check():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Actual device: ", device)
    if 'cuda' in device:
        print("Device info: {}".format(str(torch.cuda.get_device_properties(device)).split("(")[1])[:-1])

    return device
    
def print_model(model, device, save_model_root, input_shape):
    info = summary(model, torch.ones((1, 3, 32, 32)).to(device))
    info.to_csv(save_model_root + 'model_summary.csv')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    

def save_csv_history(path):
    objects = []
    with (open(path + 'history.pkl', "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
    df = pd.DataFrame(objects)
    df.to_csv(path + 'history.csv', header=False, index=False, sep=" ")
    

def plot_graph(data, label, title, path):
    epochs = range(0, len(data))
    plt.plot(epochs, data, 'orange', label=label)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid('on', color='#cfcfcf')
    plt.tight_layout()
    plt.savefig(path + title + '.pdf')
    plt.close()


def plot_history(history, path):
    plot_graph(history['train_loss'], 'Train Loss', 'Train_loss', path)
    plot_graph(history['val_acc'], 'Val Acc.', 'Val_acc', path)


def extract_history(history_file):
    with open(history_file) as file_obj:
        reader_obj = csv.reader(file_obj)
        for row in reader_obj:
            l = row[0]
            break
        l = l[:-9]
        train_loss = ast.literal_eval(l)
        val_accuracy = row[498:498+250]
        val_accuracy[-1] = val_accuracy[-1][:4]
        val_accuracy[0] = val_accuracy[0][-5:]
        val_accuracy = [float(c.strip('[ ]')) for c in val_accuracy]
    return train_loss, val_accuracy


import torch.distributed as dist

def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    def is_dist_avail_and_initialized():
        if not dist.is_available():
            return False
        if not dist.is_initialized():
            return False
        return True

    def get_world_size():
        if not is_dist_avail_and_initialized():
            return 1
        return dist.get_world_size()
    
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    data_list = [None] * world_size
    dist.all_gather_object(data_list, data)
    return data_list