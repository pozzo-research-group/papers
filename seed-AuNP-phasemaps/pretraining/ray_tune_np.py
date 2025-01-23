import sys, os, pdb, shutil, json, glob
import numpy as np
import tempfile
import torch
from filelock import FileLock
import ray
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from math import pi
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader, Dataset
from activephasemap.models.np.neural_process import NeuralProcess
from activephasemap.models.np.training import NeuralProcessTrainer
from activephasemap.models.np.utils import context_target_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_DIR = os.getcwd() + "/tune/"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

class UVVisDataset(Dataset):
    def __init__(self, root_dir):
        """
        Arguments:
            root_dir (string): Directory with all the data.
        """
        self.dir = root_dir
        self.files = glob.glob(self.dir+'/*.npz')
        self.xrange = [0,1]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        try:
            npzfile = np.load(self.files[i])
        except Exception as e:
            print('%s Could not load %s'%(type(e).__name__, self.files[i]))
        wl, I = npzfile['wl'], npzfile['I']
        wl = (wl-min(wl))/(max(wl)-min(wl))
        wl_ = torch.tensor(wl.astype(np.float32)).unsqueeze(1)
        I_ = torch.tensor(I.astype(np.float32)).unsqueeze(1)

        return wl_, I_


def load_dataset(data_loc):
    with FileLock(os.path.expanduser("~/.data.lock")):
        dataset = UVVisDataset(root_dir=data_loc)

    return dataset

def train_np(config):
    r_dim = config["r_dim"]
    z_dim = config["z_dim"]
    h_dim = config["h_dim"]
    lr = config["lr"]
    batch_size = int(config["batch_size"])

    neuralprocess = NeuralProcess(1, 1, r_dim, z_dim, h_dim)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            neuralprocess = nn.DataParallel(neuralprocess)
    neuralprocess.to(device)

    PLOT_DIR = SAVE_DIR+'%d_%d_%d_%.2E_%d/'%(r_dim, z_dim, h_dim, lr, batch_size)
    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR)
        print("Created %s directory"%PLOT_DIR)
    else:
        print("Directory %s already exists"%PLOT_DIR)

    with open(PLOT_DIR+'config.json', 'w') as fp:
        json.dump(config, fp)   

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            neuralprocess = nn.DataParallel(neuralprocess)
    neuralprocess.to(device)

    optimizer = torch.optim.Adam(neuralprocess.parameters(), lr=lr)
    np_trainer = NeuralProcessTrainer(device, neuralprocess, optimizer,
                                    num_context_range=(3, 47),
                                    num_extra_target_range=(50, 53), 
                                    print_freq=1000
                                    )

    neuralprocess.training = True

    dataset = load_dataset("/mmfs1/home/kiranvad/cheme-kiranvad/activephasemap-examples/pretrained/UVVis/uvvis_data_npy/")

    data_loader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=True,
                            collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x))
                            )
    x_plot = torch.linspace(dataset.xrange[0], dataset.xrange[1], steps = 100).reshape(1,100,1).to(device)
    np_trainer.train(data_loader, 500, x_plot=x_plot, plot_epoch=50, savedir=PLOT_DIR) 
    torch.save(neuralprocess.state_dict(), PLOT_DIR+'model.pt')
    np.save(PLOT_DIR+'loss.npy', np_trainer.epoch_loss_history) 

    train.report({"loss": (np_trainer.epoch_loss_history[-1])})

def main(num_samples=10, max_num_epochs=10):
    config = {
        "r_dim": tune.choice([16, 32, 64, 128]),
        "z_dim": tune.choice([2, 4, 8, 16]),
        "h_dim": tune.choice([16, 32, 64, 128]),
        "batch_size": tune.choice([2, 4, 8, 16]),
        "lr": tune.loguniform(1e-4, 1e-1)
    }
    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_np),
            resources={"cpu": 40, "gpu": 0.5}
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        param_space=config,
    )
    results = tuner.fit()
    
    best_result = results.get_best_result("loss", "min")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(best_result.metrics["loss"]))
    with open('best_config.json', 'w') as fp:
        json.dump(best_result.config, fp)

main(num_samples=16, max_num_epochs=100)
