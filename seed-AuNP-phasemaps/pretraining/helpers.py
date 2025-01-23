import numpy as np
import torch
from torch.distributions import Normal
from torch.utils.data import Dataset
RNG = np.random.default_rng()
import glob, pdb
import matplotlib.pyplot as plt
from activephasemap.models.np import context_target_split
from activephasemap.utils.visuals import _inset_spectra, MinMaxScaler, scaled_tickformat
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        wl_ = torch.tensor(wl).unsqueeze(1).to(torch.double)
        I_ = torch.tensor(I).unsqueeze(1).to(torch.double)

        return wl_, I_

def plot_samples(ax, model, x_target, z_dim, num_samples=100):
    z_sample = torch.randn((num_samples, z_dim))
    with torch.no_grad():
        for zi in z_sample:
            mu, _ = model.xz_to_y(x_target, zi.to(device))
            ax.plot(x_target.cpu().numpy()[0], mu.detach().cpu().numpy()[0], c='b', alpha=0.5)

    return 

def plot_posterior_samples(x_target, data_loader, model):
    fig, axs = plt.subplots(2,5, figsize=(4*5, 4*2))
    for ax in axs.flatten():
        x, y = next(iter(data_loader))
        x_context, y_context, _, _ = context_target_split(x[0:1], y[0:1], 50, 50)
        with torch.no_grad():
            for _ in range(200):
                # Neural process returns distribution over y_target
                p_y_pred = model(x_context, y_context, x_target)
                # Extract mean of distribution
                mu = p_y_pred.loc.detach()
                ax.plot(x_target.cpu().numpy()[0], mu.cpu().numpy()[0], alpha=0.05, c='b')

            ax.scatter(x_context[0].cpu().numpy(), y_context[0].cpu().numpy(), c='tab:red')
            ax.plot(x[0:1].cpu().squeeze().numpy(), y[0:1].cpu().squeeze().numpy(), c='tab:red')

    return fig, axs


def plot_zgrid_curves(z_range, x_target, model):
    z = torch.linspace(z_range[0],z_range[1],10)
    fig, ax = plt.subplots(figsize=(10, 10))
    scaler = MinMaxScaler(z_range[0],z_range[1])
    for i in range(10):
        for j in range(10):
            zij = torch.Tensor([z[i], z[j]])
            with torch.no_grad():
                yi, _ = model.xz_to_y(x_target, zij.to(device))
            norm_zij = np.array([scaler.transform(z[i].cpu().numpy()), 
                                scaler.transform(z[j].cpu().numpy())]
                                )
            _inset_spectra(norm_zij,x_target.cpu().squeeze().numpy(),
            yi.cpu().squeeze().numpy(), [], ax, show_sigma=False)

    ax.xaxis.set_major_formatter(lambda x, pos : scaled_tickformat(scaler, x, pos))
    ax.yaxis.set_major_formatter(lambda y, pos : scaled_tickformat(scaler, y, pos))
    ax.set_xlabel('Z1', fontsize=20)
    ax.set_ylabel('Z2', fontsize=20)

    return fig, ax
