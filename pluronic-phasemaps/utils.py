import numpy as np 
from autophasemap import BaseDataSet 
from scipy.interpolate import splev, splrep
import gpytorch
import torch
from gpytorch.models import ExactGP
from gpytorch.likelihoods import DirichletClassificationLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
import matplotlib.pyplot as plt


color_blindf = ["#1F77B4", "#AEC7E8", "#FF7F0E", "#FFBB78", 
                "#2CA02C", "#98DF8A", "#D62728", "#FF9896", 
                "#9467BD", "#C5B0D5", "#8C564B",  "#C49C94", 
                "#E377C2", "#F7B6D2", "#7F7F7F", "#C7C7C7", 
                "#BCBD22", "#DBDB8D", "#17BECF", "#9EDAE5"
                ]

class DataSet(BaseDataSet):
    def __init__(self, C, q, Iq, n_domain):
        super().__init__(n_domain=n_domain)
        self.t = np.linspace(0,1, num=self.n_domain)
        self.N = Iq.shape[0]
        self.Iq = Iq
        self.C = C
        self.q = q
        
    def generate(self, apply_spine = False):
        if apply_spine:
            q_ = self.q.copy()
            self.q = np.geomspace(min(self.q), max(self.q), num=self.n_domain)
            self.t = (self.q - min(self.q))/(max(self.q)-min(self.q))
            self.F = []
            for i in range(self.N):
                iq_original = self.Iq[i,:]
                spline = splrep(q_, iq_original)
                iq_downsample = splev(self.q, spline)
                self.F.append(iq_downsample/self.l2norm(self.q, iq_downsample))
        else:
            self.F = [self.Iq[i,:]/self.l2norm(self.q, self.Iq[i,:]) for i in range(self.N)]
        return
    
class DirichletGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_classes):
        super(DirichletGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean(batch_shape=torch.Size((num_classes,)))
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=torch.Size((num_classes,))),
            batch_shape=torch.Size((num_classes,)),
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def _fit_gp_model(train_x, train_y, model, likelihood, training_iter=200):
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, likelihood.transformed_targets).sum()
        loss.backward()
        if i % 50 == 0:
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                i + 1, training_iter, loss.item(),
                model.covar_module.base_kernel.lengthscale.mean().item(),
                model.likelihood.second_noise_covar.noise.mean().item()
            ))
        optimizer.step()

    return model, likelihood


def plot_phasemap_contours(data, result, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    min_prob = kwargs.get("min_prob", 0.3)
    n_clusters = len(result["templates"])

    train_x = torch.Tensor(data.C)
    train_y = torch.Tensor(result["delta_n"]).long().squeeze()

    # initialize likelihood and model
    # we let the DirichletClassificationLikelihood compute the targets for us
    likelihood = DirichletClassificationLikelihood(train_y, learn_additional_noise=True)
    model = DirichletGPModel(train_x, 
                             likelihood.transformed_targets, 
                             likelihood, 
                             num_classes=likelihood.num_classes
                             )
    model, likelihood = _fit_gp_model(train_x, 
                                      train_y, 
                                      model, 
                                      likelihood, 
                                      training_iter=200
                                      )

    n_grid_points = kwargs.get("n_grid_points", 30)
    test_d1 = np.linspace(data.C[:,0].min(), data.C[:,0].max(), n_grid_points)
    test_d2 = np.linspace(data.C[:,1].min(), data.C[:,1].max(), n_grid_points)
    test_x_mat, test_y_mat = np.meshgrid(test_d1, test_d2)
    test_x_mat, test_y_mat = torch.Tensor(test_x_mat), torch.Tensor(test_y_mat)

    test_x = torch.cat((test_x_mat.view(-1,1), test_y_mat.view(-1,1)),dim=1)

    model.eval()
    likelihood.eval()

    with gpytorch.settings.fast_pred_var(), torch.no_grad():
        test_dist = model(test_x)

    pred_samples = test_dist.sample(torch.Size((256,))).exp()
    probabilities = (pred_samples / pred_samples.sum(-2, keepdim=True)).mean(0)

    for k in range(n_clusters):
        q_ = np.geomspace(min(data.q), max(data.q), num=len(result["templates"][k]))
        flags = probabilities[k]>min_prob
        comps_k_mean = np.median(test_x[flags,:], axis=0)
        norm_ci = (comps_k_mean-data.C.min(axis=0))/((data.C.max(axis=0)-data.C.min(axis=0)))
        loc_ax = ax.transLimits.transform(norm_ci)
        ins_ax = ax.inset_axes([loc_ax[0]-0.1,loc_ax[1]-0.1,0.2,0.2])
        ins_ax.loglog(q_, result["templates"][k], color="k", lw=2.0)
        ins_ax.axis("off")

    for k in range(n_clusters):
        ax.contourf(test_x_mat.numpy(),
                    test_y_mat.numpy(),
                    probabilities[k].numpy().reshape((n_grid_points,n_grid_points)),
                    levels=[min_prob, 1.0],
                    colors=color_blindf[k],
                    alpha=0.75
                    )
    plt.show()