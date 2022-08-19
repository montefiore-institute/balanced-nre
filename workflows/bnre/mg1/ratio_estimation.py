import glob
import hypothesis as h
import numpy as np
import torch
import os

from tqdm import tqdm

from hypothesis.benchmark.mg1 import Prior
from hypothesis.nn import build_ratio_estimator
from hypothesis.nn.ratio_estimation import BaseRatioEstimator
from hypothesis.nn.ratio_estimation import RatioEstimatorEnsemble
from hypothesis.stat import highest_density_level
from hypothesis.util.data import NamedDataset
from hypothesis.util.data import NumpyDataset
from torch.utils.data import TensorDataset



prior = Prior()


extent = [ # I know, this isn't very nice :(
    prior.low[0].item(), prior.high[0].item(),
    prior.low[1].item(), prior.high[1].item(),
    prior.low[2].item(), prior.high[2].item()]


@torch.no_grad()
def load_estimator(query):
    paths = glob.glob(query)
    if len(paths) == 1:
        estimator = RatioEstimator()
        estimator.load_state_dict(torch.load(query))
    else:
        estimators = []
        for path in paths:
            estimators.append(load_estimator(path))
        estimator = RatioEstimatorEnsemble(estimators, reduce='discriminator_mean')
    estimator = estimator.to(h.accelerator)
    estimator.eval()

    return estimator


@torch.no_grad()
def compute_log_posterior(r, observable, resolution=100):
    # Prepare grid
    epsilon = 0.00001
    p1 = torch.linspace(extent[0], extent[1] - epsilon, resolution)  # Account for half-open interval of uniform prior
    p2 = torch.linspace(extent[2], extent[3] - epsilon, resolution)  # Account for half-open interval of uniform prior
    p3 = torch.linspace(extent[4], extent[5] - epsilon, resolution)
    p1 = p1.to(h.accelerator)
    p2 = p2.to(h.accelerator)
    p3 = p3.to(h.accelerator)
    g1, g2, g3 = torch.meshgrid(p1.view(-1), p2.view(-1), p3.view(-1))
    # Vectorize
    inputs = torch.cat([g1.reshape(-1, 1), g2.reshape(-1, 1), g3.reshape(-1, 1)], dim=1)
    log_prior_probabilities = prior.log_prob(inputs).view(-1, 1)
    observables = observable.repeat(resolution ** 3, 1).float()
    observables = observables.to(h.accelerator)
    log_ratios = r.log_ratio(inputs=inputs, outputs=observables)
    log_posterior = (log_prior_probabilities + log_ratios).view(resolution, resolution, resolution).cpu()

    return log_posterior, p1.cpu(), p2.cpu(), p3.cpu()


@torch.no_grad()
def compute_log_pdf(r, inputs, outputs, flow_sbi=False):
    inputs = inputs.to(h.accelerator)
    outputs = outputs.to(h.accelerator)
    log_ratios = r.log_ratio(inputs=inputs, outputs=outputs)
    log_prior = prior.log_prob(inputs)

    return (log_prior + log_ratios).squeeze()


@torch.no_grad()
def estimate_coverage(r, inputs, outputs, outputdir, alphas=[0.05]):
    n = len(inputs)
    covered = [0 for _ in alphas]
    sizes = [[] for _ in range(len(alphas))]
    bias = [0., 0., 0.]
    bias_square = [0., 0., 0.]
    variance = [0. ,0., 0.]
    resolution=90

    length_1 = (extent[1] - extent[0])/resolution
    length_2 = (extent[3] - extent[2])/resolution
    length_3 = (extent[5] - extent[4])/resolution

    for index in tqdm(range(n), "Coverages evaluated"):
        # Prepare setup
        nominal = inputs[index].squeeze().unsqueeze(0)
        observable = outputs[index].squeeze().unsqueeze(0)
        nominal = nominal.to(h.accelerator)
        observable = observable.to(h.accelerator)
        pdf, p1, p2, p3 = compute_log_posterior(r, observable, resolution=resolution)
        pdf = pdf.exp()
        nominal_pdf = compute_log_pdf(r, nominal, observable).exp()
        for i, alpha in enumerate(alphas):
            level, mask = highest_density_level(pdf, alpha, region=True)
            sizes[i].append(np.sum(mask) / np.prod(np.shape(mask)))
            if nominal_pdf >= level:
                covered[i] += 1

        pdf = pdf/(length_1*length_2*length_3*pdf.sum())

        margin_1 = pdf.sum(dim=2).sum(dim=1)*length_3*length_2
        margin_2 = pdf.sum(dim=2).sum(dim=0)*length_3*length_1
        margin_3 = pdf.sum(dim=1).sum(dim=0)*length_2*length_1
        mean_1 = (margin_1*length_1*p1).sum()
        mean_2 = (margin_2*length_2*p2).sum()
        mean_3 = (margin_3*length_3*p3).sum()
        bias[0] += torch.abs((mean_1 - nominal[0, 0]).cpu().float())
        bias[1] += torch.abs((mean_2 - nominal[0, 1]).cpu().float())
        bias[2] += torch.abs((mean_3 - nominal[0, 2]).cpu().float())
        bias_square[0] += (mean_1 - nominal[0, 0]).cpu().float()**2
        bias_square[1] += (mean_2 - nominal[0, 1]).cpu().float()**2
        bias_square[2] += (mean_3 - nominal[0, 2]).cpu().float()**2
        variance[0] += (margin_1*length_1*(p1 - mean_1)**2).sum().cpu().float()
        variance[1] += (margin_2*length_2*(p2 - mean_2)**2).sum().cpu().float()
        variance[2] += (margin_3*length_3*(p3 - mean_3)**2).sum().cpu().float()

    return [x / n for x in covered], sizes, [x / n for x in bias], [x / n for x in variance], [x / n for x in bias_square]


class RatioEstimator(BaseRatioEstimator):

    def __init__(self):
        random_variables = {"inputs": (3,), "outputs": (5,)}
        Class = build_ratio_estimator("mlp", random_variables)
        activation = torch.nn.SELU
        trunk = [256] * 6
        self.means = torch.tensor([4.449119, 17.552252, 37.19249, 75.893616, 249.51308]).to(h.accelerator)
        self.stds = torch.tensor([130.47417, 1330.6927, 3427.9126, 8741.079, 31140.67]).to(h.accelerator)
        r = Class(activation=activation, trunk=trunk)
        super(RatioEstimator, self).__init__(r=r)
        self._r = r

    def log_ratio(self, inputs, outputs, **kwargs):
        outputs = (outputs - self.means)/self.stds
        return self._r.log_ratio(inputs=inputs, outputs=outputs, **kwargs)


class DatasetJointTrain(NamedDataset):

    def __init__(self, n=None):
        inputs = np.load("mg1/data/train/inputs.npy")
        outputs = np.load("mg1/data/train/outputs.npy")
        if n is not None:
            indices = np.random.choice(np.arange(len(inputs)), n, replace=False)
            inputs = inputs[indices, :]
            outputs = outputs[indices, :]
        inputs = TensorDataset(torch.from_numpy(inputs))
        outputs = TensorDataset(torch.from_numpy(outputs))
        super(DatasetJointTrain, self).__init__(
            inputs=inputs,
            outputs=outputs)


class DatasetJointTrain1024(DatasetJointTrain):

    def __init__(self):
        super(DatasetJointTrain1024, self).__init__(n=1024)


class DatasetJointTrain2048(DatasetJointTrain):

    def __init__(self):
        super(DatasetJointTrain2048, self).__init__(n=2048)


class DatasetJointTrain4096(DatasetJointTrain):

    def __init__(self):
        super(DatasetJointTrain4096, self).__init__(n=4096)


class DatasetJointTrain8192(DatasetJointTrain):

    def __init__(self):
        super(DatasetJointTrain8192, self).__init__(n=8192)


class DatasetJointTrain16384(DatasetJointTrain):

    def __init__(self):
        super(DatasetJointTrain16384, self).__init__(n=16384)


class DatasetJointTrain32768(DatasetJointTrain):

    def __init__(self):
        super(DatasetJointTrain32768, self).__init__(n=32768)


class DatasetJointTrain65536(DatasetJointTrain):

    def __init__(self):
        super(DatasetJointTrain65536, self).__init__(n=65536)


class DatasetJointTrain131072(DatasetJointTrain):

    def __init__(self):
        super(DatasetJointTrain131072, self).__init__(n=131072)


class DatasetJointValidate(NamedDataset):

    def __init__(self):
        inputs = NumpyDataset("mg1/data/validate/inputs.npy")
        outputs = NumpyDataset("mg1/data/validate/outputs.npy")
        super(DatasetJointValidate, self).__init__(
            inputs=inputs,
            outputs=outputs)


class DatasetJointTest(NamedDataset):

    def __init__(self):
        inputs = NumpyDataset("mg1/data/test/inputs.npy")
        outputs = NumpyDataset("mg1/data/test/outputs.npy")
        super(DatasetJointTest, self).__init__(
            inputs=inputs,
            outputs=outputs)

class DatasetJointValidateSmall(NamedDataset):

    def __init__(self):
        if not os.path.exists("mg1/data/validate/inputs_small.npy"):
            inputs = np.load("mg1/data/validate/inputs.npy")
            if inputs.shape[0] >= 10000:
                inputs = inputs[:10000]

            np.save("mg1/data/validate/inputs_small.npy", inputs)

        if not os.path.exists("mg1/data/validate/outputs_small.npy"):
            outputs = np.load("mg1/data/validate/outputs.npy")
            if outputs.shape[0] >= 10000:
                outputs = outputs[:10000]

            np.save("mg1/data/validate/outputs_small.npy", outputs)
        
        inputs = NumpyDataset("mg1/data/validate/inputs_small.npy")
        outputs = NumpyDataset("mg1/data/validate/outputs_small.npy")

        super(DatasetJointValidateSmall, self).__init__(
            inputs=inputs,
            outputs=outputs)

class DatasetJointTestSmall(NamedDataset):

    def __init__(self):
        if not os.path.exists("mg1/data/test/inputs_small.npy"):
            inputs = np.load("mg1/data/test/inputs.npy")
            if inputs.shape[0] >= 10000:
                inputs = inputs[:10000]

            np.save("mg1/data/test/inputs_small.npy", inputs)

        if not os.path.exists("mg1/data/test/outputs_small.npy"):
            outputs = np.load("mg1/data/test/outputs.npy")
            if outputs.shape[0] >= 10000:
                outputs = outputs[:10000]

            np.save("mg1/data/test/outputs_small.npy", outputs)

        inputs = NumpyDataset("mg1/data/test/inputs_small.npy")
        outputs = NumpyDataset("mg1/data/test/outputs_small.npy")

        super(DatasetJointTestSmall, self).__init__(
            inputs=inputs,
            outputs=outputs)