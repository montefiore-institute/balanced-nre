import glob
import hypothesis as h
import numpy as np
import torch

from tqdm import tqdm

from hypothesis.benchmark.weinberg import Prior
from hypothesis.nn import build_ratio_estimator
from hypothesis.nn.ratio_estimation import BaseRatioEstimator
from hypothesis.nn.ratio_estimation import RatioEstimatorEnsemble
from hypothesis.stat import highest_density_level
from hypothesis.util.data import NamedDataset
from hypothesis.util.data import NumpyDataset
from torch.utils.data import TensorDataset



prior = Prior()


extent = [prior.low.item(), prior.high.item()]


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
        estimator = RatioEstimatorEnsemble(estimators, reduce='ratio_mean')
    estimator = estimator.to(h.accelerator)
    estimator.eval()

    return estimator


@torch.no_grad()
def compute_log_posterior(r, observable, resolution=100):
    # Prepare grid
    epsilon = 0.00001
    inputs = torch.linspace(extent[0], extent[1] - epsilon, resolution).view(-1, 1)  # Account for half-open interval of uniform prior
    inputs = inputs.to(h.accelerator)
    log_prior_probabilities = prior.log_prob(inputs).view(-1, 1)
    observables = observable.repeat(resolution, 1).float()
    observables = observables.to(h.accelerator)
    log_ratios = r.log_ratio(inputs=inputs, outputs=observables)
    log_posterior = (log_prior_probabilities + log_ratios).view(resolution).cpu()

    return log_posterior, inputs.cpu()



@torch.no_grad()
def compute_log_pdf(r, inputs, outputs, flow_sbi=False):
    inputs = inputs.to(h.accelerator)
    outputs = outputs.to(h.accelerator)
    log_ratios = r.log_ratio(inputs=inputs, outputs=outputs)
    log_prior = prior.log_prob(inputs)

    return (log_prior + log_ratios).squeeze()


@torch.no_grad()
def estimate_coverage(r, inputs, outputs, alphas=[0.05]):
    n = len(inputs)
    covered = [0 for _ in alphas]
    sizes = [[] for _ in range(len(alphas))]
    bias = [0., 0.]
    bias_square = [0., 0.]
    variance = [0., 0.]
    resolution = 90
    length_1 = (extent[1] - extent[0])/resolution

    for index in tqdm(range(n), "Coverages evaluated"):
        # Prepare setup
        nominal = inputs[index].squeeze().unsqueeze(0)
        observable = outputs[index].squeeze().unsqueeze(0)
        nominal = nominal.to(h.accelerator)
        observable = observable.to(h.accelerator)
        pdf, p1 = compute_log_posterior(r, observable, resolution=resolution)
        pdf = pdf.exp()
        p1 = p1[:, 0]
        nominal_pdf = compute_log_pdf(r, nominal, observable).exp()
        for i, alpha in enumerate(alphas):
            level, mask = highest_density_level(pdf, alpha, region=True)
            sizes[i].append(np.sum(mask) / np.prod(np.shape(mask)))
            if nominal_pdf >= level:
                covered[i] += 1

        pdf = pdf/(length_1*pdf.sum())
        margin_1 = pdf
        mean_1 = (margin_1*length_1*p1).sum()
        bias[0] += torch.abs((mean_1 - nominal[0]).cpu().float())
        bias_square[0] += (mean_1 - nominal[0]).cpu().float()**2
        variance[0] += (margin_1*length_1*(p1 - mean_1)**2).sum().cpu().float()

    return [x / n for x in covered], sizes, [x / n for x in bias], [x / n for x in variance], [x / n for x in bias_square]


class RatioEstimator(BaseRatioEstimator):

    def __init__(self):
        random_variables = {"inputs": (1,), "outputs": (20,)}
        Class = build_ratio_estimator("mlp", random_variables)
        activation = torch.nn.SELU
        trunk = [256] * 6
        r = Class(activation=activation, trunk=trunk)
        super(RatioEstimator, self).__init__(r=r)
        self._r = r

    def log_ratio(self, **kwargs):
        return self._r.log_ratio(**kwargs)


class DatasetJointTrain(NamedDataset):

    def __init__(self, n=None):
        inputs = np.load("weinberg/data/train/inputs.npy")
        outputs = np.load("weinberg/data/train/outputs.npy")
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
        inputs = NumpyDataset("weinberg/data/validate/inputs.npy")
        outputs = NumpyDataset("weinberg/data/validate/outputs.npy")
        super(DatasetJointValidate, self).__init__(
            inputs=inputs,
            outputs=outputs)


class DatasetJointTest(NamedDataset):

    def __init__(self):
        inputs = NumpyDataset("weinberg/data/test/inputs.npy")
        outputs = NumpyDataset("weinberg/data/test/outputs.npy")
        super(DatasetJointTest, self).__init__(
            inputs=inputs,
            outputs=outputs)
