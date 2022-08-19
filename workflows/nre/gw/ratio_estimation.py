import glob
import hypothesis as h
import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm

from hypothesis.benchmark.gravitational_waves import Prior
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
    prior.low[1].item(), prior.high[1].item()]


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
    p1 = torch.linspace(extent[0], extent[1] - epsilon, resolution)  # Account for half-open interval of uniform prior
    p2 = torch.linspace(extent[2], extent[3] - epsilon, resolution)  # Account for half-open interval of uniform prior
    p1 = p1.to(h.accelerator)
    p2 = p2.to(h.accelerator)
    g1, g2 = torch.meshgrid(p1.view(-1), p2.view(-1))
    # Vectorize
    inputs = torch.cat([g1.reshape(-1, 1), g2.reshape(-1, 1)], dim=1)
    log_prior_probabilities = prior.log_prob(inputs).view(-1, 1)
    observables = observable.repeat(resolution ** 2, 1, 1).float()
    observables = observables.to(h.accelerator)
    log_ratios = r.log_ratio(inputs=inputs, outputs=observables)
    log_posterior = (log_prior_probabilities + log_ratios).view(resolution, resolution).cpu()

    return log_posterior, p1.cpu(), p2.cpu()


@torch.no_grad()
def compute_log_pdf(r, inputs, outputs, flow_sbi=False, batch_size=128):
    inputs = inputs.to(h.accelerator)
    outputs = outputs.to(h.accelerator)
    log_ratios = torch.empty(inputs.shape[0]).to(h.accelerator)

    for b in range(0, inputs.shape[0], batch_size):
        cur_inputs = inputs[b:b+batch_size]
        cur_outputs = outputs[b:b+batch_size]
        log_ratios[b:b+batch_size] = r.log_ratio(inputs=cur_inputs, outputs=cur_outputs).squeeze(1)

    log_prior = prior.log_prob(inputs)

    return (log_prior + log_ratios).squeeze()


@torch.no_grad()
def estimate_coverage(r, inputs, outputs, alphas=[0.05]):
    n = len(inputs)
    covered = [0 for _ in alphas]
    sizes = [[] for _ in range(len(alphas))]
    bias = [0., 0.]
    bias_square = [0., 0.]
    variance = [0. ,0.]

    resolution = 90

    length_1 = (extent[1] - extent[0])/resolution
    length_2 = (extent[3] - extent[2])/resolution

    for index in tqdm(range(n), "Coverages evaluated"):
        # Prepare setup
        nominal = inputs[index].squeeze().unsqueeze(0)
        observable = outputs[index].squeeze().unsqueeze(0)
        nominal = nominal.to(h.accelerator)
        observable = observable.to(h.accelerator)
        pdf, p1, p2 = compute_log_posterior(r, observable, resolution=resolution)
        pdf = pdf.exp()
        nominal_pdf = compute_log_pdf(r, nominal, observable).exp()
        for i, alpha in enumerate(alphas):
            level, mask = highest_density_level(pdf, alpha, region=True)
            sizes[i].append(np.sum(mask) / np.prod(np.shape(mask)))
            if nominal_pdf >= level:
                covered[i] += 1

        #print("length 1 = {}".format(length_1))
        #print("length 2 = {}".format(length_2))
        pdf = pdf/(length_1*length_2*pdf.sum())
        #print("pdf integral = {}".format(length_1*length_2*pdf.sum()))
        margin_1 = pdf.sum(dim=1)*length_2
        margin_2 = pdf.sum(dim=0)*length_1
        #print("margin 1 integral = {}".format(length_1*margin_1.sum()))
        #print("margin 2 integral = {}".format(length_2*margin_2.sum()))
        #print("pdf integral = {}".format(length_1*length_2*pdf.sum()))
        #print("margin 1 = {}".format(margin_1))
        #print("margin 2 = {}".format(margin_2))
        #print("p1 = {}".format(p1))
        #print("p2 = {}".format(p2))
        mean_1 = (margin_1*length_1*p1).sum()
        mean_2 = (margin_2*length_2*p2).sum()
        bias[0] += torch.abs((mean_1 - nominal[0, 0]).cpu().float())
        bias[1] += torch.abs((mean_2 - nominal[0, 1]).cpu().float())
        bias_square[0] += (mean_1 - nominal[0, 0]).cpu().float()**2
        bias_square[1] += (mean_2 - nominal[0, 1]).cpu().float()**2
        variance[0] += (margin_1*length_1*(p1 - mean_1)**2).sum().cpu().float()
        variance[1] += (margin_2*length_2*(p2 - mean_2)**2).sum().cpu().float()


    return [x / n for x in covered], sizes, [x / n for x in bias], [x / n for x in variance], [x / n for x in bias_square]


class RatioEstimator(BaseRatioEstimator):

    def __init__(self):
        super(RatioEstimator, self).__init__(
            denominator = "inputs|outputs",
            random_variables={"inputs": (2,), "outputs": (2, 8192)})

        nb_channels = 16
        fc_layers = [nb_channels + 2, 128, 128, 128, 1]

        cnn = [nn.Conv1d(in_channels=2, out_channels=nb_channels, kernel_size=1)]

        for i in range(13):
            cnn.append(nn.Conv1d(in_channels=nb_channels, out_channels=nb_channels, kernel_size=2, dilation=2**i))
            cnn.append(nn.SELU())


        self.features = nn.Sequential(*cnn)
        fc = []
        for i in range(len(fc_layers) - 1):
            fc.append(nn.Linear(fc_layers[i], fc_layers[i+1]))
            fc.append(nn.SELU())

        fc.pop()
        self.fc = nn.Sequential(*fc)

        self.features.type(torch.float32)
        self.fc.type(torch.float32)


    def log_ratio(self, inputs, outputs, **kwargs):
        features = self.features(outputs).view(outputs.shape[0], -1)
        concat = torch.cat((features, inputs), 1)
        return self.fc(concat)


class DatasetJointTrain(NamedDataset):

    def __init__(self, n=None):
        inputs = np.load("gw/data/train/inputs.npy")
        outputs = np.load("gw/data/train/outputs.npy")
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
        inputs = NumpyDataset("gw/data/validate/inputs.npy")
        outputs = NumpyDataset("gw/data/validate/outputs.npy")
        super(DatasetJointValidate, self).__init__(
            inputs=inputs,
            outputs=outputs)


class DatasetJointTest(NamedDataset):

    def __init__(self):
        inputs = NumpyDataset("gw/data/test/inputs.npy")
        outputs = NumpyDataset("gw/data/test/outputs.npy")
        super(DatasetJointTest, self).__init__(
            inputs=inputs,
            outputs=outputs)
