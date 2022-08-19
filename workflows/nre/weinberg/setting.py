from hypothesis.benchmark.weinberg import Prior
from hypothesis.benchmark.weinberg import Simulator as WeinbergSimulator

class Simulator(WeinbergSimulator):
    def __init__(self, default_beam_energy=40.0, num_samples=20):
        super(Simulator, self).__init__(default_beam_energy=default_beam_energy, num_samples=num_samples)

memory = '4GB'
ngpus = 0
