import argparse
import awflow as aw
import glob
import hypothesis as h
import importlib
import numpy as np
import os
import papermill as pm
import sys
import torch
import hooks

from awflow.contrib.simulate import generate
from hypothesis.util.data.numpy import merge



# Increase recursion depth (large workflow)
sys.setrecursionlimit(10000)

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--estimators', type=int, default=5, help='Number of estimators to train (default: 5).')
parser.add_argument('--gamma', type=float, default=None, help='Regularizer strength (default: 100.0).')
parser.add_argument('--simulation_budget', type=int, default=None, help='simulation budget to run (default: run all budgets).')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001).')
parser.add_argument('--partition', type=str, default=None, help='Slurm partition to execute the pipeline on (default: none).')
parser.add_argument('--problem', type=str, default='', help='Problem to execute (default: none).')
parser.add_argument('--redo', action='store_true', help='Remove the generating files, pipeline will be re-executed (default: false).')
parser.add_argument('--schedule_gamma', action='store_true', help='Schedule the parameter gamma (default: false).')
parser.add_argument('--small_test_val', action='store_true', help='Use smaller test and validation datasets (default: false).')
parser.add_argument('--debug', action='store_true', help='Use smaller test and validation datasets (default: false).')
arguments, _ = parser.parse_known_args()

# Check if the pipeline needs to be re-executed.
if arguments.redo:
    os.system('rm -rf output')

# Check if a problem has been specified
if not os.path.exists(arguments.problem):
    raise ValueError('Unknown problem:', arguments.problem)
problem = arguments.problem

# Load the problem scope (I was previously using importlib)
if problem == 'gw':
    from gw import setting
    from gw.ratio_estimation import estimate_coverage
    from gw.ratio_estimation import load_estimator
    from gw.ratio_estimation import compute_log_pdf
elif problem == 'lotka_volterra':
    from lotka_volterra import setting
    from lotka_volterra.ratio_estimation import estimate_coverage
    from lotka_volterra.ratio_estimation import load_estimator
    from lotka_volterra.ratio_estimation import compute_log_pdf
elif problem == 'mg1':
    from mg1 import setting
    from mg1.ratio_estimation import estimate_coverage
    from mg1.ratio_estimation import load_estimator
    from mg1.ratio_estimation import compute_log_pdf
elif problem == 'slcp':
    from slcp import setting
    from slcp.ratio_estimation import estimate_coverage
    from slcp.ratio_estimation import load_estimator
    from slcp.ratio_estimation import compute_log_pdf
elif problem == 'spatialsir':
    from spatialsir import setting
    from spatialsir.ratio_estimation import estimate_coverage
    from spatialsir.ratio_estimation import load_estimator
    from spatialsir.ratio_estimation import compute_log_pdf
elif problem == 'weinberg':
    from weinberg import setting
    from weinberg.ratio_estimation import estimate_coverage
    from weinberg.ratio_estimation import load_estimator
    from weinberg.ratio_estimation import compute_log_pdf

# Setup the problem setting
Prior = setting.Prior
Simulator = setting.Simulator
memory = setting.memory
ngpus = setting.ngpus
if ngpus == 0:
    workers = 0
else:
    workers = 4

# Script parameters
batch_sizes = [128]
confidence_levels = np.linspace(0.05, 0.95, 19)
learning_rate = arguments.lr
num_estimators = arguments.estimators
schedule_gamma = arguments.schedule_gamma
small_test_val = arguments.small_test_val
debug = arguments.debug
if arguments.simulation_budget is None:
    if debug:
        simulation_budgets = [2 ** i for i in range(15, 16)]
    else:
        simulation_budgets = [2 ** i for i in range(10, 18)]
    
else:
    simulation_budgets = [arguments.simulation_budget]


# Utilities ####################################################################

@torch.no_grad()
def simulate(outputdir, budget):
    # Check if files have been generated.
    inputs_exists = os.path.exists(outputdir + '/inputs.npy')
    outputs_exists = os.path.exists(outputdir + '/outputs.npy')
    if not inputs_exists or not outputs_exists:
        simulator = Simulator()
        prior = Prior()
        inputs = prior.sample((budget,)).view(budget, -1)
        outputs = simulator(inputs)
        np.save(outputdir + '/inputs.npy', inputs.float().numpy())
        np.save(outputdir + '/outputs.npy', outputs.float().numpy())


def optimize(outputdir, budget, gamma):

    if small_test_val:
        val = "DatasetJointValidateSmall"
        test = "DatasetJointTestSmall"
    else:
        val = "DatasetJointValidate"
        test = "DatasetJointTest"
    
    if debug:
        epochs = 2
    else:
        epochs = 500

    command = r"""python -m hypothesis.bin.ratio_estimation.train --batch-size {batch_size} \
                    --data-test {problem}.ratio_estimation.{test} \
                    --data-train {problem}.ratio_estimation.DatasetJointTrain{budget} \
                    --data-validate {problem}.ratio_estimation.{val} \
                    --epochs {epochs} \
                    --gamma {gamma} \
                    --show \
                    --estimator {problem}.ratio_estimation.RatioEstimator \
                    --lr {lr} \
                    --lrsched-on-plateau \
                    --workers {workers} \
                    --out {out}""".format(
        batch_size=batch_size,
        budget=budget,
        gamma=gamma,
        lr=learning_rate,
        problem=problem,
        out=outputdir,
        workers=workers,
        val=val,
        test=test,
        epochs=epochs)

    if schedule_gamma:
        print("add hook")
        command += " --hooks hooks.load_hooks"

    os.system(command)


def generate_simulations():
    root = problem + '/data'
    dependencies = []

    # Simulation pipeline properties
    n = simulation_budgets[-1]
    props = {
        'cpus': 1,
        'memory': '8GB',
        'timelimit': '1-00:00:00'}
    # Training
    g_train = generate(simulate, n, root + '/train', blocks=128, **props)
    dependencies.extend(merge_blocks(root + '/train', g_train))
    # Testing
    g_test = generate(simulate, n, root + '/test', blocks=128, **props)
    dependencies.extend(merge_blocks(root + '/test', g_test))
    # Validation
    g_validate = generate(simulate, n, root + '/validate', blocks=128, **props)
    dependencies.extend(merge_blocks(root + '/validate', g_validate))
    # Coverage
    g_coverage = generate(simulate, 10000, root + '/coverage', blocks=10, **props)
    dependencies.extend(merge_blocks(root + '/coverage', g_coverage))

    return dependencies


def merge_blocks(directory, dependencies):

    # Merge input files
    @aw.cpus(4)
    @aw.dependency(dependencies)
    @aw.memory('16GB')
    @aw.postcondition(aw.exists(directory + '/inputs.npy'))
    def merge_inputs():
        files = glob.glob(directory + '/blocks/*/inputs.npy')
        files.sort()
        merge(input_files=files,
              output_file=directory + '/inputs.npy',
              tempfile=directory + '/temp_inputs',
              in_memory=False,
              axis=0)

    # Merge output files
    @aw.cpus(4)
    @aw.dependency(dependencies)
    @aw.memory('16GB')
    @aw.postcondition(aw.exists(directory + '/outputs.npy'))
    def merge_outputs():
        files = glob.glob(directory + '/blocks/*/outputs.npy')
        files.sort()
        merge(input_files=files,
              output_file=directory + '/outputs.npy',
              tempfile=directory + '/temp_outputs',
              in_memory=False,
              axis=0)

    return [merge_inputs, merge_outputs]


# Workflow definition ##########################################################


def train_and_evaluate(budget, batch_size, dependencies=None):
    if dependencies is None:
        dependencies = []

    root = problem + '/output/estimator/' + str(budget) + '/' + str(batch_size) + '/' + str(learning_rate)
    if arguments.gamma is None:
        gamma = 100.
    else:
        gamma = arguments.gamma
        root += '/' + str(gamma)

    if schedule_gamma:
        root += '/schedule'

    os.makedirs(root, exist_ok=True)

    # Train the ratio estimator
    @aw.cpus_and_memory(4, memory)
    @aw.dependency(dependencies)
    @aw.gpus(ngpus)
    @aw.postcondition(aw.num_files(root + '/*/weights.th', num_estimators))
    @aw.tasks(num_estimators)
    @aw.timelimit('72:00:00')
    def train(task_index):
        outputdir = root + '/' + str(task_index)
        os.makedirs(outputdir, exist_ok=True)
        if not os.path.exists(outputdir + '/weights.th'):
            optimize(outputdir, budget, gamma)

    # The training method is the dependency
    dependencies.append(train)

    # Evaluate its expected coverage
    @aw.cpus_and_memory(4, memory)
    @aw.dependency(train)
    @aw.gpus(ngpus)
    @aw.postcondition(aw.num_files(root + '/*/coverage.npy', num_estimators))
    @aw.postcondition(aw.num_files(root + '/*/contour-sizes.npy', num_estimators))
    @aw.postcondition(aw.num_files(root + '/*/bias.npy', num_estimators))
    @aw.postcondition(aw.num_files(root + '/*/bias_square.npy', num_estimators))
    @aw.postcondition(aw.num_files(root + '/*/variance.npy', num_estimators))
    @aw.tasks(num_estimators)
    @aw.timelimit('48:00:00')
    def coverage(task_index):
        outputdir = root + '/' + str(task_index)
        if not os.path.exists(outputdir + '/coverage.npy') or not os.path.exists(outputdir + '/contour-sizes.npy') \
        or not os.path.exists(outputdir + '/bias.npy') or not os.path.exists(outputdir + '/variance.npy')\
        or not os.path.exists(outputdir + '/bias_square.npy'):
            inputs = torch.from_numpy(np.load(problem + '/data/coverage/inputs.npy'))
            outputs = torch.from_numpy(np.load(problem + '/data/coverage/outputs.npy'))
            if small_test_val:
                inputs = inputs[:1000]
                outputs = outputs[:1000]

            if debug:
                inputs = inputs[:50]
                outputs = outputs[:50]

            outputs = outputs.to(h.accelerator)
            inputs = inputs.to(h.accelerator)

            ratio_estimator = load_estimator(outputdir + '/weights.th')
            # Modify this
            coverages, contour_sizes, bias, variance, bias_square = estimate_coverage(ratio_estimator, inputs, outputs, outputdir, confidence_levels)
            np.save(outputdir + '/coverage.npy', coverages)
            np.save(outputdir + '/contour-sizes.npy', contour_sizes)
            np.save(outputdir + '/bias.npy', bias)
            np.save(outputdir + '/bias_square.npy', bias_square)
            np.save(outputdir + '/variance.npy', variance)

    # Evaluate expected coverage of ensemble
    @aw.cpus_and_memory(4, memory)
    @aw.dependency(train)
    @aw.gpus(ngpus)
    @aw.postcondition(aw.exists(root + '/coverage.npy'))
    @aw.postcondition(aw.exists(root + '/contour-sizes.npy'))
    @aw.postcondition(aw.exists(root + '/bias.npy'))
    @aw.postcondition(aw.exists(root + '/bias_square.npy'))
    @aw.postcondition(aw.exists(root + '/variance.npy'))
    @aw.timelimit('48:00:00')
    def coverage_ensemble():
        outputdir = root
        if not os.path.exists(outputdir + '/coverage.npy') or not os.path.exists(outputdir + '/contour-sizes.npy') \
        or not os.path.exists(outputdir + '/bias.npy') or not os.path.exists(outputdir + '/variance.npy') \
        or not os.path.exists(outputdir + '/bias_square.npy'):
            inputs = torch.from_numpy(np.load(problem + '/data/coverage/inputs.npy'))
            outputs = torch.from_numpy(np.load(problem + '/data/coverage/outputs.npy'))
            if small_test_val:
                inputs = inputs[:1000]
                outputs = outputs[:1000]

            if debug:
                inputs = inputs[:50]
                outputs = outputs[:50]

            outputs = outputs.to(h.accelerator)
            inputs = inputs.to(h.accelerator)
            ratio_estimator = load_estimator(outputdir + '/*/weights.th')  # Load ensemble
            coverages, contour_sizes, bias, variance, bias_square = estimate_coverage(ratio_estimator, inputs, outputs, outputdir, confidence_levels)
            np.save(outputdir + '/coverage.npy', coverages)
            np.save(outputdir + '/contour-sizes.npy', contour_sizes)
            np.save(outputdir + '/bias.npy', bias)
            np.save(outputdir + '/bias_square.npy', bias_square)
            np.save(outputdir + '/variance.npy', variance)

    # Evaluate the approximate log posterior of the nominal values
    @aw.cpus_and_memory(4, memory)
    @aw.dependency(train)
    @aw.gpus(ngpus)
    @aw.postcondition(aw.num_files(root + '/*/log_posterior.npy', num_estimators))
    @aw.tasks(num_estimators)
    @aw.timelimit('2:00:00')
    def log_posterior(task_index):
        outputdir = root + '/' + str(task_index)
        outputfile = outputdir + '/log_posterior.npy'
        if not os.path.exists(outputfile):
            with torch.no_grad():
                ratio_estimator = load_estimator(outputdir + '/weights.th')
                # Load the data
                inputs = torch.from_numpy(np.load(problem + '/data/coverage/inputs.npy'))
                inputs = inputs.to(h.accelerator)
                outputs = torch.from_numpy(np.load(problem + '/data/coverage/outputs.npy'))
                outputs = outputs.to(h.accelerator)
                # Compute the log posterior
                log_posterior = compute_log_pdf(ratio_estimator, inputs, outputs).mean().cpu().numpy()
                np.save(outputfile, log_posterior)


    # Evaluate the state of the trained estimator's balancing condition
    @aw.cpus_and_memory(4, memory)
    @aw.dependency(train)
    @aw.postcondition(aw.num_files(root + '/*/balancing.npy', num_estimators))
    @aw.tasks(num_estimators)
    @aw.timelimit('48:00:00')
    def balancing_condition(task_index):
        outputdir = root + '/' + str(task_index)
        outputfile = outputdir + '/balancing.npy'
        if not os.path.exists(outputfile):
            with torch.no_grad():
                ratio_estimator = load_estimator(outputdir + '/weights.th')
                # Load the data
                inputs = torch.from_numpy(np.load(problem + '/data/coverage/inputs.npy'))
                inputs = inputs.to(h.accelerator)
                outputs = torch.from_numpy(np.load(problem + '/data/coverage/outputs.npy'))
                outputs = outputs.to(h.accelerator)
                # Compute expected discriminator output of the joint
                d_joint = ratio_estimator(inputs=inputs, outputs=outputs)[0]
                # Compute expected distriminator output of the product of marginals
                outputs = outputs[torch.randperm(len(inputs))]
                d_marginal = ratio_estimator(inputs=inputs, outputs=outputs)[0]
                # Estimate the balancing condition.
                balancing = (d_joint + d_marginal).mean().squeeze().cpu().numpy()
                np.save(outputfile, balancing)

    # Evaluate the state of the trained ensemble's balancing condition
    @aw.cpus_and_memory(4, memory)
    @aw.dependency(train)
    @aw.postcondition(aw.exists(root + '/balancing.npy'))
    @aw.timelimit('48:00:00')
    def balancing_condition_ensemble():
        outputdir = root
        outputfile = outputdir + '/balancing.npy'
        if not os.path.exists(outputfile):
            with torch.no_grad():
                ratio_estimator = load_estimator(outputdir + '/*/weights.th')
                # Load the data
                inputs = torch.from_numpy(np.load(problem + '/data/coverage/inputs.npy'))
                inputs = inputs.to(h.accelerator)
                outputs = torch.from_numpy(np.load(problem + '/data/coverage/outputs.npy'))
                outputs = outputs.to(h.accelerator)
                # Compute expected discriminator output of the joint
                d_joint = ratio_estimator(inputs=inputs, outputs=outputs)[0]
                # Compute expected distriminator output of the product of marginals
                outputs = outputs[torch.randperm(len(inputs))]
                d_marginal = ratio_estimator(inputs=inputs, outputs=outputs)[0]
                # Estimate the balancing condition.
                balancing = (d_joint + d_marginal).mean().squeeze().cpu().numpy()
                np.save(outputfile, balancing)

    # Compute the MI as E_joint[log rhat]
    @aw.cpus_and_memory(4, memory)
    @aw.dependency(train)
    @aw.postcondition(aw.num_files(root + '/*/mi-1.npy', num_estimators))
    @aw.tasks(num_estimators)
    @aw.timelimit('48:00:00')
    def mutual_information_1(task_index):
        outputdir = root + '/' + str(task_index)
        outputfile = outputdir + '/mi-1.npy'
        if not os.path.exists(outputfile):
            with torch.no_grad():
                ratio_estimator = load_estimator(outputdir + '/weights.th')
                # Load the data
                inputs = torch.from_numpy(np.load(problem + '/data/coverage/inputs.npy'))
                inputs = inputs.to(h.accelerator)
                outputs = torch.from_numpy(np.load(problem + '/data/coverage/outputs.npy'))
                outputs = outputs.to(h.accelerator)
                # Compute the estimated mutual information
                mi = ratio_estimator.log_ratio(inputs=inputs, outputs=outputs).mean().cpu().numpy()
                np.save(outputfile, mi)

    # Compute the MI as E_joint[log rhat] of the ensemble
    @aw.cpus_and_memory(4, memory)
    @aw.dependency(train)
    @aw.postcondition(aw.exists(root + '/mi-1.npy'))
    @aw.timelimit('48:00:00')
    def mutual_information_1_ensemble():
        outputdir = root
        outputfile = outputdir + '/mi-1.npy'
        if not os.path.exists(outputfile):
            with torch.no_grad():
                ratio_estimator = load_estimator(outputdir + '/*/weights.th')
                # Load the data
                inputs = torch.from_numpy(np.load(problem + '/data/coverage/inputs.npy'))
                inputs = inputs.to(h.accelerator)
                outputs = torch.from_numpy(np.load(problem + '/data/coverage/outputs.npy'))
                outputs = outputs.to(h.accelerator)
                # Compute the estimated mutual information
                mi = ratio_estimator.log_ratio(inputs=inputs, outputs=outputs).mean().cpu().numpy()
                np.save(outputfile, mi)

    # Compute the MI as E_joint[dhat * log rhat] + E_marginals[dhat * log rhat]
    @aw.cpus_and_memory(4, memory)
    @aw.dependency(train)
    @aw.postcondition(aw.num_files(root + '/*/mi-2.npy', num_estimators))
    @aw.tasks(num_estimators)
    @aw.timelimit('48:00:00')
    def mutual_information_2(task_index):
        outputdir = root + '/' + str(task_index)
        outputfile = outputdir + '/mi-2.npy'
        if not os.path.exists(outputfile):
            with torch.no_grad():
                ratio_estimator = load_estimator(outputdir + '/weights.th')
                # Load the data
                inputs = torch.from_numpy(np.load(problem + '/data/coverage/inputs.npy'))
                inputs = inputs.to(h.accelerator)
                outputs = torch.from_numpy(np.load(problem + '/data/coverage/outputs.npy'))
                outputs = outputs.to(h.accelerator)
                # Compute the estimated mutual information through the joint
                d_joint, log_r_joint = ratio_estimator(inputs=inputs, outputs=outputs)
                mi_joint = (d_joint * log_r_joint).mean()
                # Compute the estimated mutual information through the product of marginals
                outputs = outputs[torch.randperm(len(inputs))]
                d_marginal, log_r_marginal = ratio_estimator(inputs=inputs, outputs=outputs)
                mi_marginal = (d_marginal * log_r_marginal).mean()
                # Combine the estimated MI
                mi = (mi_joint + mi_marginal).cpu().numpy()
                np.save(outputfile, mi)

    # Compute the MI as E_joint[dhat * log rhat] + E_marginals[dhat * log rhat] for the ensemble
    @aw.cpus_and_memory(4, memory)
    @aw.dependency(train)
    @aw.postcondition(aw.exists(root + '/mi-2.npy'))
    @aw.timelimit('48:00:00')
    def mutual_information_2_ensemble():
        outputdir = root
        outputfile = outputdir + '/mi-2.npy'
        if not os.path.exists(outputfile):
            with torch.no_grad():
                ratio_estimator = load_estimator(outputdir + '/*/weights.th')
                # Load the data
                inputs = torch.from_numpy(np.load(problem + '/data/coverage/inputs.npy'))
                inputs = inputs.to(h.accelerator)
                outputs = torch.from_numpy(np.load(problem + '/data/coverage/outputs.npy'))
                outputs = outputs.to(h.accelerator)
                # Compute the estimated mutual information through the joint
                d_joint, log_r_joint = ratio_estimator(inputs=inputs, outputs=outputs)
                mi_joint = (d_joint * log_r_joint).mean()
                # Compute the estimated mutual information through the product of marginals
                outputs = outputs[torch.randperm(len(inputs))]
                d_marginal, log_r_marginal = ratio_estimator(inputs=inputs, outputs=outputs)
                mi_marginal = (d_marginal * log_r_marginal).mean()
                # Combine the estimated MI
                mi = (mi_joint + mi_marginal).cpu().numpy()
                np.save(outputfile, mi)

    # Compute the E_marginal[rhat]
    @aw.cpus_and_memory(4, memory)
    @aw.dependency(train)
    @aw.postcondition(aw.num_files(root + '/*/marginal_rhat.npy', num_estimators))
    @aw.tasks(num_estimators)
    @aw.timelimit('48:00:00')
    def marginal_rhat(task_index):
        outputdir = root + '/' + str(task_index)
        outputfile = outputdir + '/marginal_rhat.npy'
        if not os.path.exists(outputfile):
            with torch.no_grad():
                ratio_estimator = load_estimator(outputdir + '/weights.th')
                # Load the data
                inputs = torch.from_numpy(np.load(problem + '/data/coverage/inputs.npy'))
                inputs = inputs.to(h.accelerator)
                outputs = torch.from_numpy(np.load(problem + '/data/coverage/outputs.npy'))
                outputs = outputs.to(h.accelerator)
                outputs = outputs[torch.randperm(len(inputs))]
                # Compute the estimated mutual information
                marginal_rhat = ratio_estimator.log_ratio(inputs=inputs, outputs=outputs).exp().mean().cpu().numpy()
                np.save(outputfile, marginal_rhat)

    # Compute the E_marginal[rhat] of the ensemble
    @aw.cpus_and_memory(4, memory)
    @aw.dependency(train)
    @aw.postcondition(aw.exists(root + '/marginal_rhat.npy'))
    @aw.timelimit('48:00:00')
    def marginal_rhat_ensemble():
        outputdir = root
        outputfile = outputdir + '/marginal_rhat.npy'
        if not os.path.exists(outputfile):
            with torch.no_grad():
                ratio_estimator = load_estimator(outputdir + '/*/weights.th')
                # Load the data
                inputs = torch.from_numpy(np.load(problem + '/data/coverage/inputs.npy'))
                inputs = inputs.to(h.accelerator)
                outputs = torch.from_numpy(np.load(problem + '/data/coverage/outputs.npy'))
                outputs = outputs.to(h.accelerator)
                outputs = outputs[torch.randperm(len(inputs))]
                # Compute the estimated mutual information
                marginal_rhat = ratio_estimator.log_ratio(inputs=inputs, outputs=outputs).exp().mean().cpu().numpy()
                np.save(outputfile, marginal_rhat)


dependencies = generate_simulations()
for batch_size in batch_sizes:
    for budget in simulation_budgets:
        train_and_evaluate(budget, batch_size, dependencies=dependencies)


if __name__ == '__main__':
    aw.execute(partition=arguments.partition, name=problem)
