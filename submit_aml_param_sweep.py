"""Script which submits other scripts to be run in Azure Machine Learning."""

import argparse
import os

from azureml.core import (
    Environment,
    Experiment,
    ScriptRunConfig,
    Workspace
)
from azureml.train.hyperdrive import (
    BayesianParameterSampling,
    choice,
    HyperDriveConfig,
    MedianStoppingPolicy,
    PrimaryMetricGoal,
    uniform
)


def _parse_args():
    parser = argparse.ArgumentParser("Azure ML Experiment Runner")
    parser.add_argument("name", help="Name of the experiment")
    parser.add_argument("compute", help="Name of the compute target")
    parser.add_argument("script_path", help="Path to the script to run")
    parser.add_argument("script_args", help="The script args")
    parser.add_argument("--num-runs", type=int, default=20)
    env_default = "AzureML-pytorch-1.9-ubuntu18.04-py37-cuda11-gpu"
    parser.add_argument("--env",
                        default=env_default,
                        help="The curated environment to use.")
    return parser.parse_args()


PARAMS_BY_SCRIPT = {
    "train_nerf.py": {
        "pos-freq": uniform()
    },
    "train_tiny_nerf.py": {
        "positional": {

        },

        "gaussian": {

        }
    },
    "train_image_regression.py": {
        "positional": {

        },
        "gaussian": {

        }
    }
}


def _main():
    args = _parse_args()

    ws = Workspace.from_config()
    experiment = Experiment(workspace=ws, name=args.name)
    env_path = os.path.join("azureml", "aml_env.yml")
    environment = Environment.from_conda_specification("training", env_path)
    param_sampling = BayesianParameterSampling({
            "learning_rate": uniform(0.05, 0.1),
            "batch_size": choice(16, 32, 64, 128)
        }
    )
    early_termination_policy = MedianStoppingPolicy()
    script_run_config = ScriptRunConfig(source_directory=".",
                                        script=args.script_path,
                                        arguments=args.script_args.split(),
                                        compute_target=args.compute,
                                        environment=environment)

    hd_config = HyperDriveConfig(run_config=script_run_config,
                                 hyperparameter_sampling=param_sampling,
                                 policy=early_termination_policy,
                                 primary_metric_name="psnr_val",
                                 primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
                                 max_total_runs=100,
                                 max_concurrent_runs=4)

    run = experiment.submit(hd_config)

    aml_url = run.get_portal_url()
    print(aml_url)


if __name__ == "__main__":
    _main()
