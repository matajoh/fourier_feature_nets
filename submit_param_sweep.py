"""Script which submits other scripts to be run in Azure Machine Learning."""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import os

from azureml.core import (
    Environment,
    Experiment,
    ScriptRunConfig,
    Workspace
)
from azureml.train.hyperdrive import (
    BayesianParameterSampling,
    HyperDriveConfig,
    PrimaryMetricGoal,
    uniform
)


def _parse_args():
    parser = ArgumentParser("Azure ML Experiment Runner",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("name", help="Name of the experiment")
    parser.add_argument("compute", help="Name of the compute target")
    parser.add_argument("script_path", help="Path to the script to run")
    parser.add_argument("param", help="The parameter to sweep")
    parser.add_argument("script_args", help="The script args")
    parser.add_argument("--num-runs", type=int, default=20,
                        help="Total number of runs")
    parser.add_argument("--concurrent_runs", type=int, default=4,
                        help="Number of runs at the same time.")
    parser.add_argument("--min-val", type=float, default=1,
                        help="The minimum value in the sampling range.")
    parser.add_argument("--max-val", type=float, default=10)
    env_default = "AzureML-pytorch-1.9-ubuntu18.04-py37-cuda11-gpu"
    parser.add_argument("--env",
                        default=env_default,
                        help="The curated environment to use.")
    return parser.parse_args()


def _main():
    args = _parse_args()

    ws = Workspace.from_config()
    experiment = Experiment(workspace=ws, name=args.name)
    env_path = os.path.join("azureml", "aml_env.yml")
    environment = Environment.from_conda_specification("training", env_path)
    param_sampling = BayesianParameterSampling({
            "--{}".format(args.param): uniform(args.min_val, args.max_val)
        }
    )
    script_run_config = ScriptRunConfig(source_directory=".",
                                        script=args.script_path,
                                        arguments=args.script_args.split(),
                                        compute_target=args.compute,
                                        environment=environment)

    hd_config = HyperDriveConfig(run_config=script_run_config,
                                 hyperparameter_sampling=param_sampling,
                                 primary_metric_name="psnr_val",
                                 primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
                                 max_total_runs=args.num_runs,
                                 max_concurrent_runs=args.concurrent_runs)

    run = experiment.submit(hd_config)

    aml_url = run.get_portal_url()
    print(aml_url)


if __name__ == "__main__":
    _main()
