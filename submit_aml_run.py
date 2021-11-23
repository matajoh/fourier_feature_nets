"""Script which submits other scripts to be run in Azure Machine Learning."""

import argparse
import os

from azureml.core import (
    Environment,
    Experiment,
    ScriptRunConfig,
    Workspace
)


def _parse_args():
    parser = argparse.ArgumentParser("Azure ML Experiment Runner")
    parser.add_argument("name", help="Name of the experiment")
    parser.add_argument("compute", help="Name of the compute target")
    parser.add_argument("script_path", help="Path to the script to run")
    parser.add_argument("script_args", help="The script args")
    return parser.parse_args()


def _main():
    args = _parse_args()

    ws = Workspace.from_config()
    experiment = Experiment(workspace=ws, name=args.name)
    env_path = os.path.join("azureml", "aml_env.yml")
    environment = Environment.from_conda_specification("training", env_path)
    environment.environment_variables["AZUREML_COMPUTE_USE_COMMON_RUNTIME"] = "false"
    config = ScriptRunConfig(source_directory=".",
                             script=args.script_path,
                             arguments=args.script_args.split(),
                             compute_target=args.compute,
                             environment=environment)

    run = experiment.submit(config)

    aml_url = run.get_portal_url()
    print(aml_url)


if __name__ == "__main__":
    _main()
