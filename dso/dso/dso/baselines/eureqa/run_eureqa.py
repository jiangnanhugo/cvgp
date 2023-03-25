import os
import json
import multiprocessing

import click
import pandas as pd
import datarobot as dr

MAX_WAIT = 3600
THRESHOLD = 1e-10


def work(arg):
    try:
        benchmark, eureqa_params, seed = arg
        print("Running {} with seed {}...".format(benchmark, seed))
        # Get the project
        project = get_project(project_name=benchmark)

        # Get the base model
        base_model = get_base_model(project=project)

        # Get the custom model
        model = get_model(project=project,
                          base_model=base_model,
                          eureqa_params=eureqa_params,
                          seed=seed)

        # For noisy datasets, evaluate all solutions along the pareto front.
        # If success, use that. Otherwise, use the most complex solution.
        if "_n" in benchmark:
            for solution in model.get_pareto_front().solutions:
                solution = solution.expression.split("Target = ")[-1]


        # Otherwise, evaluate the best solution
        else:
            solution = model.get_pareto_front().solutions[-1].expression
            solution = solution.split("Target = ")[-1]
        print("benchmark", benchmark)
        print('the solution is', solution)
        # Append results
        df = pd.DataFrame({
            "benchmark": [benchmark],
            "seed": [seed],
            "project_id": [project.id],
            "base_model_id": [base_model.id],
            "model_id": [model.id],
            "solution": [solution]
        })

    except Exception as e:
        print("Hit '{}' exception for {} on seed {}!".format(e, benchmark, seed))
        df = None

    return df


def start_client(credential_filepath):
    """Start the DataRobot client"""

    with open(credential_filepath, 'r') as f:
        credentials = json.load(f)

    # Start the DataRobot client
    print("Connecting to client...")
    dr.Client(token=credentials["token"], endpoint=credentials["endpoint"])


def get_project(project_name):
    """Get the project, or create one if it doesn't exist."""

    # If the project exists, return it.
    for project in dr.Project.list():
        if project.project_name == project_name:
            return project

    # Otherwise, create the project.
    path = os.path.join(project_name + ".csv")
    df = pd.read_csv(path, header=None)
    df.columns = ["x{}".format(i + 1) for i in range(len(df.columns) - 1)] + ["y"]
    project = dr.Project.start(project_name=project_name,
                               sourcedata=df,
                               autopilot_on=False,
                               target='y')

    # Unlock holdout dataset (required to access all training data)
    project.unlock_holdout()

    return project


def get_base_model(project):
    """Get the base model, or create one if it doesn't exist."""

    # If the base model exists in this project, return it.
    models = project.get_models()

    if len(models) > 0:
        model = models[0]

    else:
        # Find the Eureqa symbolic regression algorithm
        bp = [bp for bp in project.get_blueprints() if "Eureqa" in bp.model_type and "Instant" in bp.model_type][0]

        # Train the base model (required before adjusting parameters)
        model_job_id = project.train(bp, sample_pct=100.0)
        job = dr.ModelJob.get(model_job_id=model_job_id, project_id=project.id)
        model = job.get_result_when_complete(max_wait=MAX_WAIT)

    return model


def get_model(project, base_model, eureqa_params, seed):
    """Get the model, or create one if it doesn't exist."""

    # Set custom parameters
    tune = base_model.start_advanced_tuning_session()
    for parameter_name, value in eureqa_params.items():
        tune.set_parameter(parameter_name=parameter_name, value=value)

    # Set the seed
    tune.set_parameter(parameter_name="random_seed", value=seed)

    # Train the custom model.
    # The model may have already been run, which causes an error. This can
    # happen when the connection is lost, so the model completes on the server
    # but is not recorded on the client. When this happens, search for the model
    # with identical parameters and return it.
    try:
        job = tune.run()
        model = job.get_result_when_complete(max_wait=MAX_WAIT)
    except dr.errors.JobAlreadyRequested:
        print("Job was already requested! Searching for existing model...")
        models = project.get_models()
        eureqa_params_copy = eureqa_params.copy()
        eureqa_params_copy["random_seed"] = seed
        for model in models:
            parameters = model.start_advanced_tuning_session().get_parameters()["tuning_parameters"]
            parameters = {p["parameter_name"]: p["current_value"] for p in parameters}
            if parameters == eureqa_params_copy:
                print("Found existing model!", model.id)
                return model
        assert False, "Job was alredy run but could not find existing model."

    return model


# NOTE: Mac users first run `export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES`
# To run main experiments on Nguyen benchmarks: `python run_eureqa.py`
# To run main experiments on Constant benchmarks: `python run_eureqa.py --benchmark_set=Constant`
# To run noise experiments: `python run_eureqa.py results_noise.csv --mc=10 --seed_shift=1000 --sweep`
# To run test experiment: `python run_eureqa.py results_test.csv --num_workers=2 --mc=3 --seed_shift=123 --benchmark_set=Test`
@click.command()
@click.argument("results_path", type=str, default="results.csv")  # Path to existing results CSV, to checkpoint runs.
@click.option("--config_path", type=str, default="PATH-TO-CONFIG-FILE", help="Path to Eureqa JSON configuration.")
@click.option("--credential_path", type=str, default="PATH-TO-credential-FILE", help="Path to Eureqa JSON configuration.")
@click.option("--mc", type=int, default=5, help="Number of seeds to run.")
@click.option("--num_workers", type=int, default=10, help="Number of workers.")
@click.option("--seed_shift", type=int, default=0, help="Starting seed value.")
@click.option("--dataset_path", type=str, default="PATH-TO-dataset-FILE", help="Path to csv dataset")
@click.option("--nvars", type=int, default=2, help="number of multiple variables.")
def main(results_path, config_path,credential_path, mc, num_workers, seed_shift, dataset_path, nvars):
    """Run Eureqa on benchmarks for multiple random seeds."""

    # Load Eureqa paremeters
    eureqa_params = json.load(open(config_path, 'r'))
    eureqa_params['building_block__input_variable'] = nvars

    df = None
    write_header = True

    # Define the work
    args = []
    seeds = [i + seed_shift for i in range(mc)]
    prog_num = 10
    for seed in seeds:
        for i in range(prog_num):

            benchmarks = []
            benchmark = os.path.join(dataset_path, "prog_" + str(i))
            benchmarks.append(benchmark)

            for benchmark in benchmarks:
                args.append((benchmark, eureqa_params, seed))

    # Farm out the work
    if num_workers > 1:
        pool = multiprocessing.Pool(processes=num_workers, initializer=start_client, initargs=(credential_path,))
        for result in pool.imap_unordered(work, args):
            if result is not None:
                pd.DataFrame(result, index=[0]).to_csv(results_path + "/eureqa_result.csv", header=write_header, mode='a', index=False)
                write_header = False
    else:
        start_client(credential_path)
        for arg in args:
            result = work(arg)
            pd.DataFrame(result, index=[0]).to_csv(results_path + "/eureqa_result.csv", header=write_header, mode='a', index=False)
            write_header = False



if __name__ == "__main__":
    main()
