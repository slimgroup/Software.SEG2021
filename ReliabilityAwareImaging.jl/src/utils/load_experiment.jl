export load_experiment

    """
    load_experiment(sim_name::String)

    A function that uses the input variables to the experiment, passed by dictionaty
    `parsed_args`, generated via `ArgParse.jl` package, to determine which experiment
    data to load from `datadir(sim_name)`, where `sim_name` is the simulation name.

    In case there are multiple experiments with the same variable valus, the last
    experiment will be returned

 *Input*:

 - `sim_name`: Simulation name to upload
 - `collect_results`: Simulation name to upload

    """

function load_experiment(parsed_args::Dict; return_path=false)

    # Upload data/sim_name
    sim_name = parsed_args["sim_name"]
    experiments = collect_results(datadir(sim_name), black_list=["Params", "fval"])

    # Drop experiments that miss one or some of the input arguments
    dropmissing!(experiments)

    # Keep experiment with same variables
    for (key, value) in parsed_args
        experiments = experiments[experiments[!, string(key)] .== value, :]
    end

    @assert size(experiments, 1) > 0 "No saved experiments found with such input values"

    println(size(experiments, 1), " experiment(s) found — loading the latest")
    experiment_path = experiments[1, :path]
    Params = wload(experiment_path)["Params"]
    fval = wload(experiment_path)["fval"]

    return_path ? (return Params, fval, experiment_path) : (return Params, fval)

end
