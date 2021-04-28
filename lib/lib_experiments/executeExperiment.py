"""Module that define experiments process."""


## Experiments process definition
def execute_experiment(experiment):
    results = []
    for flow in experiment['flows']:
        if flow['type'] == "SEPARATE_DATA":
            flow['flow'](experiment['dataset'])

        if flow['type'] == "LOAD_DATA":
            data = flow['flow'](experiment['dataset'], flow['transformation'], flow['split'])

        if flow['type'] == "INITIALIZE_MODEL":
            model = flow['flow'](flow['initialization'], experiment['dataset']['nb_classes'], flow['network'])

        if flow['type'] == "TRAIN_MODEL":
            results.append(flow['flow'](flow['device'], flow['hyperparameters'], 
                                        model, flow['optimizer'], flow['criterion'], flow['metrics'], 
                                        *data))

        if flow['type'] == "ANALYZE_RESULTS":
            print("analyzing BLA BLA BLA")

    print(results[0][1])
    print("Experiment done !")
