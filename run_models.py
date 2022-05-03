import warnings
warnings.filterwarnings("ignore")
import gpforecaster as gpf
import htsmodels as hts
import tsaugmentation as tsag
import argparse


def create_groups_from_data(dataset_name):
    dataset = tsag.preprocessing.PreprocessDatasets(dataset_name)
    groups = dataset.apply_preprocess()

    vis = tsag.visualization.Visualizer(dataset_name)

    return groups, vis


def create_transformations(dataset_name):
    # Create transformed datasets
    data = tsag.transformations.CreateTransformedVersions(dataset_name)
    
    # Parameters for the tourism dataset
    data.parameters = {"jitter": 1.5,
                       "scaling": 0.3,
                       "magnitude_warp": 0.3,
                       "time_warp": 0.005}

    data.create_new_version_single_transf()


def run_original_algorithm(dataset_name, algorithms, transformations, groups, aggregate_key):
    for algorithm in algorithms:
        for k in transformations:
            # run algorithms for the original version of the dataset
            if algorithm=='deepar':
                run_deepar(dataset=f'{dataset_name}_{algorithm}_{k}_orig_s0', groups=groups)
            elif algorithm=='mint':
                run_mint(dataset=f'{dataset_name}_{algorithm}_{k}_orig_s0',
                         groups=groups,
                         aggregate_key=aggregate_key)
            elif algorithm=='gpf':
                run_gpf(dataset=f'{dataset_name}_{algorithm}_{k}_orig_s0', groups=groups)


def run_algorithm(dataset_name, algorithms, transformations, groups, vis, aggregate_key):
    for algorithm in algorithms:
        for k in transformations:
            # run algorithms for the transformed versions of the dataset
            vis._read_files(f'single_transf_{k}')
            for i in range(6):
                for j in range(10):
                    groups['train']['data'] = vis.y_new[i, j]
                    if algorithm=='deepar':
                        run_deepar(dataset=f'{dataset_name}_{algorithm}_{k}_v{i}_s{j}', groups=groups)
                    elif algorithm=='mint':
                        run_mint(dataset=f'{dataset_name}_{algorithm}_{k}_v{i}_s{j}', 
                                groups=groups, 
                                aggregate_key=aggregate_key)
                    elif algorithm=='gpf':
                        run_gpf(dataset=f'{dataset_name}_{algorithm}_{k}_v{i}_s{j}', groups=groups)

def run_deepar(dataset, groups):
    deepar = hts.models.DeepAR(dataset=dataset, groups=groups, 
                               store_prediction_samples=True, 
                               store_prediction_points=True)
    model = deepar.train()
    forecasts = deepar.predict(model)
    samples = deepar.results(forecasts)
    res = deepar.metrics(samples)
    deepar.store_metrics(res)

def run_mint(dataset, groups, aggregate_key):
    mint = hts.models.MinT(dataset=dataset,
                           groups=groups,
                           aggregate_key=aggregate_key, 
                           store_prediction_samples=True, 
                           store_prediction_points=True)
    forecasts = mint.train()
    df_results = mint.results(forecasts)
    res = mint.metrics(df_results)
    mint.store_metrics(res)

def run_gpf(dataset, groups):
    gpf_model = gpf.model.GPF(dataset, groups, 
                              store_prediction_samples=True, 
                              store_prediction_points=True)
    model, like = gpf_model.train()
    samples = gpf_model.predict(model, like)
    res = gpf_model.metrics(samples)
    gpf_model.store_metrics(res)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', '-a', 
            nargs='+', 
            help="select an algorithm", 
            type=str, 
            default=['gpf', 'mint', 'deepar'])
    parser.add_argument('--transformation', '-t', 
            nargs='+', 
            help="select a transformation", 
            type=str,
            default=['jitter', 'scaling', 'magnitude_warp', 'time_warp']) 
    parser.add_argument('--dataset', '-d', 
            nargs='+',
            help="select a dataset", 
            type=str, 
            default=['tourism'])
    parser.add_argument('--execution', '-e', 
            nargs='+',
            help="select type of execution", 
            type=str, 
            default=['original'])
    algo_transf = {}
    for k, v in parser.parse_args()._get_kwargs():
        algo_transf[k] = v

    assert all(x in ['gpf', 'mint', 'deepar'] for x in algo_transf['algorithm']), 'The algorithm is not implemented'        
    assert all(x in ['jitter', 'scaling', 'magnitude_warp', 'time_warp'] for x in algo_transf['transformation']), 'Transformation not implemented'
    assert all(x in ['tourism', 'prison', 'm5', 'police'] for x in algo_transf['dataset']), 'Dataset not implemented'
    assert all(x in ['original', 'transformed'] for x in algo_transf['execution']), 'Execution not implemented'

    return algo_transf


if __name__ == "__main__":
    algo_transf = parse_args()

    if algo_transf['dataset'][0]=='tourism':
        aggregate_key = '(State / Zone / Region) * Purpose'
    elif algo_transf['dataset'][0]=='prison':
        aggregate_key = 'Gender * Legal * State'
    elif algo_transf['dataset'][0]=='m5':
        aggregate_key = '(Category / Department / Item) * (State * Store)'
    elif algo_transf['dataset'][0]=='police':
        aggregate_key = 'Crime * Beat * Street * Zip'

    groups, vis = create_groups_from_data(algo_transf['dataset'][0])
    #create_transformations(algo_transf['dataset'][0])
    if algo_transf['execution'][0]=='original':
        run_original_algorithm(algo_transf['dataset'][0], algo_transf['algorithm'], algo_transf['transformation'], groups, aggregate_key)
    elif algo_transf['execution'][0]=='transformed':
        run_algorithm(algo_transf['dataset'][0], algo_transf['algorithm'], algo_transf['transformation'], groups, vis, aggregate_key)
