import gpforecaster as gpf
import htsmodels as hts
import tsaugmentation as tsag
import matplotlib.pyplot as plt
import os

dataset_name = 'tourism'
dataset = tsag.preprocessing.PreprocessDatasets(dataset_name)
groups = dataset.apply_preprocess()

data = tsag.transformations.CreateTransformedVersions(dataset_name)
data.parameters = {"jitter": 1.2,
                   "scaling": 0.2,
                   "magnitude_warp": 0.1,
                   "time_warp": 0.001}

vis = tsag.visualization.Visualizer(dataset_name)

algorithm = 'mint'
for k in ['jitter', 'scaling', 'magnitude_warp', 'time_warp']:
    vis._read_files(f'single_transf_{k}')
    for i in range(6):
        for j in range(10):
            groups['train']['data'] = vis.y_new[i, j]
            mint = hts.models.MinT(dataset=f'{dataset_name}_{algorithm}_{k}_v{i}_s{j}',
                                   groups=groups,
                                   aggregate_key = '(State / Zone / Region) * Purpose')
            forecasts = mint.train()
            results = mint.results(forecasts)
            res = mint.metrics(results)
            mint.store_metrics(res)
