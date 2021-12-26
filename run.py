import gpforecaster as gpf
import htsmodels as hts
import tsaugmentation as tsag
import matplotlib.pyplot as plt
import os
import sys

dataset_name = 'tourism'
dataset = tsag.preprocessing.PreprocessDatasets(dataset_name)
groups = dataset.apply_preprocess()

vis = tsag.visualization.Visualizer(dataset_name)

tansfs = sys.argv[1:]

algorithm = 'gpf'
for k in transfs:
    vis._read_files(f'single_transf_{k}')
    for i in range(6):
        for j in range(10):
            groups['train']['data'] = vis.y_new[i, j]
            gpf_model = gpf.model.GPF(f'{dataset_name}_{algorithm}_{k}_v{i}_s{j}', groups)
            model, like = gpf_model.train()
            mean, lower, upper = gpf_model.predict(model, like)
            res = gpf_model.metrics(mean)
            gpf_model.store_metrics(res)
