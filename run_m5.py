import gpforecaster as gpf
import htsmodels as hts
import tsaugmentation as tsag
import matplotlib.pyplot as plt
import os

dataset_name = 'prison'
dataset = tsag.preprocessing.PreprocessDatasets(dataset_name)
groups = dataset.apply_preprocess()

algorithm = 'gpf'
gpf_model = gpf.model.GPF(f'{dataset_name}_{algorithm}', groups)
model, like = gpf_model.train()
mean, lower, upper = gpf_model.predict(model, like)
res = gpf_model.metrics(mean)
gpf_model.store_metrics(res)

algorithm = 'mint'
mint = hts.models.MinT(dataset=f'{dataset_name}_{algorithm}',
                                   groups=groups,
                                   aggregate_key = '(Category / Department) * (State / Store / Item)')
forecasts = mint.train()
results = mint.results(forecasts)
res = mint.metrics(results)
mint.store_metrics(res)

algorithm = 'deepar'
deepar = hts.models.DeepAR(dataset=f'{dataset_name}_{algorithm}', groups=groups)
model = deepar.train()
forecasts = deepar.predict(model)
results = deepar.results(forecasts)
res = deepar.metrics(results)
deepar.store_metrics(res)
