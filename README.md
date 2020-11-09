# BOASF

The code for paper "BOASF: A Unified Framework for Speeding up Automatic Machine Learning via Adaptive Successive Filtering".

## Dataset

All datasets are from openml dataset. You should put all openml datasets in a base path and each dataset should have three file, for example:
```
sample_data/475_X.npy
sample_data/475_y.npy
sample_data/475_cat.npy

sample_data/37_X.npy
sample_data/37_y.npy
sample_data/37_cat.npy
```

## Run

You can run the scripts in the scripts folder, for example:

```
sh scripts/openml_model_selection_boasf.sh # use BOASF to select model
sh scripts/openml_hyperparameter_search_boasf_lr.sh # use BOASF to search hyperparameter of LogisticRegression
sh scripts/openml_hyperparameter_search_boasf_rf.sh # use BOASF to search hyperparameter of RandomForest 
```

## License

The codes and models in this repo are released under the GNU GPLv3 license.