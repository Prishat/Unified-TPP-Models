## Interoperable Module for Training Temporal Point Process Models

### Model Classes
The Neural Network model Architectures and their corresponding class in the module is as follows:
1. Neural Jump Stochastic Differential Equations (NJSDE)
2. Recurrent Marked Temporal Point Processes (RMTPP)
3. Transformer Hawkes Process (THP)
4. The Neural Hawkes Process (NHP)

All the models are present in `models.py` file, the dataset loaders
are present in `datasets.py` and the default config files are
present in `config.py` file.

### Steps to Train and Evaluate the models
The example shows the steps to run __NJSDE__ on the __Pilotelevator__ dataset (from NHP).

- Import the __NHP__ data reader class and the default configurations.
```python
from module.datasets import nhpDatareader
from module.config import nhp_dataset_params
```
- Initialize the class object and read the data
```python
nhp_data = nhpDatareader(nhp_dataset_params)
train_data, val_data, total_event_num = nhp_data.read_data()
```
- Process the dataset to according to __NJSDE__ model specifications.
```python
train, tspan = nhp_data.process_njsde()
```

- Import the __NJSDE__ model and it's default configuration.
```python
from module.models import njsde
from module.config import njsde_params
```

- Initialize the model and run the `train` function.
```python
model = njsde(njsde_params)
model.train(train, tspan)
```

- To get the testset results
```python
model.predict(test, dt=1.0/30.0)
```
 
To see how to use other models with different datasets, look at the notebooks in the __examples__ folder.

### Required Installation
```
pip install git+https://github.com/000Justin000/torchdiffeq.git@jj585
```
