# GRAND
Implement 'GRAND: Graph Neural Diffusion'

### Experiment 1 : Comparison GRAND vs GCN
```python
cd Experiment_1
python train1-wandb.py
```

### Experiment 2 : GNN based hyperparameter tuning
```python
cd Experiment_2
python train2-sweep-gnn.py
```

### Experiment 3 : ODE based hyperparameter tuning
```python
cd Experimentl_3
python train3-sweep-grand.py
```

### Experiment 4 : Comparison tuned GRAND vs original GRAND
```python
cd Experiment_4
python train-basic.py
python train_tuning.py
```
