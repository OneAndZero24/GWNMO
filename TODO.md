# TODO

## Few-Shot
- ~~check GWNMOFS~~
- make GWNMOFS fit in memory
- consider create_graph
- **implement**
    - dedicated dataset transformations for all feature extractors
    - **MO self-attention**
    - grad FE learning along with target, average pooling on weights/grads
    - MO batch norm
- **grids inline with few-shot-hypernets-public**

## Experiment improvements
- **k-fold**
- better classification metrics

## Misc
- optuna integration

## Further
- model-parallelization that would allow for bigger models  
- test network per param
- **cross model**
- cross dataset in classical setting MNIST -> FMNIST