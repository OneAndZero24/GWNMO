# TODO

## Few-Shot
- **implement**
    - dedicated dataset transformations for all feature extractors
    - **MAML**
    - **MO Self-Attention**
    - grad FE learning along with target, average pooling on weights/grads
- **debug**
    - **for other ways and query values matrix mult fails (debug why)**
    - **check for vanishing grads**
    - MO batch norm
- **grids**

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