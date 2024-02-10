# TODO

## Few-Shot
- **implement**
    - dedicated dataset transformations for all feature extractors
    - **MO self-attention**
    - grad FE learning along with target, average pooling on weights/grads
    - MO batch norm
- **debug**
    - ~dataset~ 
    - ~batch forward through model~
    - *training loop*
    - for other ways and query values matrix mult fails (debug why)
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