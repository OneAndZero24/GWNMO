# TODO

## Few-Shot
- ~~check GWNMOFS~~
- *part of FE trainable + smaller target "last layer test"*
- **model level parallelism**
- consider create_graph
- **implement**
    - dedicated dataset transformations for all feature extractors
    - **self-attention layer as MO**
    - grad FE learning along with target, average pooling on weights/grads
    - MO batch norm
- **grids inline with few-shot-hypernets-public**

## Experiment improvements
- **k-fold**
- better classification metrics

## Misc
- optuna integration

## Further
- test network per param
- **cross target model**
- cross dataset in classical setting MNIST -> FMNIST