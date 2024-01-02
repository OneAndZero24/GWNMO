# TODO

## Fixes
- ~~grids SGD~~
    - ~~MNIST~~
    - ~~CIFAR10~~
    - ~~FMNIST~~
- ~~GWNMO Adam grid FMNIST~~
- ~~REFACTOR AFFTER FIX~~
- ~~GWNMO grid norm vs nonorm~~
- ~~REDO PDF~~
- ~~implement HG normalization~~
- ~~HG normalization CIFAR10 grid~~

## Few-Shot
- **implement**
    - ~~implement~~
    - ~~DEBUG~~
    - ~~fix few-shot - ways classes target output~~
    - ~~check gwnmofs training_step~~
    - ~~check differentation in torchviz~~
    - ~~fix memory consumption by fs datase*~~
    - **for other ways and query values matrix mult fails (debug why)**
    - dedicated dataset transformations for all feature extractors
- ~~bigger test grid~~
- ~~ones instead of weighting (optional till set epoch, flag)~~
- optuna integration
- **grids**

## Experiment improvements
- **make sure dataset is permuted between epochs**
- **k-fold**
- ~~log training metrics~~
- better classification metrics

## Further
- **smaller architectures**
- **metaoptimizer architecture**
    - **grad FE learning along with target, average pooling on weights/grads**
    - **self attention on grad**
    - bigger/other architectures
- model-parallelization that would allow for bigger models  
- test network per param
- **cross model**
- cross dataset in classical setting MNIST -> FMNIST
- bigger alternative grids