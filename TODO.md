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
- ~~few-shot~~
    - ~~implement~~
    - ~~DEBUG~~
    - ~~fix few-shot - ways classes target output~~
    - ~~check gwnmofs training_step~~
    - ~~check differentation in torchviz~~
- **compare to tables from prev paper crosschar**
    - *implement same architecture* Mati
    - grids
- extend
    - MetaSGD/MAML
    - grids omniglot

## Experiment improvements
- **make sure dataset is permuted between epochs**
- **k-fold**
- ~~log training metrics~~
- better classification metrics
- **smaller architectures**

## Further
- **attention on optimizer steps**
- **metaoptimizer architecture**
    - **grad FE learning along with target, average pooling on weights/grads**
    - **self attention on grad**
    - bigger/other architectures
- model-parallelization that would allow for bigger models  
- test network per param
- **cross model**
- cross dataset in classical setting MNIST -> FMNIST
- bigger alternative grids
- **longer experiments**
- GWNMO as LLM finetuning