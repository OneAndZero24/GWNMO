# Gradient Weighting by Neural Meta Optimizer
*Jan Miksa* @ GMUM Jagiellonian University  
*Mateusz Rajski* @ GMUM Jagiellonian University

Meta-learning method inspired by hypergradient methods and earlier works of using neural network to control training process of others.  
Uses a neural network to "weight" target network's gradient during descent. Can be use as meta-optimizer in classical learning process or realize few-shot learning.

*Work in progress...*

## Installation

```
pip install -r requirements.txt  
```

## Running

Help:  
```
python gwnmo -h 
```

Example:  
```
# Classic
python gwnmo 100 --lr=0.01 --gamma=0.01 classic --reps=1 --dataset=mnist --module=gwnmo
# FewShot
python gwnmo 100 fewshot --dataset=omniglot --module=gwnmofs
```
