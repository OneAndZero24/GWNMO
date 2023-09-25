# Gradient Weighting by Neural Meta Optimizer
*Jan Miksa* @ GMUM Jagiellonian University

Meta-learning method inspired by hypergradient methods and earlier works of using neural networs to control training process of others.  
Uses a neural netowrk to "weight" target network's gradient during descent.

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
python gwnmo 100 classic --reps=1 --dataset=mnist --module=gwnmo --lr=0.01 --gamma=0.01
# FewShot
python gwnmo 10 fewshot --dataset=omniglot --module=gwnmofs
```