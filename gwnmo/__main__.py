from utils import parser, map2cmd, logger

from modules.classic.adam import Adam
from modules.classic.gwnmo import GWNMO
from modules.classic.hypergrad import HyperGrad

from modules.fewshot.gwnmofs import GWNMOFS

from train import train
from trainfs import train as train_twostep

args = parser.parse_args()

logger.tag(args)

dataset_gen = map2cmd['dataset'][args.dataset]
Module = map2cmd['module'][args.module]

if args.mode == 'classic':
    if Module == GWNMO:
        module = Module(args.lr, args.gamma, not args.nonorm)
    elif Module == HyperGrad:
        module = Module(args.lr, args.gamma)
    else:
        module = Module(args.lr)

    logger.log_model_summary(module)

    train(dataset_gen(), args.epochs, args.reps, module)
else:
    if Module == GWNMOFS:
        module = Module(args.lr, args.lr2, args.gamma, not args.nonorm, args.steps, args.ways, args.shots)

    logger.log_model_summary(module)

    train_twostep(dataset_gen(args.ways, args.shots), args.epochs, module)