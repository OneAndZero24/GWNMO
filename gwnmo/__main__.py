from utils import parser, map2cmd, logger

from modules.classic.adam import Adam
from modules.classic.gwnmo import GWNMO
from modules.classic.hypergrad import HyperGrad

from train import train, train_twostep


args = parser.parse_args()

logger.tag(args)

if args.mode == 'classic':
    dataset = map2cmd['dataset'][args.dataset]()
    Module = map2cmd['module'][args.module]

    global module
    if Module == GWNMO:
        module = Module(args.lr, args.gamma, not args.nonorm)
    elif Module == HyperGrad:
        module = Module(args.lr, args.gamma)
    else:
        module = Module(args.lr)

    logger.log_model_summary(module)

    if args.twostep:
        train_twostep(dataset, args.epochs, args.reps, module)
    else:
        train(dataset, args.epochs, args.reps, module)
else:
    dataset = map2cmd['dataset'][args.dataset]()
    Module = map2cmd['module'][args.module]

    # TODO

    trainfs()