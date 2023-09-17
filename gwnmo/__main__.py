from utils import parser, map2cmd, logger

from modules.classic.adam import Adam
from modules.classic.gwnmo import GWNMO
from modules.classic.hypergrad import HyperGrad

from train import train
from trainfs import train as train_twostep

args = parser.parse_args()

logger.tag(args)

dataset = map2cmd['dataset'][args.dataset]()
Module = map2cmd['module'][args.module]

global module
if Module == GWNMO:
    module = Module(args.lr, args.gamma, not args.nonorm)
elif Module == HyperGrad:
    module = Module(args.lr, args.gamma)
else:
    module = Module(args.lr)

# TODO FS GWNMOFS, MAML, MetaSGD

logger.log_model_summary(module)

if args.mode == 'classic':
    train(dataset, args.epochs, args.reps, module)
else:
    train_twostep()