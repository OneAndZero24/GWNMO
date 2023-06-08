from utils import parser, map2cmd, logger
from neptune_logger import NeptuneLogger

from modules.adam import Adam
from modules.gwnmo import GWNMO
from modules.hypergrad import HyperGrad

from train import train, train_twostep


args = parser.parse_args()

if args.noneptune == False:
    logger = NeptuneLogger(args)

dataset = map2cmd['dataset'][args.dataset]()
Module = map2cmd['module'][args.module]

global module
if Module == GWNMO:
    module = Module(args.lr, args.gamma, args.nonorm)

if Module == HyperGrad:
    module = Module(args.lr, args.gamma)
else:
    module = Module(args.lr)

if logger is not None:
    logger.log_model_summary(module)

if args.twostep:
    train_twostep(dataset, args.epochs, args.reps, module)
else:
   train(dataset, args.epochs, args.reps, module)
