from utils import device, parser, map2cmd, logger
from modules.classic.adam import Adam
from modules.classic.gwnmo import GWNMO
from modules.classic.hypergrad import HyperGrad

from modules.fewshot.gwnmofs import GWNMOFS

from train import train
from trainfs import train as train_twostep

if __name__ == '__main__':
    args = parser.parse_args()

    if args.offline == False and args.debugger != True:
        logger.toggle_online()

    logger.get().tag(args)

    dataset_gen = map2cmd['dataset'][args.dataset]
    Module = map2cmd['module'][args.module]

    if args.mode == 'classic':
        if Module in {GWNMO, HyperGrad}:
            module = Module(args.lr, args.gamma, not args.nonorm)
        else:
            module = Module(args.lr)

        logger.get().log_model_summary(module)

        train(dataset_gen(), args.epochs, args.reps, module)
    else:
        if Module == GWNMOFS:
            module = Module(args.lr, args.lr2, args.gamma, not args.nonorm, args.steps, args.ways, args.shots, trainable_fe = args.trainable_fe, feature_extractor_backbone=args.backbone_type)

        logger.get().log_model_summary(module)

        train_twostep(dataset_gen(device, args.ways, args.shots), args.epochs, module)