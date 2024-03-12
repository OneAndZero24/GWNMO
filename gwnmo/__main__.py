import torch

from utils import device, parser, map2cmd, logger
from modules.classic.adam import Adam
from modules.classic.gwnmo import GWNMO
from modules.classic.hypergrad import HyperGrad

from modules.fewshot.gwnmofs import GWNMOFS

from train import train
from trainfs import train as train_fs

if __name__ == '__main__':
    args = parser.parse_args()

    if args.offline == False:
        logger.toggle_online()

    logger.get().tag(args)
    logger.get().output('cuda_devices', torch.cuda.device_count())

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
            module = Module(args.lr, args.lr2, args.gamma, not args.nonorm, args.steps, args.ways, args.shots, args.query, args.trainable_fe, args.backbone_type)
        else:
            module = Module(args.lr, args.lr2, args.steps, args.ways, args.shots, args.query, args.trainable_fe, args.backbone_type)

        logger.get().log_model_summary(module)

        dataset = dataset_gen(device, args.ways, args.shots, args.query, args.tasks)
        train_fs(dataset, args.epochs, module, args.no_weighting)