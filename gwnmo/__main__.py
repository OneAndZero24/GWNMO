import experiments.classic.mnist as mnist
from utils import parser

args = parser.parse_args()
if args.exp == 'mnist.gwnmo':
    mnist.gwnmo(args.epochs, args.lr, args.gamma)
elif args.exp == 'mnist.adam':
    mnist.adam(args.epochs, args.lr)
elif args.exp == 'mnist.hypergrad':
    mnist.hypergrad(args.epochs, args.lr)