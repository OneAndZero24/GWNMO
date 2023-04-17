from gwnmo.experiments.classic.mnist import exp
from gwnmo.utils import parser

args = parser.parse_args()
exp(args.epochs, args.lr)