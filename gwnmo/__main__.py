from gwnmo.experiments.classic.mnist import gwnmo
from gwnmo.utils import parser

args = parser.parse_args()
gwnmo(args.epochs, args.lr)
# TODO flags to select scenario