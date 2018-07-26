import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p',
        '--print-every',
        help='print every specific number of iteration',
        metavar='',
        type=int,
        default=100)
    parser.add_argument(
        '-b',
        '--batch-size',
        help='set number of batch size',
        metavar='',
        type=int,
        default=128)
    parser.add_argument(
        '-e',
        '--num-epochs',
        help='set number of total epochs',
        metavar='',
        type=int,
        default=5)
    parser.add_argument(
        '-lr',
        '--learning-rate',
        help='set training learning rate',
        metavar='',
        type=float,
        default=1e-3)
    parser.add_argument(
        '-v',
        '--verbose',
        help='set whether verbose train or not',
        action='store_true')
    parser.add_argument(
        '-t',
        '--train',
        help='set whether run in train mode or not',
        action='store_true')
    parser.add_argument(
        '-l',
        '--log',
        help='set whether save some result files',
        action='store_true')
    parser.add_argument(
        '-s',
        '--save-every',
        help='if set, save model while training',
        metavar='',
        type=int,
        default=0)
    parser.add_argument(
        '-m',
        '--model',
        help='load_pretrained model',
        metavar='',
        type=str,
        default=None)
    args = parser.parse_args()
    return args
