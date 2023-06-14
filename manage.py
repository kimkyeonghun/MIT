import argparse
from command import CommandManager

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'c',
        help='specify the name of the command you want to run',
        metavar='COMMAND',
    )
    # BERTopic arguments
    parser.add_argument(
        '--load_mode',
        default=False,
    )
    parser.add_argument(
        '--model_name',
    )
    parser.add_argument(
        '--reduce_topic',
        default=900,
        type=int
    )
    parser.add_argument(
        '--kensho',
        default=False,
    )

    # KeyBERT arguments
    parser.add_argument(
        '--year',
        default=2,
        type=int
    )
    parser.add_argument(
        '--thres',
        default=0.25,
        type=float
    )
    args = parser.parse_args()

    runner = CommandManager(args)
    runner.run()
