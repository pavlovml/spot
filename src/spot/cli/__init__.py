from . import train, test, demo
from argparse import ArgumentParser
import sys

parser = ArgumentParser(description='An object detection toolchain for Caffe (Faster RCNN)')
subparsers = parser.add_subparsers()
train.add_subparser(subparsers)
test.add_subparser(subparsers)
demo.add_subparser(subparsers)

def run():
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    else:
        args = parser.parse_args()
        args.func(args)
