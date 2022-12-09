import sys

from pathlib import Path
import yaml
import argparse

import utils
import trainer

parser = argparse.ArgumentParser()
parser.add_argument("-c", help="config location",required=True)
args = parser.parse_args()

config_path = args.c

deep_trainer = utils.TrainEngine(config_pth=config_path)
deep_trainer.run()