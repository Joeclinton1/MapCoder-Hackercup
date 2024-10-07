import math
import os

from .Matus import Matus
from . import utils
from ..results import write_debug
import re
import xml.etree.ElementTree as ET
import concurrent.futures
import threading
import json
import time


# Path to the prompts YAML file
cwd = os.path.dirname(os.path.abspath(__file__))
prompts_file = os.path.join(cwd, 'prompt_templates/prompts_joe.yaml')
algorithms_file = os.path.join(cwd, 'prompt_templates/algorithm_list.yaml')
lang_specific_file = os.path.join(cwd, 'prompt_templates/lang_specific_tips.yaml')

# constants that affect how much computation it will use
NUM_PARALLEL = 5
NUM_SETS = 1
NUM_TRICKS_PER_SET = 2
MAX_IMPROVEMENT_TRIES = 1
NUM_SHOTS = 8