from typing import List
from math import pi
import yaml
import argparse
import glob
import logging
import math
import os
import numpy.random as random
import sys
from queue import Empty
import pygame
from pygame.locals import KMOD_CTRL
from pygame.locals import K_ESCAPE
from pygame.locals import K_q
import numpy as np
import logging
import shutil
from numpy.linalg import inv
import datetime
from PIL import Image
import weakref
import collections
from queue import Queue
import struct
import cv2
import open3d as o3d
from tqdm import tqdm
import re

# ==============================================================================
# -- Find CARLA module ---------------------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('/home/ying/CARLA_0.9.13/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- Add PythonAPI for release mode --------------------------------------------
# ==============================================================================
try:
    sys.path.append('/home/ying/CARLA_0.9.13/PythonAPI/carla')
except IndexError:
    pass

import carla
from carla import ColorConverter as cc

from agents.navigation.behavior_agent import BehaviorAgent  # pylint: disable=import-error
from agents.navigation.basic_agent import BasicAgent  # pylint: disable=import-error