from __future__ import annotations

# Standard library
import base64
import json
import os
import random
import sys
import time

# Numerics, plotting, video
import numpy as np
import matplotlib.pyplot as plt
import imageio

# Notebook display helpers
from IPython.display import HTML, Markdown, display

# Gymnasium + MiniGrid
import gymnasium as gym
from gymnasium import spaces
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Door, Goal, Key, Lava, Wall
from minigrid.minigrid_env import MiniGridEnv as BaseMiniGridEnv

# Image preprocessing + deep learning
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

# On headless Linux (e.g. Colab) MiniGrid's renderer needs a virtual display.
if "google.colab" in sys.modules or sys.platform.startswith("linux"):
    try:
        import pyvirtualdisplay
        pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()
    except Exception as exc:  # pragma: no cover
        print(f"[display] virtual display not started: {exc}")

%matplotlib inline
plt.rcParams["figure.figsize"] = (6.0, 5.0)
