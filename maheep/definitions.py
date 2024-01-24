from pprint import pprint
import os
import numpy as np
import torch
import torchvision
from torchvision import datasets, models, transforms
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import re


def p(x):
    print()
    pprint(x)