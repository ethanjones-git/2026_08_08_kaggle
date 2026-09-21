import os
import sys
import gc
import re
import math
import time
import json
import traceback
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoConfig


'''
train, test = dataset(train_prct_sample = float, test_prct_sample = float)

trained_model_1 = model_1().train(cahce=True, train)

prediction = model_1().predict(test)

'''