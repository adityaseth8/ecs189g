from code.base_class.method import method
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Method_MNIST(method, nn.Module):
    data = None

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)