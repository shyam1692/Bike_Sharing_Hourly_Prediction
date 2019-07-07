# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 00:54:59 2019

@author: Shyam
"""

import argparse
parser = argparse.ArgumentParser(description="PyTorch implementation of Temporal Segment Networks")
parser.add_argument('filename', type=str, help = 'train / test file name')
parser.add_argument('--train', action="store_true")