import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import csv
from pandas import DataFrame

with open("../input/train.json") as train_file:
    train_data = json.load(train_file)
with open("../input/test.json") as test_file:
    test_data = json.load(test_file)
with open("../input/validation.json") as validation_file:
    validation_dat = json.load(validation_file)

