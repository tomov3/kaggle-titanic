import numpy as np
import pandas as pd

TRAIN_DATA = '../input/train.csv'
TEST_DATA = '../input/test.csv'

from logging import getLogger

logger = getLogger(__name__)

def read_csv(path):
    logger.debug('enter')
    df = pd.read_csv(path);
    logger.debug('exit')

    return df

def load_train_data():
    logger.debug('enter')
    df = read_csv(TRAIN_DATA)
    logger.debug('exit')

    return df

def load_test_data():
    logger.debug('enter')
    df = read_csv(TEST_DATA)
    logger.debug('exit')

    return df
