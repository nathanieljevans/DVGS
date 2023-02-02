#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

'''
# NOTE: 
# this file is modified from: https://github.com/google-research/google-research/blob/master/dvrl/data_loading.py

Citation: 

@misc{https://doi.org/10.48550/arxiv.1909.11671,
  doi = {10.48550/ARXIV.1909.11671},
  url = {https://arxiv.org/abs/1909.11671},
  author = {Yoon, Jinsung and Arik, Sercan O. and Pfister, Tomas},
  keywords = {Machine Learning (cs.LG), Machine Learning (stat.ML), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Data Valuation using Reinforcement Learning},
  publisher = {arXiv},
  year = {2019},
  copyright = {arXiv.org perpetual, non-exclusive license}
}

# DVRL license: 

# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

Data loading and preprocessing functions.
'''

import io
import os

import zipfile

import numpy as np
import pandas as pd

from six.moves import urllib
from sklearn import preprocessing


def load_tabular_data(data_name, dict_no, noise_rate, out='../data/adult/'):
  """Loads Adult Income and Blog Feedback datasets.
    This module loads the two tabular datasets and saves train.csv, valid.csv and
    test.csv files under data_files directory.
    UCI Adult data link: https://archive.ics.uci.edu/ml/datasets/Adult
    UCI Blog data link: https://archive.ics.uci.edu/ml/datasets/BlogFeedback
    If noise_rate > 0.0, adds noise on the datasets.
    Then, saves train.csv, valid.csv, test.csv on './data_files/' directory
    Args:
      data_name: 'adult' or 'blog'
      dict_no: training and validation set numbers
      noise_rate: label corruption ratio
    Returns:
      noise_idx: indices of noisy samples
  """

  # Loads datasets from links
  uci_base_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'

  # Adult Income dataset
  if data_name == 'adult':

    train_url = uci_base_url + 'adult/adult.data'
    test_url = uci_base_url + 'adult/adult.test'

    data_train = pd.read_csv(train_url, header=None)
    data_test = pd.read_csv(test_url, skiprows=1, header=None)

    df = pd.concat((data_train, data_test), axis=0)

    # Column names
    df.columns = ['Age', 'WorkClass', 'fnlwgt', 'Education', 'EducationNum',
                  'MaritalStatus', 'Occupation', 'Relationship', 'Race',
                  'Gender', 'CapitalGain', 'CapitalLoss', 'HoursPerWeek',
                  'NativeCountry', 'Income']

    # Creates binary labels
    df['Income'] = df['Income'].map({' <=50K': 0, ' >50K': 1,
                                      ' <=50K.': 0, ' >50K.': 1})

    # Changes string to float
    df.Age = df.Age.astype(float)
    df.fnlwgt = df.fnlwgt.astype(float)
    df.EducationNum = df.EducationNum.astype(float)
    df.EducationNum = df.EducationNum.astype(float)
    df.CapitalGain = df.CapitalGain.astype(float)
    df.CapitalLoss = df.CapitalLoss.astype(float)

    # One-hot encoding
    df = pd.get_dummies(df, columns=['WorkClass', 'Education', 'MaritalStatus',
                                      'Occupation', 'Relationship',
                                      'Race', 'Gender', 'NativeCountry'])

    # Sets label name as Y
    df = df.rename(columns={'Income': 'Y'})
    df['Y'] = df['Y'].astype(int)

    # Resets index
    df = df.reset_index()
    df = df.drop(columns=['index'])

  # Blog Feedback dataset
  elif data_name == 'blog':

    resp = urllib.request.urlopen(uci_base_url + '00304/BlogFeedback.zip')
    zip_file = zipfile.ZipFile(io.BytesIO(resp.read()))

    # Loads train dataset
    train_file_name = 'blogData_train.csv'
    data_train = pd.read_csv(zip_file.open(train_file_name), header=None)

    # Loads test dataset
    data_test = []
    for i in range(29):
      if i < 9:
        file_name = 'blogData_test-2012.02.0'+ str(i+1) + '.00_00.csv'
      else:
        file_name = 'blogData_test-2012.02.'+ str(i+1) + '.00_00.csv'

      temp_data = pd.read_csv(zip_file.open(file_name), header=None)

      if i == 0:
        data_test = temp_data
      else:
        data_test = pd.concat((data_test, temp_data), axis=0)

    for i in range(31):
      if i < 9:
        file_name = 'blogData_test-2012.03.0'+ str(i+1) + '.00_00.csv'
      elif i < 25:
        file_name = 'blogData_test-2012.03.'+ str(i+1) + '.00_00.csv'
      else:
        file_name = 'blogData_test-2012.03.'+ str(i+1) + '.01_00.csv'

      temp_data = pd.read_csv(zip_file.open(file_name), header=None)

      data_test = pd.concat((data_test, temp_data), axis=0)

    df = pd.concat((data_train, data_test), axis=0)

    # Removes rows with missing data
    df = df.dropna()

    # Sets label and named as Y
    df.columns = df.columns.astype(str)

    df['280'] = 1*(df['280'] > 0)
    df = df.rename(columns={'280': 'Y'})
    df['Y'] = df['Y'].astype(int)

    # Resets index
    df = df.reset_index()
    df = df.drop(columns=['index'])

  # Splits train, valid and test sets
  train_idx = range(len(data_train))
  train = df.loc[train_idx]

  test_idx = range(len(data_train), len(df))
  test = df.loc[test_idx]

  train_idx_final = np.random.permutation(len(train))[:dict_no['train']]

  temp_idx = np.random.permutation(len(test))
  valid_idx_final = temp_idx[:dict_no['valid']] + len(data_train)
  test_idx_final = temp_idx[dict_no['valid']:] + len(data_train)

  train = train.loc[train_idx_final]
  valid = test.loc[valid_idx_final]
  test = test.loc[test_idx_final]

  # Adds noise on labels
  y_train = np.asarray(train['Y'])
  y_train, noise_idx = corrupt_label(y_train, noise_rate)
  train['Y'] = y_train

  # Saves data
  if not os.path.exists(out):
    os.makedirs(out)

  train.to_csv(f'{out}/train.csv', index=False)
  valid.to_csv(f'{out}/valid.csv', index=False)
  test.to_csv(f'{out}/test.csv', index=False)

  # Returns indices of noisy samples
  return noise_idx



def preprocess_data(normalization, train_file_name, valid_file_name, test_file_name, data):
    """Loads datasets, divides features and labels, and normalizes features.
    Args:
      normalization: 'minmax' or 'standard'
      train_file_name: file name of training set
      valid_file_name: file name of validation set
      test_file_name: file name of testing set
    Returns:
      x_train: training features
      y_train: training labels
      x_valid: validation features
      y_valid: validation labels
      x_test: testing features
      y_test: testing labels
      col_names: column names
    """

    # Loads datasets
    train = pd.read_csv(f'{data}/'+train_file_name)
    valid = pd.read_csv(f'{data}/'+valid_file_name)
    test = pd.read_csv(f'{data}/'+test_file_name)

    # Extracts label
    y_train = np.asarray(train['Y'])
    y_valid = np.asarray(valid['Y'])
    y_test = np.asarray(test['Y'])

    # Drops label
    train = train.drop(columns=['Y'])
    valid = valid.drop(columns=['Y'])
    test = test.drop(columns=['Y'])

    # Column names
    col_names = train.columns.values.astype(str)

    # Concatenates train, valid, test for normalization
    df = pd.concat((train, valid, test), axis=0)

    # Normalization
    if normalization == 'minmax':
      scaler = preprocessing.MinMaxScaler()
    elif normalization == 'standard':
      scaler = preprocessing.StandardScaler()

    scaler.fit(df)
    df = scaler.transform(df)

    # Divides df into train / valid / test sets
    train_no = len(train)
    valid_no = len(valid)
    test_no = len(test)

    x_train = df[range(train_no), :]
    x_valid = df[range(train_no, train_no + valid_no), :]
    x_test = df[range(train_no+valid_no, train_no+valid_no+test_no), :]

    return x_train, y_train, x_valid, y_valid, x_test, y_test, col_names



# https://github.com/google-research/google-research/blob/master/dvrl/dvrl_utils.py
def corrupt_label(y_train, noise_rate):
    """Corrupts training labels.
    Args:
      y_train: training labels
      noise_rate: input noise ratio
    Returns:
      corrupted_y_train: corrupted training labels
      noise_idx: corrupted index
    """

    y_set = list(set(y_train))

    # Sets noise_idx
    temp_idx = np.random.permutation(len(y_train))
    noise_idx = temp_idx[:int(len(y_train) * noise_rate)]

    # Corrupts label
    corrupted_y_train = y_train[:]

    for itt in noise_idx:
      temp_y_set = y_set[:]
      del temp_y_set[y_train[itt]]
      rand_idx = np.random.randint(len(y_set) - 1)
      corrupted_y_train[itt] = temp_y_set[rand_idx]

    return corrupted_y_train, noise_idx