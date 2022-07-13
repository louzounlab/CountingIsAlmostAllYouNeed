# from preprocessing import Preprocessing
import argparse
import copy
import csv
import json
import math
import os
import pickle
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
from Count.Datasets import RepertoireDataset
from colorama import Fore
from scipy import stats
from sklearn.metrics import f1_score
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import roc_auc_score
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ARGS = {}

"""
The class contains all the information about repertoires in the training/validation/test set
"""


class Repertoires:
    def __init__(self, name, nump):
        self.name = name
        self.combinations = {}
        self.__id2patient = {}
        self.score = {}
        self.outlier = {}
        self.size = nump
        self.cmv = None

    """
    creates the dict the connects people to their index
    """

    def personal_information(self, directory, files=None):
        directory = Path(directory)
        if files is None:
            files = list(directory.glob('*'))
        else:
            self.size = min(self.size, len(files))
        samples = random.sample(range(len(files)), self.size)
        count = 0
        for ind, item in tqdm(enumerate(files), total=self.size, desc="Maintain patient order",
                              bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.LIGHTBLUE_EX, Fore.RESET)):

            if ind not in samples:
                continue

            # attr hold the patient and the path to him
            attr = item.stem.split('_')
            attr.append(item.as_posix())
            self.__id2patient[tuple(attr)] = count
            count += 1

    """
    creates the vector that tells us whether a person i is pos/neg
    """

    def create_cmv_vector(self):

        self.cmv = np.zeros(self.size, dtype=np.int8)
        for item, index in self.__id2patient.items():
            if item[1] == 'positive':
                self.cmv[index] = 1

    """
    Create the dictionary to which all the data is loaded to. For each TCR in the data, 
    A vector is created that contains for each repertoire the number of times the TCR appears 
    in the repertoire. The vector is saved in the combinations dictionary.
    """

    def create_combinations(self):
        print(f"\n{Fore.LIGHTBLUE_EX}Generate a quantity of {Fore.LIGHTMAGENTA_EX}instances combination{Fore.RESET}")
        start_time = time.time()
        for personal_info, ind in tqdm(self.__id2patient.items(), total=self.size, desc='Create Combinations',
                                       bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.LIGHTBLUE_EX, Fore.RESET)):
            # reading each person's file to fill the appropriate place in the dict for that person
            if ind >= self.size:
                break
            _, _, path = personal_info
            df = pd.read_csv(path, usecols=['combined'])
            v_comb = df["combined"]
            for element in v_comb:
                if element not in self.combinations:
                    self.combinations[element] = np.zeros(self.size, dtype=np.int8)
                self.combinations[element][ind] += 1
        print(f"{Fore.LIGHTBLUE_EX}Amount of combinations: {Fore.RED}{len(self.combinations)}{Fore.RESET}")
        print(
            f"{Fore.LIGHTBLUE_EX}Generate a quantity of instances combinations, {Fore.RED}time elapsed: {time.time() - start_time:.2f}s{Fore.RESET}\n")

    def save_data(self, directory, files=None):
        self.personal_information(directory, files=files)
        self.create_combinations()
        self.create_cmv_vector()

    """
    calculated the chi squared score for each TCR in the data
    """

    def scatter_score(self):
        self.score = {}
        numn = np.count_nonzero(1 - self.cmv)
        nump = np.count_nonzero(self.cmv)
        pos_precent = nump / (numn + nump)
        print("Calculating chi squared score")
        for element, val in tqdm(self.combinations.items()):
            sumrec = np.count_nonzero(val)
            if sumrec < 2:
                self.score[element] = 0
            else:
                sumPos = np.dot(np.sign(val), self.cmv)
                self.score[element] = abs(sumPos - pos_precent * sumrec) * (sumPos - pos_precent * sumrec) / (
                        pos_precent * sumrec)
            if abs(self.score[element]) > 50:
                del self.score[element]

    '''
    Finds the numrec most reactive TCRs, with the largest chi squared score. 
    The actual number or reactive TCRs might be slightly larger that numrec if there are TCRs with exactly the same chi squared score. 
    '''

    def outlier_finder(self, numrec):
        self.scatter_score()
        for element, score in self.score.items():
            self.score[element] = abs(score)
        self.score = dict(sorted(self.score.items(), key=lambda item: item[1], reverse=True))
        element, min_score = list(self.score.items())[numrec]
        print("Reactive TCRs found:")
        for element, score in list(self.score.items()):
            if score >= min_score:
                print(f'{element} score: {score}')
                self.outlier[element] = self.combinations[element]

    """
    Preparing data to learning. Extracting reactive TCRs from the train set and sampling them from each file in the training/validation/test sets
    """

    def prepToLearning(self, test, val, numrec):
        if "train" != self.name:
            print("can\'t use test data as training data")
            return
        self.outlier_finder(numrec=numrec)
        trainData = np.transpose(np.asmatrix([np.sign(row) for row in self.outlier.values()]))
        testData = np.transpose(np.asmatrix([np.sign(test.combinations[key][:])
                                             if key in test.combinations.keys()
                                             else np.zeros(test.size)
                                             for key in self.outlier.keys()]))
        valData = np.transpose(np.asmatrix([np.sign(val.combinations[key][:])
                                            if key in val.combinations.keys()
                                            else np.zeros(val.size)
                                            for key in self.outlier.keys()]))
        return trainData, valData, testData

    '''
    Recieved onehot data of each TCR and output data as a list of indices
    '''

    def from_onehot_to_idxs(self, params):
        idxs_params = []
        for param in params:
            idxs = []
            for idx, value in enumerate(param.tolist()[0]):
                if value != 0:
                    idxs.append(idx + 1)
            if len(idxs) == 0:
                idxs.append(0)
            idxs_params.append(idxs)
        return idxs_params

    '''
    Calculating f1 score
    '''

    def max_f1_score(self, labels, preds):
        f1_scores = []
        for thresh in preds:
            new_preds = np.array(preds)
            new_preds[new_preds < thresh] = 0
            new_preds[new_preds >= thresh] = 1
            f1 = f1_score(labels, new_preds)
            f1_scores.append(f1)
        return max(f1_scores)

    '''
    Testing model
    '''

    def test_model(self, model, test_loader, loss_func, device):
        model.eval()
        preds = []
        losses = []
        labels = []
        for x, y in tqdm(test_loader):
            with torch.no_grad():
                output = model(x.to(device))
                loss = loss_func(output.view(-1, 1), y.to(device).float().view(-1, 1))
                preds.append(output.item())
                losses.append(loss.item())
                labels.append(y.item())
        avg_loss = sum(losses) / len(losses)
        auc = roc_auc_score(labels, preds)
        f1 = self.max_f1_score(labels, preds)
        return auc, f1, avg_loss

    '''
    Running the count model
    '''

    def countModel(self, test, validation, numrec):
        trainParam, valParam, testParam = self.prepToLearning(test, validation, numrec)
        nn_test = self.from_onehot_to_idxs(testParam)
        test_data = RepertoireDataset(nn_test, list(test.cmv))
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
        labels = []
        counts = []
        for x, y in test_loader:
            counts.append(x.size(dim=1))
            labels.append(y.item())
        auc = sklearn.metrics.roc_auc_score(labels, counts)
        f1 = self.max_f1_score(labels, counts)
        print('test F1 score', f1)
        print('test AUC', auc)
        results = {'auc': auc, 'f1 score': f1}
        return results


'''
main function
'''


def run(config_dict, param_dict):
    all_results = []
    path = config_dict['test_data_file_path']
    files = list(Path(path).glob('*'))
    test = Repertoires("test", 1000)
    print("Loading Test...")
    test.save_data(path, files=files)
    ARGS = param_dict
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cv = ARGS['cv']
    auc = 0
    path = config_dict['train_data_file_path']
    samples = config_dict['samples']
    if samples == -1:
        samples = 1000
    for j in range(cv):
        train = Repertoires("train", samples)
        files = list(Path(path).glob('*'))
        N = len(files)
        train_list = files[: int(j * N / cv)] + files[int((j + 1) * N / cv):]
        val_list = files[int(j * N / cv): int((j + 1) * N / cv)]
        validation = Repertoires("val", len(val_list))
        print("Loading Train...")
        train.save_data(path, files=train_list)
        print("Loading Validation...")
        validation.save_data(path, files=val_list)
        results = train.countModel(test, validation, ARGS['numrec'])
        all_results.append(results)
        auc += results['auc'] / cv
    print('Final AUC', auc)
    return all_results


if __name__ == "__main__":
    main()
