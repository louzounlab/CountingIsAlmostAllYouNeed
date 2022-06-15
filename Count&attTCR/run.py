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
from Datasets import RepertoireDataset
from Models import attTCR
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
Collects the arguments entered by the user
"""


def get_args():
    global ARGS
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', '--data', required=True, type=str, help='The Data Directory')
    parser.add_argument('-ae', '--ae', required=True, type=str,
                        help='The autoencoder dict from a pretrained autoemncoder')
    parser.add_argument('-epochs', '--epochs', default=20, type=int,
                        help='Number of epochs the model is trained. (default: 20)')
    parser.add_argument('-lr', '--lr', default=1e-4, type=float,
                        help='The learning rate of the optimizer. (default: 1e-4)')
    parser.add_argument('-wd', '--wd', default=4e-4, type=float,
                        help='The weight decay of the optimizer. (default: 4e-4)')
    parser.add_argument('-hl', '--hl', default=30, type=int, help='The hidden layer size. (default: 30)')
    parser.add_argument('-cv', '--cv', default=9, type=int, help='The number of cross validations. (default: 9)')
    parser.add_argument('-model', '--model', default="attTCR", type=str,
                        help='The model type. Can be either "count" or "attTCR". (default: attTCR)')
    parser.add_argument('-numrec', '--numrec', default=125, type=int,
                        help='Number of reactive TCRs to sample. (default: 125)')
    parser.add_argument('-sample', '--sample', default=684, type=int,
                        help='Number of TCRs to sample in the training set. the actual number sampled is the minimum between the number you provided and the actual number of repertoires available. (default: 684)')
    parser.add_argument('-device', '--device', default='cuda:0', type=str,
                        help='The device the model is trained on. (default: cuda:0)')
    args = parser.parse_args()
    ARGS['data'] = args.data
    ARGS['ae'] = args.ae
    ARGS['epochs'] = args.epochs
    ARGS['lr'] = args.lr
    ARGS['wd'] = args.wd
    ARGS['hl'] = args.hl
    ARGS['model'] = args.model
    ARGS['numrec'] = args.numrec
    ARGS['sample'] = args.sample
    ARGS['cv'] = args.cv
    ARGS['device'] = args.device
    now = datetime.now()
    dt_string = now.strftime("%d.%m.%Y-%H:%M:%S")
    dir_name = os.path.join('Runs', dt_string)
    if not os.path.exists('Runs'):
        os.mkdir('Runs')
    os.mkdir(dir_name)
    ARGS['dir_name'] = dir_name


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
    Training and Evaluating model
    '''

    def train_model(self, model, train_loader, val_loader, optimizer, loss_func, epochs, device, test_loader=None):
        max_AUC = 0
        best_model = None
        train_losses = []
        val_losses = []
        train_aucs = []
        val_aucs = []
        test_aucs = []
        for epoch in range(epochs):
            model.train()
            print('epoch ', epoch + 1)
            losses = []
            preds = []
            labels = []
            print("Training")
            for x, y in tqdm(train_loader):
                output = model(x.to(device))
                loss = loss_func(output.view(-1, 1), y.to(device).float().view(-1, 1))
                losses.append(loss.item())
                preds.append(output.item())
                labels.append(y.item())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            train_losses.append(sum(losses) / len(losses))
            train_aucs.append(roc_auc_score(labels, preds))
            print(f'Train - loss: {sum(losses) / len(losses)} AUC: {roc_auc_score(labels, preds)}')
            model.eval()
            losses = []
            labels = []
            preds = []
            print('Evaluating')
            for x, y in tqdm(val_loader):
                with torch.no_grad():
                    output = model(x.to(device))
                    loss = loss_func(output.view(-1, 1), y.to(device).float().view(-1, 1))
                    losses.append(loss.item())
                    labels.append(y.item())
                    preds.append(output.item())

            val_loss = sum(losses) / len(losses)
            val_auc = roc_auc_score(labels, preds)
            val_losses.append(val_loss)
            val_aucs.append(val_auc)
            print(f"Validation - Loss: {val_loss} AUC:{val_auc}")
            if val_auc > max_AUC:
                max_AUC = val_auc
                best_model = copy.deepcopy(model)
                print("new best")
            if test_loader is not None:
                auc, f1, loss = self.test_model(model, test_loader, loss_func, device)
                test_aucs.append(auc)
                print(f'Test - Loss {loss} AUC {auc}')
        return best_model, train_losses, train_aucs, val_losses, val_aucs, test_aucs

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
    Embedding each TCR to an input to the encoder
    '''

    def embed(self, tcr, v_gene, amino_pos_to_num, max_length, v_dict):
        padding = torch.zeros(1, max_length)
        for i in range(len(tcr)):
            amino = tcr[i]
            pair = (amino, i)
            padding[0][i] = amino_pos_to_num[pair]
        combined = torch.cat((padding, v_dict[v_gene]), dim=1)
        return combined

    '''
    Creating a dictionary of the inpot embeddings to all the reactive TCRs
    '''

    def get_emb_dict(self, amino_pos_to_num, max_length, v_dict):
        emb_dict = {}
        for i, tcr in enumerate(self.outlier.keys()):
            details = tcr.split('_')
            emb_dict[i + 1] = self.embed(details[0], details[1], amino_pos_to_num, max_length, v_dict).to(
                ARGS['device'])
        emb_dict[0] = torch.zeros(1, max_length + 1).to(ARGS['device'])
        return emb_dict

    '''
    Running the attTCR model
    '''

    def attTCR(self, test, validation, numrec, ae_dict, device):
        trainParam, valParam, testParam = self.prepToLearning(test, validation, numrec)
        emb_dict = self.get_emb_dict(ae_dict['amino_pos_to_num'], ae_dict['max_len'], ae_dict['v_dict'])
        nn_train = self.from_onehot_to_idxs(trainParam)
        nn_val = self.from_onehot_to_idxs(valParam)
        nn_test = self.from_onehot_to_idxs(testParam)
        train_data = RepertoireDataset(nn_train, list(self.cmv))
        val_data = RepertoireDataset(nn_val, list(validation.cmv))
        test_data = RepertoireDataset(nn_test, list(test.cmv))
        train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=1, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
        model = attTCR(10, ae_dict['max_len'], ae_dict['vgene_dim'], ae_dict['enc_dim'],
                       ae_dict['model_state_dict'], emb_dict, hidden_layer=ARGS['hl']).to(device)
        optimizer = Adam(model.parameters(), lr=ARGS['lr'], weight_decay=ARGS['wd'])
        loss_func = nn.BCELoss()
        print('Training Model...')
        model, train_losses, train_aucs, val_losses, val_aucs, test_aucs = self.train_model(model, train_loader,
                                                                                            val_loader,
                                                                                            optimizer, loss_func,
                                                                                            epochs=ARGS['epochs'],
                                                                                            device=device,
                                                                                            test_loader=test_loader)
        print('Testing Model...')
        auc, f1, loss = self.test_model(model, test_loader, loss_func, device)
        avg_auc = sum(test_aucs[-5:]) / 5
        print('test AUC', auc)
        print('avg last AUC', avg_auc)
        print('max AUC', max(test_aucs))
        print('test loss', loss)
        results = {'auc': auc, 'f1 score': f1, 'avg_las_auc': avg_auc, 'max_auc': max(test_aucs), 'test_loss': loss}
        self.plot(train_losses, train_aucs, val_losses, val_aucs,
                  os.path.join(ARGS['dir_name'], ARGS['model'] + "_test_AUC:" + str(round(auc, 3))))
        return results

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
    Plots figures off loss and AUC per epoch
    '''

    def plot(self, train_loss, train_auc, val_loss, val_auc, name):
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot(train_loss)
        axs[0, 0].grid(color="w")
        axs[0, 0].set_facecolor('xkcd:light gray')
        axs[0, 0].set_title("Train Loss")
        axs[1, 0].plot(train_auc)
        axs[1, 0].grid(color="w")
        axs[1, 0].set_facecolor('xkcd:light gray')
        axs[1, 0].set_title("Train AUC")
        axs[1, 0].sharex(axs[0, 0])
        axs[0, 1].grid(color="w")
        axs[0, 1].set_facecolor('xkcd:light gray')
        axs[0, 1].plot(val_loss)
        axs[0, 1].set_title("Validation Loss")
        axs[1, 1].plot(val_auc)
        axs[1, 1].grid(color="w")
        axs[1, 1].set_facecolor('xkcd:light gray')
        axs[1, 1].set_title("Validation AUC")
        fig.tight_layout()
        plt.savefig(name + ".png")


'''
main function
'''


def main():
    get_args()
    all_results = []
    path = os.path.join(ARGS['data'], 'Test')
    files = list(Path(path).glob('*'))
    test = Repertoires("test", 1000)
    print("Loading Test...")
    test.save_data(os.path.join(ARGS['data'], 'Test'), files=files)
    ae_dict = torch.load(ARGS['ae'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cv = ARGS['cv']
    model = ARGS['model']
    auc = 0
    path = os.path.join(ARGS['data'], 'Train')
    for j in range(cv):
        train = Repertoires("train", ARGS['sample'])
        files = list(Path(path).glob('*'))
        N = len(files)
        train_list = files[: int(j * N / cv)] + files[int((j + 1) * N / cv):]
        val_list = files[int(j * N / cv): int((j + 1) * N / cv)]
        validation = Repertoires("val", len(val_list))
        print("Loading Train...")
        train.save_data(path, files=train_list)
        print("Loading Validation...")
        validation.save_data(path, files=val_list)
        if "attTCR" in model:
            results = train.attTCR(test, validation, ARGS['numrec'], ae_dict, device)
            all_results.append(results)
            auc += results['auc'] / cv
        if "count" in model:
            results = train.countModel(test, validation, ARGS['numrec'], device)
            all_results.append(results)
            auc += results['auc'] / cv
    print('Final AUC', auc)
    dict_writer = csv.DictWriter(open(os.path.join(ARGS['dir_name'], 'results.csv'), 'w'),
                                 fieldnames=list(results.keys()))
    dict_writer.writeheader()
    dict_writer.writerows(all_results)
    with open(os.path.join(ARGS['dir_name'], 'params.csv'), 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['param', 'value'])
        for key, value in ARGS.items():
            writer.writerow([key, value])


if __name__ == "__main__":
    main()
