import copy
import csv
import json
import os
import random
from random import shuffle

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

'''
Parameters
'''

specific_lr = 0.0001
specific_dropout = 0.2
specific_first_layer = 800
specific_second_layer = 1100
weight_v = 1
weight_cdr3 = 0.01
a_lim = 5
b_lim = 10

'''
Autoencoder network
'''


class Model(nn.Module):
    def __init__(self, max_len, embedding_dim, vgenes_dim, v_dict, encoding_dim=30):
        super(Model, self).__init__()
        self.encoding_dim = encoding_dim
        self.max_len = max_len
        self.vgenes_dim = vgenes_dim
        self.embedding_dim = embedding_dim
        self.v_dict = v_dict
        self.vocab_size = max_len * 20 + 2 + vgenes_dim
        self.embedding = nn.Embedding(self.vocab_size, embedding_dim, padding_idx=-1)
        self.encoder = nn.Sequential(
            nn.Linear(self.embedding_dim * (self.max_len + 1), specific_first_layer),
            nn.ELU(),
            nn.Dropout(specific_dropout),
            nn.Linear(specific_first_layer, specific_second_layer),
            nn.ELU(),
            nn.Dropout(specific_dropout)
        )
        self.mu = nn.Linear(specific_second_layer, self.encoding_dim)
        self.log_sigma = nn.Linear(specific_second_layer, self.encoding_dim)
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, specific_second_layer),
            nn.ELU(),
            nn.Dropout(specific_dropout),
            nn.Linear(specific_second_layer, specific_first_layer),
            nn.ELU(),
            nn.Dropout(specific_dropout),
            nn.Linear(specific_first_layer, self.max_len * 21 + self.vgenes_dim)
        )

    def reparameterize(self, x_mu, x_log_sigma):
        std = torch.exp(0.5 * x_log_sigma)
        eps = torch.randn_like(std)
        return x_mu + eps * std

    def forward(self, padded_input):
        x_emb = self.embedding(padded_input.long())
        x_emb = x_emb.view(-1, (self.max_len + 1) * self.embedding_dim)
        x = self.encoder(x_emb)
        x_mu = self.mu(x)
        x_log_sigma = self.log_sigma(x)
        encoded = self.reparameterize(x_mu, x_log_sigma)
        decoded = self.decoder(encoded)
        decoded = decoded.view(-1, 1, self.max_len * 21 + self.vgenes_dim)
        tcr_chain, v_gene = torch.split(decoded, self.max_len * 21, dim=2)
        v_gene = F.softmax(v_gene.view(-1, self.vgenes_dim), dim=1)
        tcr_chain = F.softmax(tcr_chain.view(-1, self.max_len, 21), dim=2)  # check if needed
        output = torch.cat((tcr_chain.view(-1, self.max_len * 21), v_gene), dim=1)
        return output, x_mu, x_log_sigma


def get_beta(a, b, num_epoch):
    if num_epoch % b < a:
        return (num_epoch % b) / a
    else:
        return 1


'''
Calculates loss function
'''


def loss_func(x, y, mu, log_sigma, max_len, encoding_dim, num_epoch):
    tcr_chain, v_gene = torch.split(x, max_len * 21, dim=1)
    y_tcr, y_vgene = torch.split(y, max_len * 21, dim=1)
    MSE_tcr = F.mse_loss(tcr_chain, y_tcr, reduction='sum')
    MSE_vgene = F.mse_loss(v_gene, y_vgene, reduction='sum')
    MSE = weight_cdr3 * MSE_tcr + weight_v * MSE_vgene
    KLD = -0.5 * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp())

    beta = get_beta(a_lim, b_lim, num_epoch)
    return 10 * (1 / 12.26) * MSE + beta * (1 / (encoding_dim * 1.4189)) * KLD, MSE, KLD


'''
Loading all the data from files
'''


def load_all_data(path, samples_count=1000):
    all_data = []
    for directory, subdirectories, files in os.walk(path):
        for file in tqdm(files):
            with open(os.path.join(directory, file), mode='r') as infile:
                reader = csv.DictReader(infile, delimiter=',')
                data = [(row['amino_acid'], row['v_family']) for row in reader if
                        row['amino_acid'] != '' and 'OR' not in row['v_family'] and row['v_family'] != '']
                max_tcr, max_vgene = data[0]
                for tcr, v_gene in data:
                    if len(tcr) > len(max_tcr) and str(tcr).find('*') == -1 and str(tcr).find('X') == -1:
                        max_tcr = tcr
                        max_vgene = v_gene
                random.shuffle(data)
                if samples_count == 'all':
                    all_data += [(str(i) + 'X', v_gene) for i, v_gene in data if
                                 str(i).find('*') == -1 and str(i).find('X') == -1]
                else:
                    all_data += [(str(i) + 'X', v_gene) for i, v_gene in
                                 data[:samples_count] + [(max_tcr, max_vgene)]
                                 if str(i).find('*') == -1 and str(i).find('X') == -1]
    return all_data


'''
Getting a dictionary of all the v genes in the dataset
'''


def get_all_vgenes_dict(source):
    v_genes = []
    v_dict = {}
    for directory, subdirectories, files in os.walk(source):
        for file in files:
            reader = csv.DictReader(open(os.path.join(directory, file), 'r'), delimiter=',')
            for line in reader:
                v_gene = line['v_family']
                if v_gene != "" and 'OR' not in v_gene:
                    v_gene = v_gene.split('|')[0]
                    v_genes.append(v_gene)
    v_genes = list(set(v_genes))
    for i in range(len(v_genes)):
        tensor = torch.zeros(1, len(v_genes))
        tensor[0][i] = 1
        v_dict[v_genes[i]] = tensor
    print(f'len of different V genes: {len(v_dict)}')
    return v_dict


'''
Finding the max length of TCR in the data
'''


def find_max_len(tcrs):
    return max([len(tcr) for tcr in tcrs])


'''
Creating onehot vectors of the TCRs
'''


def pad_one_hot(tcr, v_gene, amino_to_ix, amino_pos_to_num, max_length, v_dict, v_dict_for_loss):
    padding = torch.zeros(1, max_length)
    padding_for_loss = torch.zeros(1, max_length * (20 + 1))
    for i in range(len(tcr)):
        amino = tcr[i]
        pair = (amino, i)
        padding[0][i] = amino_pos_to_num[pair]
        padding_for_loss[0][i * (20 + 1) + amino_to_ix[amino]] = 1
    combined = torch.cat((padding, v_dict[v_gene]), dim=1)
    combined_for_loss = torch.cat((padding_for_loss, v_dict_for_loss[v_gene]), dim=1)
    return combined, combined_for_loss


'''
Getting batches for learning
'''


def get_batches(tcrs, amino_to_ix, amino_pos_to_num, batch_size, max_length, vgene_dim, v_dict, v_dict_for_loss):
    # Initialization
    batches = []
    batches_for_loss = []
    index = 0
    # Go over all data
    while index < len(tcrs) // batch_size * batch_size:
        # Get batch sequences and math tags
        batch_tcrs = tcrs[index:index + batch_size]
        # Update index
        index += batch_size
        # Pad the batch sequences
        padded_tcrs = torch.zeros((batch_size, max_length + 1))
        padded_tcrs_for_loss = torch.zeros((batch_size, max_length * (20 + 1) + vgene_dim))
        for i in range(batch_size):
            tcr, v_gene = batch_tcrs[i]
            combined, combined_for_loss = pad_one_hot(tcr, v_gene, amino_to_ix, amino_pos_to_num, max_length, v_dict,
                                                      v_dict_for_loss)
            padded_tcrs[i] = combined
            padded_tcrs_for_loss[i] = combined_for_loss
        # Add batch to list
        batches.append(padded_tcrs)
        batches_for_loss.append(padded_tcrs_for_loss)
    # Return list of all batches
    return batches, batches_for_loss


'''
Training the autoencoder for one epoch
'''


def train_epoch(batches, batches_for_loss, model, loss_function, optimizer, device, n_epoch):
    model.train()
    zip_batches = list(zip(batches, batches_for_loss))
    shuffle(zip_batches)
    batches, batches_for_loss = zip(*zip_batches)
    total_loss = 0
    mse_loss = 0
    kl_loss = 0
    for i in tqdm(range(len(batches))):
        padded_tcrs = batches[i]
        padded_tcrs_for_loss = batches_for_loss[i]
        # Move to GPU
        padded_tcrs = padded_tcrs.to(device)
        padded_tcrs_for_loss = padded_tcrs_for_loss.to(device)
        model.zero_grad()
        pred, mu, log_sigma = model(padded_tcrs)
        # Compute loss
        loss, mse, kl = loss_function(pred, padded_tcrs_for_loss, mu, log_sigma, model.max_len, model.encoding_dim,
                                      n_epoch)
        # Update model weights
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        mse_loss += mse.item()
        kl_loss += kl.item()
    # Return average loss
    print("loss:", total_loss / len(batches))
    print("mse loss:", mse_loss / len(batches))
    print("kl loss:", kl_loss / len(batches))
    return total_loss / len(batches), mse_loss / len(batches), kl_loss / len(batches)


'''
Training the autoencoder model
'''


def train_model(batches, validation_batches, batches_for_loss, validation_batches_for_loss, max_len, encoding_dim,
                epochs, vgene_dim, v_dict, device, embedding_dim, early_stopping=False):
    model = Model(max_len=max_len, embedding_dim=embedding_dim, vgenes_dim=vgene_dim, v_dict=v_dict,
                  encoding_dim=encoding_dim)
    model.to(device)
    loss_function = loss_func
    optimizer = optim.Adam(model.parameters(), lr=specific_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-6)
    train_loss_list = list()
    val_loss_list = list()
    train_mse_list, val_mse_list, train_kl_list, val_kl_list = list(), list(), list(), list()
    min_loss = float('inf')
    counter = 0
    best_model = 'None'
    for n_epoch, epoch in enumerate(range(epochs)):
        print(f'Epoch: {epoch + 1} / {epochs}')
        train_loss, train_mse, train_kl = train_epoch(batches, batches_for_loss, model, loss_function, optimizer,
                                                      device, n_epoch)
        train_loss_list.append(train_loss)
        train_mse_list.append(train_mse)
        train_kl_list.append(train_kl)
        val_loss, val_mse, val_kl = run_validation(model, validation_batches, validation_batches_for_loss,
                                                   loss_function, max_len, vgene_dim, device, n_epoch)
        val_loss_list.append(val_loss)
        val_mse_list.append(val_mse)
        val_kl_list.append(val_kl)
        print("Val loss:", val_loss)
        if val_loss < min_loss:
            min_loss = val_loss
            counter = 0
            best_model = copy.deepcopy(model)
        elif early_stopping and counter == 20:
            break
        else:
            counter += 1
    num_epochs = epoch + 1
    if best_model == 'None':
        return model, train_loss_list, val_loss_list, num_epochs, train_mse_list, val_mse_list, train_kl_list, val_kl_list
    else:
        return best_model, train_loss_list, val_loss_list, num_epochs, train_mse_list, val_mse_list, train_kl_list, val_kl_list


def read_pred(pred, ix_to_amino):
    batch_tcrs = []
    for tcr in pred:
        c_index = torch.argmax(tcr, dim=1)
        t = ''
        for index in c_index:
            if ix_to_amino[index.item()] == 'X':
                break
            t += ix_to_amino[index.item()]
        batch_tcrs.append(t)
    return batch_tcrs


def count_mistakes(true_tcr, pred_tcr):
    mis = 0
    for i in range(min(len(true_tcr), len(pred_tcr))):
        if not true_tcr[i] == pred_tcr[i]:
            mis += 1
    return mis


def evaluate(save_path, batches, batches_for_loss, model, ix_to_amino, max_len, vgene_dim, device, EPOCHS):
    model.eval()
    print("evaluating")
    zip_batches = list(zip(batches, batches_for_loss))
    shuffle(zip_batches)
    batches, batches_for_loss = zip(*zip_batches)
    acc = 0
    acc_1mis = 0
    acc_2mis = 0
    acc_3mis = 0
    count = 0
    for i in range(len(batches)):
        with torch.no_grad():
            padded_tcrs = batches[i]
            padded_tcrs_for_loss = batches_for_loss[i]
            padded_tcrs_for_loss = padded_tcrs_for_loss.view(-1, 1, max_len * (20 + 1) + vgene_dim)
            tcr_chain, v_gene = torch.split(padded_tcrs_for_loss, max_len * (20 + 1), dim=2)
            true_tcrs = read_pred(tcr_chain.view(-1, max_len, 20 + 1), ix_to_amino)
            # Move to GPU
            padded_tcrs = padded_tcrs.to(device)
            # model.zero_grad()
            pred, mu, log_sigma = model(padded_tcrs)
            pred = pred.view(-1, 1, max_len * (20 + 1) + vgene_dim)
            tcr_chain, v_gene = torch.split(pred, max_len * (20 + 1), dim=2)
            pred_tcrs = read_pred(tcr_chain.view(-1, max_len, 20 + 1), ix_to_amino)
            # print('true:', true_tcrs)
            # print('pred:', pred_tcrs)
            for j in range(len(batches[i])):
                count += 1
                if len(true_tcrs[j]) != len(pred_tcrs[j]):
                    continue
                mis = count_mistakes(true_tcrs[j], pred_tcrs[j])
                if mis == 0:
                    acc += 1
                if mis <= 1:
                    acc_1mis += 1
                if mis <= 2:
                    acc_2mis += 1
                if mis <= 3:
                    acc_3mis += 1
    acc /= count
    acc_1mis /= count
    acc_2mis /= count
    acc_3mis /= count
    print("acc:", acc)
    print("acc_1mis:", acc_1mis)
    print("acc_2mis:", acc_2mis)
    print("acc_3mis:", acc_3mis)

    with open(f'{save_path}/vae_vgene_accuracy.csv', 'w') as csvfile:
        fieldnames = ['Accuracy', '1 Mismatch', '2 Mismatches', '3 Mismatches']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'Accuracy': str(acc), '1 Mismatch': str(acc_1mis),
                         '2 Mismatches': str(acc_2mis), '3 Mismatches': str(acc_3mis)})



def run_validation(model, validation, validation_for_loss, loss_function, max_len, vgene_dim, device, n_epoch):
    model.eval()
    zip_batches = list(zip(validation, validation_for_loss))
    shuffle(zip_batches)
    validation, validation_for_loss = zip(*zip_batches)
    total_loss_val = 0
    total_mse = 0
    total_kl = 0
    for i in range(len(validation)):
        with torch.no_grad():
            padded_tcrs = validation[i]
            padded_tcrs_for_loss = validation_for_loss[i]
            # Move to GPU
            padded_tcrs = padded_tcrs.to(device)
            padded_tcrs_for_loss = padded_tcrs_for_loss.to(device)
            # model.zero_grad()
            pred, mu, log_sigma = model(padded_tcrs)
            # Compute loss
            loss, loss_mse, loss_kl = loss_function(pred, padded_tcrs_for_loss, mu, log_sigma, model.max_len,
                                                    model.encoding_dim, n_epoch)
            total_loss_val += loss.item()
            total_mse += loss_mse.item()
            total_kl += loss_kl.item()
    return total_loss_val / len(validation), total_mse / len(validation), total_kl / len(validation)


def plot_loss(train, val, num_epochs, save_dir, real_epochs):
    epochs = [e for e in range(num_epochs)]
    label1 = 'Train'
    label2 = 'Validation'
    plt.figure(2)
    plt.plot(epochs, train, 'bo', color='mediumaquamarine', label=label1)
    plt.plot(epochs, val, 'b', color='cornflowerblue', label=label2)
    plt.xlabel('Epoch')
    plt.ylabel('Average loss')
    plt.legend()
    plt.savefig(f'{save_dir}/vae_vgene_loss_function.pdf', bbox_inches="tight",
                pad_inches=1)
    plt.close()


def plot_mse(train, val, num_epochs, save_dir, real_epochs):
    epochs = [e for e in range(num_epochs)]
    label1 = 'Train'
    label2 = 'Validation'
    plt.figure(2)
    plt.plot(epochs, train, 'bo', color='mediumaquamarine', label=label1)
    plt.plot(epochs, val, 'b', color='cornflowerblue', label=label2)
    plt.xlabel('Epoch')
    plt.ylabel('Average loss')
    plt.legend()
    plt.savefig(f'{save_dir}/vae_vgene_mse_loss_function.pdf', bbox_inches="tight",
                pad_inches=1)
    plt.close()


def plot_kl(train, val, num_epochs, save_dir, real_epochs):
    epochs = [e for e in range(num_epochs)]
    label1 = 'Train'
    label2 = 'Validation'
    plt.figure(2)
    plt.plot(epochs, train, 'bo', color='mediumaquamarine', label=label1)
    plt.plot(epochs, val, 'b', color='cornflowerblue', label=label2)
    plt.xlabel('Epoch')
    plt.ylabel('Average loss')
    plt.legend()
    plt.savefig(f'{save_dir}/vae_vgene_kl_loss_function.pdf', bbox_inches="tight",
                pad_inches=1)
    plt.close()


def create_dict_for_amino_and_position(amino_acids, max_len, vgene_dict_):
    amino_pos_to_num = dict()
    count = 1
    for amino in amino_acids:
        for pos in range(max_len):
            pair = (amino, pos)
            amino_pos_to_num[pair] = count
            count += 1
    amino = 'X'
    for pos in range(max_len):
        pair = (amino, pos)
        amino_pos_to_num[pair] = count
    count += 1
    dict_for_v = dict()
    for key in vgene_dict_:
        dict_for_v[key] = torch.tensor([[count]])
        count += 1
    return amino_pos_to_num, dict_for_v


def main():
    with open('parameters.json') as f:
        parameters = json.load(f)

    root = parameters["root"]
    save_path = 'Results'

    embedding_dim = 10

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    save_dir = save_path

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    EPOCHS = parameters["EPOCHS"]  # number of epochs for each model
    ENCODING_DIM = parameters["ENCODING_DIM"]  # number of dimensions in the embedded space
    SAMPLES_COUNT = parameters["SAMPLES_COUNT"]  # How many TCRs to take from each file in the training set

    amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
    amino_to_ix = {amino: index for index, amino in enumerate(amino_acids + ['X'])}
    ix_to_amino = {index: amino for index, amino in enumerate(amino_acids + ['X'])}

    batch_size = 50
    print('loading data')
    tcrs = load_all_data(root, SAMPLES_COUNT)
    print('finished loading')
    groups = [tcr[0] for tcr in tcrs]
    gss = GroupShuffleSplit(n_splits=1, train_size=.8)
    for train_idx, test_idx in gss.split(tcrs, tcrs, groups):
        train = [tcrs[i] for i in train_idx]
        test = [tcrs[i] for i in test_idx]
    groups = [tcr[0] for tcr in train]
    gss = GroupShuffleSplit(n_splits=1, train_size=.85)
    for train_idx, test_idx in gss.split(train, train, groups):
        validation = [train[i] for i in test_idx]
        train = [train[i] for i in train_idx]
    max_len = find_max_len([tcr for tcr, vgene in tcrs])
    print('max_len:', max_len)
    vgene_dict_for_loss = get_all_vgenes_dict(root)
    amino_pos_to_num, vgene_dict = create_dict_for_amino_and_position(amino_acids, max_len, vgene_dict_for_loss)
    print('Creating Batches... This might take a few minutes')
    train_batches, train_batches_for_loss = get_batches(train, amino_to_ix, amino_pos_to_num, batch_size, max_len,
                                                        len(vgene_dict), vgene_dict, vgene_dict_for_loss)
    test_batches, test_batches_for_loss = get_batches(test, amino_to_ix, amino_pos_to_num, batch_size, max_len,
                                                      len(vgene_dict), vgene_dict, vgene_dict_for_loss)
    validation_batches, validation_batches_for_loss = get_batches(validation, amino_to_ix, amino_pos_to_num, batch_size,
                                                                  max_len, len(vgene_dict), vgene_dict,
                                                                  vgene_dict_for_loss)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    encoding_dim = ENCODING_DIM
    model, train_loss_list, val_loss_list, num_epochs, train_mse_list, val_mse_list, train_kl_list, val_kl_list = train_model(
        train_batches, validation_batches, train_batches_for_loss, validation_batches_for_loss, max_len,
        encoding_dim=encoding_dim, epochs=EPOCHS, device=device, embedding_dim=embedding_dim, vgene_dim=len(vgene_dict),
        v_dict=vgene_dict, early_stopping=True)
    plot_loss(train_loss_list, val_loss_list, num_epochs, save_dir, EPOCHS)
    plot_mse(train_mse_list, val_mse_list, num_epochs, save_dir, EPOCHS)
    plot_kl(train_kl_list, val_kl_list, num_epochs, save_dir, EPOCHS)
    evaluate(save_dir, test_batches, test_batches_for_loss, model, ix_to_amino, max_len, len(vgene_dict), device,
             EPOCHS)
    torch.save({
        'amino_to_ix': amino_to_ix,
        'ix_to_amino': ix_to_amino,
        'batch_size': batch_size,
        'v_dict': vgene_dict,
        'v_dict_for_loss': vgene_dict_for_loss,
        'amino_pos_to_num': amino_pos_to_num,
        'max_len': max_len,
        'input_dim': 20 + 1,
        'vgene_dim': len(vgene_dict),
        'enc_dim': encoding_dim,
        'model_state_dict': model.state_dict(),
    }, f'{save_dir}/vae.pt')


if __name__ == '__main__':
    main()
