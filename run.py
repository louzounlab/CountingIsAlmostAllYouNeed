import argparse
import csv
import json
import os
from datetime import datetime

from Count import count
from attTCR import attTCR
from gTCR import gTCR


def set_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Config file path", type=str)
    parser.add_argument("--params", help="Hyper parameters file path",
                        type=str)
    parser.add_argument("--model", help="The model you want to run",
                        type=str)
    return parser


if __name__ == '__main__':
    parser = set_arguments()
    args = parser.parse_args()
    config_file_path = args.config
    params_file_path = args.params
    model = args.model
    config_dict = json.load(open(config_file_path, 'r'))
    params_dict = json.load(open(params_file_path, 'r'))
    print(model)
    if model == 'attTCR':
        all_results = attTCR.run(config_dict, params_dict)
    if model == 'count':
        all_results = count.run(config_dict, params_dict)
    if model == 'gTCR':
        all_results = gTCR.run(config_dict, params_dict)
        results_dict = []
        for i in range(len(all_results[0])):
            results_dict.append(
                {'Test AUC': all_results[2][i], 'Train AUC': all_results[0][i], 'Val AUC': all_results[1][i],
                 'Alpha': all_results[4][i]})
        all_results = results_dict
    now = datetime.now()
    dt_string = now.strftime("%d.%m.%Y-%H:%M:%S")
    dir_name = os.path.join('Runs', f'{dt_string}_{model}')
    if not os.path.exists('Runs'):
        os.mkdir('Runs')
    os.mkdir(dir_name)
    dict_writer = csv.DictWriter(open(os.path.join(dir_name, 'results.csv'), 'w'),
                                 fieldnames=list(all_results[0].keys()))
    dict_writer.writeheader()
    dict_writer.writerows(all_results)
    with open(os.path.join(dir_name, 'params.csv'), 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['param', 'value'])
        for key, value in params_dict.items():
            writer.writerow([key, value])
