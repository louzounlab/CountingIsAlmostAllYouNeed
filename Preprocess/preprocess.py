import os
import re

import pandas as pd
from tqdm import tqdm


def get_common_tcrs(source, thresh):
    tcr_counting_dict = {}
    for directory, _, files in os.walk(source):
        for file in tqdm(files):
            df = pd.read_csv(os.path.join(source, file), usecols=['amino_acid'], delimiter='\t')
            try:
                target = re.search('(?<=Cytomegalovirus\s).{1}', df.sample_tags[0])[0]
            except:
                continue
            df = df.dropna()
            tcrs = df['amino_acid'].unique().tolist()
            for tcr in tcrs:
                if 'X' not in tcr and tcr != '':
                    if tcr in tcr_counting_dict:
                        tcr_counting_dict[tcr] += 1
                    else:
                        tcr_counting_dict[tcr] = 1
    common_tcrs = []
    for tcr, count in tcr_counting_dict.items():
        if count >= thresh:
            common_tcrs.append(tcr)
    print(f"Total number of common tcrs:", len(common_tcrs))
    return common_tcrs


def filter_data(source, dest, common_tcrs):
    if not os.path.exists(dest):
        os.makedirs(dest)
    switch = {'-': 'negative', '+': 'positive'}
    for directory, _, files in os.walk(source):
        for file in tqdm(files):
            df = pd.read_csv(os.path.join(source, file),
                             usecols=['amino_acid', 'frequency', 'v_family', 'j_family', 'sample_tags'], delimiter='\t')
            try:
                target = re.search('(?<=Cytomegalovirus\s).{1}', df.sample_tags[0])[0]
            except:
                continue
            df = df.dropna()
            df = df[df['amino_acid'].isin(common_tcrs)]
            cols = ['amino_acid', 'v_family', 'j_family']
            df['target'] = target
            df['combined'] = df[cols].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
            df.drop('sample_tags', axis=1, inplace=True)
            file = file.replace('_MC1', '')
            output_file = file[:-4] + '_' + switch[target] + '.csv'
            df.to_csv(os.path.join(dest, output_file), index=False)


if __name__ == '__main__':
    info = 'cmv_emerson_2017.tsv'
    source = 'Dataset'
    destination = 'Filtered Data'
    info_df = pd.read_csv(info, delimiter='\t')
    common_tcrs = get_common_tcrs(source, 7)
    filter_data(source, destination, common_tcrs)
