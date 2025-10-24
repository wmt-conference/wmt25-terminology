import pandas as pd
import os
import json
import time

import nltk
nltk.download('punkt_tab')

from termbasedmetric import TermBasedMetric

import argparse

def run_cycle_track2(src_lang, tgt_lang, mode, filepath):
    '''
    runs evaluation for the 2nd track, for a given source, target language and mode,
        collecting the statistics and the extracted data (terms, alignments, pseudoreferences)

    :param src_lang: source language
    :param tgt_lang: target language
    :param mode: evaluation mode ('proper', 'random', 'noterm')
    :param filepath: path to the directory containing the submitted files

    '''

    total_stats = {}
    file_list = [f for f in os.listdir(filepath) if f.endswith('.tsv') and src_lang+tgt_lang in f and mode in f]

    TBM = TermBasedMetric(src_lang, tgt_lang, 'predefined', 'llm')
    error_files = []
    for file in file_list:
        print(file)
        
        try:
            # using try-except statements because some random/noterm texts were not aligned w.r.t. terminologies
            system_name, year, _, _, _ = file.split('.')
            TBM.load(system_name, mode=mode, year=year, track=2, file_type='tsv')
            TBM.extract_keywords()
            TBM.align(test=False)
            pseudoref_choices = ['first', 'frequent', 'predefined']
            var_dict = {pseudoref_choice: {'micro': 0, 'macro': 0} for pseudoref_choice in pseudoref_choices}
            flat_df = []
            for pseudoref_choice in pseudoref_choices:
                _, _, _, sel = TBM.assign_pseudoreferences(pseudoref_choice)
                micro, flat_micro = TBM.compute_metric('micro')
                macro, _ = TBM.compute_metric('macro')
                if pseudoref_choice == 'first':
                    sel_first = flat_micro.copy(deep=True)
                elif pseudoref_choice == 'frequent':
                    sel_freq = flat_micro.copy(deep=True)
                print(f'pseudoref_choice: {pseudoref_choice}, metric: {micro}/{macro}')
                flat_micro.rename(columns={'pseudoref': f'pseudoref_{pseudoref_choice}'}, inplace=True)
                flat_df.append(flat_micro)
                var_dict[pseudoref_choice]['micro'] = micro
                var_dict[pseudoref_choice]['macro'] = macro
            print((sel_first == sel_freq).value_counts())
            if system_name not in total_stats.keys():
                total_stats[system_name] = {}
            total_stats[system_name][year] = var_dict
            flat_df = pd.concat(flat_df, axis=1)
            flat_df.to_csv(f'pseudorefs/{file}_pseudoref.tsv', sep='\t')
            TBM.bitext_df[['terms', 'src_raw', 'mt_raw', 'src_terms', 'alg_terms', 'norm_tgt_terms']].to_csv(f'processed/{file}_processed.tsv', sep='\t')
        except:
            error_files.append(file)
    return total_stats, error_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--srclang')
    parser.add_argument('-t', '--tgtlang')
    parser.add_argument('-m', '--mode')
#    parser.add_argument('-l', '--track')
    args=parser.parse_args()
    print(args.srclang, args.tgtlang, args.mode)
    a = time.time()
    stats_dict, error_files = run_cycle_track2(args.srclang, args.tgtlang, args.mode, f'data/submissions/track2')
    b = time.time()
    print(f'finished in {b-a} seconds')
    print(error_files)
    with open(f'stats-{args.srclang}{args.tgtlang}.{args.mode}.json', 'w') as fp:
        json.dump(stats_dict, fp)