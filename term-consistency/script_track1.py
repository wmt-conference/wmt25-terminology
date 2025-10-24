import pandas as pd
import os
import json
import time

import nltk
nltk.download('punkt_tab')

from termbasedmetric import TermBasedMetric

import argparse

def run_cycle(src_lang, tgt_lang, mode, filepath):
    '''
    runs evaluation for the 1st track, for a given source, target language and mode,
        collecting the statistics and the extracted data (terms, alignments, pseudoreferences)

    :param src_lang: source language
    :param tgt_lang: target language
    :param mode: evaluation mode ('proper', 'random', 'noterm')
    :param filepath: path to the directory containing the submitted files
    '''

    # collect all statistics in the format: {system: {pseudoref_choice: {macro: score, micro: score}}}
    total_stats = {}
    # read all submitted files
    file_list = [f for f in os.listdir(filepath) if f.endswith('.jsonl') and src_lang+tgt_lang in f and mode in f]

    # initializing term based metric
    TBM = TermBasedMetric(src_lang, tgt_lang, 'predefined', 'llm')

    # catching files with raised errors (deprecated)
    error_files = []
    for file in file_list:
        print(file)
        
        system_name = file.split('.')[0]
        # loading the submitted file
        TBM.load(system_name, mode=mode)
        # extracting keywords
        TBM.extract_keywords()
        # aligning source terms to translated sentences
        TBM.align(test=False)
        pseudoref_choices = ['first', 'frequent', 'predefined']
        var_dict = {pseudoref_choice: {'micro': 0, 'macro': 0} for pseudoref_choice in pseudoref_choices}
        flat_df = []
        for pseudoref_choice in pseudoref_choices:
            # assigning pseudoreferences for all three pseudoreference types (first, frequent, predefined)
            _, _, _, sel = TBM.assign_pseudoreferences(pseudoref_choice)
            # computing the micro and macro averaged accuracies for pseudoreference and aligned term overlaps
            micro, flat_micro = TBM.compute_metric('micro')
            macro, _ = TBM.compute_metric('macro')
            if pseudoref_choice == 'first':
                sel_first = flat_micro.copy(deep=True)
            elif pseudoref_choice == 'frequent':
                sel_freq = flat_micro.copy(deep=True)
            flat_micro.rename(columns={'pseudoref': f'pseudoref_{pseudoref_choice}'}, inplace=True)
            flat_df.append(flat_micro)
            var_dict[pseudoref_choice]['micro'] = micro
            var_dict[pseudoref_choice]['macro'] = macro
        # adding all statistics to total dict
        total_stats[system_name] = var_dict
        flat_df = pd.concat(flat_df, axis=1)
        # saving the file with pseudoreferences (and aligned term translations)
        flat_df.to_csv(f'pseudorefs/{file}_pseudoref.tsv', sep='\t')
        # saving the file with all initial data (sentences, terms, extracted translations)
        TBM.bitext_df[['terms', 'src_raw', 'mt_raw', 'src_terms', 'alg_terms', 'norm_tgt_terms']].to_csv(f'processed/{file}_processed.tsv', sep='\t')

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
    stats_dict, error_files = run_cycle(args.srclang, args.tgtlang, args.mode, f'data/submissions/track1')
    b = time.time()
    print(f'finished in {b-a} seconds')
    print(error_files)
    with open(f'stats-{args.srclang}{args.tgtlang}.{args.mode}.json', 'w') as fp:
        json.dump(stats_dict, fp)