import sys
import argparse
import pandas as pd
from datetime import datetime
from sentence_transformers import SentenceTransformer
from polyfuzz import PolyFuzz
from polyfuzz.models  import Embeddings
from flair.embeddings import TransformerWordEmbeddings
from flair.embeddings import SentenceTransformerDocumentEmbeddings
import os
import json
import re
import stanza
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt

class DocPreprocessor:

    def __init__(self):
        '''
        initializing the packages necessary for sentence/paragraph alignment (LaBSE, PolyFuzz), and lemmatization (stanza)
        LaBSE implementation was taken from https://github.com/aswinpradeep/Bilingual-Sentence-Aligner/tree/main
        '''

        self.embeddings = SentenceTransformerDocumentEmbeddings('LaBSE')
        self.LaBSE = Embeddings(self.embeddings, min_similarity=0, model_id="LaBSE")
        self.model = PolyFuzz([self.LaBSE])
        self.stanza_en = stanza.Pipeline('en', processors='tokenize,lemma', lemma_pretagged=True, tokenize_pretokenized=False)


    def load(self, filename, filepath='data/submissions/track2/'):
        """
        Loads and processes JSONL data from a specified file.
        Additionally, it derives configuration details by interpreting the
        given filename.

        :param filename (str): The name of the file to load the JSONL data from.
        :param filepath (str): Directory path where the file is located. Defaults to 'data/submissions/track2/'.

        :return: self.config (dict): Configuration derived from the filename.
        :return: self.data (List[dict]): List containing parsed JSON data from the file.
        """
        self.filename, self.filepath = filename, filepath
        self.config = self._name_unwrapper(filename)
        with open(f'{filepath}{filename}', 'r') as f:
            data = [json.loads(s) for s in f.readlines()]

        self.data = data

    def split(self, similarity_threshold=0.4, separator='\n\n'):
        """
        Splits and aligns paragraphs from the dataset on paragraph/sentence level.

        First, splits paragraph by a delimiter ('naive'). If the number of paragraphs is different in source and target, initiates
        LaBSE-based search of aligned paragraphs in target texts. Additionally, calculates paragraph similarity scores
        (to possibly filter out paragraphs that are most likely not aligned).

        :param similarity_threshold (float): The threshold value for sentence similarity score to determine whether
                                          paragraphs should be passed for multi-to-multi alignment. Defaults to 0.4.
        :param separator (str): The delimiter used to separate paragraphs in the dataset. Defaults to '\n\n'.

        :return: self.df: A pandas DataFrame containing resulting data: aligned paragraphs,
            alignment algorithm (naive/labse) and similarity scores.

        """
        df_data = []
        sent_counter = 0
        for par_idx, datapoint in enumerate(self.data):
            src_paragraphs, tgt_paragraphs = self._paragraph_aligner(datapoint,False, sep=separator)
            sent_counter += len(src_paragraphs)
            if len(src_paragraphs) == len(tgt_paragraphs):

                alignment = 'naive'
                for sent_idx, (src, tgt) in enumerate(zip(src_paragraphs, tgt_paragraphs)):
                    score = self._one_one_aligner(src, tgt)
                    df_data.append([par_idx, sent_idx, alignment, src, tgt, score])
                    # sent_counter += 1
            else:
                #print('problematic paragraphs')
                #print(len(src_paragraphs), len(tgt_paragraphs))
                alignment = 'labse'
                for sent_idx, (src, tgt) in enumerate(zip(src_paragraphs, tgt_paragraphs)):
                #    print(f'{sent_idx} processing paragraph...')
                    score = self._one_one_aligner(src, tgt)
                    if score < similarity_threshold:
                #        print('lower than threshold, passing to multi-multi aligner')
                        src_left, tgt_left = src_paragraphs[sent_idx:], tgt_paragraphs[sent_idx:]
                #        print(f'src_left: {src_left}, tgt_left: {tgt_left}')
                        break
                    else:
                        df_data.append([par_idx, sent_idx, 'naive', src, tgt, score])
                        # sent_counter += 1
                        src_left, tgt_left = None, None
                if src_left is not None and tgt_left is not None:
                    aligned_triplets = self._many_to_many_aligner(src_left, tgt_left)
                #    print(f'len(aligned_triplets): {len(aligned_triplets)}')
                    for line in aligned_triplets:
                        s, t, score = line
                        df_data.append([par_idx, -1, alignment, s, t, score])

        self.df = pd.DataFrame(df_data, columns=['paragraph', 'sentence', 'alignment', self.config['src_lang'], self.config['tgt_lang'], 'score'])
        #print(f'total sentences: {sent_counter}')
        return self.df

    def retrieve_terms(self, clear_1tomany=False, local_proper_terms=False, random_terms=False):

        """
        Retrieves terms from the provided global (document-level) dictionary, and assigns them to the respective pargraphs.
            Optionally filters out one-to-many relationships.

        :param random_terms : bool, optional. TODO: delete
        :param clear_1tomany : bool, default False. If True, removes one-to-many dictionary items, ensuring only singular direct mappings.
        local_proper_terms : bool, default False. Whether to use terms specified within the same document as terminology (or to use the 'proper' submission).

        :returns: 'terms' column in self.df: contains dictionaries of terms and their definitions (always 1-to-1) in the format {term: [definition]}.
        """
        self.total_term_dict = self._retrieve_total_term_dict(local_proper_terms=local_proper_terms)
        self.norm2term_dict = self._create_norm2term_dict()
        if self.config['src_lang'] == 'en':
            self.df['norm_en'] = self.df[self.config['src_lang']].apply(lambda x: self._normalize_en_paragraph(x))
            self.df['terms'] = self.df['norm_en'].apply(lambda x: self._find_terms_in_paragraph(x))
        elif self.config['src_lang'] == 'zh':
            self.df['terms'] = self.df[self.config['src_lang']].apply(lambda x: self._find_terms_in_paragraph(x))
        if clear_1tomany:
            self.df['terms'] = self.df['terms'].apply(lambda x: self._clear_1tomany(x))
            #self.df = self.df[len(self.df['terms']) == 1]

        #    print(f'self.df.shape after trimming: {self.df.shape}')
        return self.df

    def visualize_scores(self):
        '''
        auxiliary function for visualizing the scores of the alignment algorithm for each paragraph given a document.
        '''
        colors = ['blue' if a == 'naive' else 'red' for a in self.df['alignment']]
        plt.scatter(range(self.df.shape[0]), self.df['score'], c=colors)
        plt.title(self.filename)
        plt.show()
        return

    def stats(self):
        """
        Compute and display basic statistics of the 'score' column in the dataframe.

        :return: prints the mean, standard deviation, and number of cases with a score below 0.5.
        """
        mean, std = self.df['score'].mean(), self.df['score'].std()
        less_05 = self.df[self.df['score'] < 0.5].shape[0]
        print(f'{self.filename}: mean±std: {mean}±{std}; {less_05} cases with score < 0.5')

    def save(self, filepath='data/submissions/track2_aligned/'):
        """
        Saves the filtered DataFrame to a specified file path in TSV format.

        :param filepath: str, default 'data/submissions/track2_aligned/'. The directory where the file will be saved.

        :return: new_filename: str, The name of the saved file.
        :return: generates the file saved in the filepath directory.
        """
        new_filename = self.filename.replace('.jsonl', '.tsv')
        print(f'df shape before saving: {self.df.shape}')
        subset = self.df[self.df['terms'] != {}]
        print(f'df shape after saving: {subset.shape}')
        subset.to_csv(f'{filepath}{new_filename}', index=None, sep='\t')
        return new_filename + ' saved'

    def _name_unwrapper(self, filename):
        """
        transforms the file name into the dictionary of file configuration.

        :param filename (str): The delimited filename to be parsed
            in the format "<system>.<year>.<langpair>.<mode>.jsonl" (example: 'stitch.2021.enzh.noterm.jsonl').

        :return dict: A config dictionary containing the extracted information:
                - 'system': The system name (str).
                - 'year': The year (str).
                - 'src_lang': The source language code (str, 2 characters).
                - 'tgt_lang': The target language code (str, 2 characters).
                - 'mode': The terminology mode (str).
        """

        system, year, langpair, mode, _ = filename.split('.')
        src_lang, tgt_lang = langpair[:2], langpair[2:]
        return {'system': system, 'year': year, 'src_lang': src_lang, 'tgt_lang': tgt_lang, 'mode': mode}

    def _paragraph_aligner(self, datapoint, enforce_correct, sep='\n\n'):
        """
        Aligns paragraphs from source and target texts in a given datapoint based on a specified separator.

        :param datapoint (dict): A dictionary containing source and target texts.
        :param enforce_correct (bool): A flag indicating whether to enforce equal paragraph counts between source and target texts (used at dev stage).
        :param sep (str): A separator used for splitting paragraphs in the target text. Defaults to '\n\n'.

        :return: (tuple): A tuple containing two lists: paragraphs from the source text and paragraphs from the target text.
        Raises:
            AssertionError: If 'enforce_correct' is True and the source and target texts have different
                paragraph counts, an AssertionError is raised.
        """
        src_text, tgt_text = datapoint[self.config['src_lang']], datapoint[self.config['tgt_lang']]

        src_paragraphs = src_text.split('\n\n')
        tgt_paragraphs = tgt_text.split(sep)

        if enforce_correct:
            assert len(src_paragraphs) == len(tgt_paragraphs)

        return src_paragraphs, tgt_paragraphs

#    def _look_error_datapoint(self, datapoint): #, src, tgt
#        '''
#
#        '''
#        src_paragraphs, tgt_paragraphs = self._paragraph_aligner(datapoint, enforce_correct=False)
#        len_tgt = len(tgt_paragraphs)
#        src_paragraphs = src_paragraphs + ['---' for _ in range(len_tgt - len(src_paragraphs))]
#        df = pd.DataFrame({'src': src_paragraphs, 'mt': tgt_paragraphs})
#        return df

    def _many_to_many_aligner(self, src_paragraphs, tgt_paragraphs):
        """
        Aligns multiple source paragraphs to multiple target paragraphs based on a LaBSE model.
        LaBSE implementation was taken from https://github.com/aswinpradeep/Bilingual-Sentence-Aligner/tree/main

        :param src_paragraphs: List[Str], The source paragraphs to align.
        :param tgt_paragraphs: List[Str], The target paragraphs to align.

        :return: Pandas.DataFrame containing the alignment results and LaBSE similarity scores.
        """
        output = self.model.match(src_paragraphs, tgt_paragraphs)
        dfx = self.model.get_matches()
        return dfx.values.tolist()

    def _one_one_aligner(self, src_sent, tgt_sent):
        """
        computes the similarity score of two pre-aligned sentences using LaBSE.

        :param src_sent (str): The source sentence to align.
        :param tgt_sent (str): The target sentence to align.

        :return: float: The similarity score between the source and target sentences.
        """
        output = self.model.match([src_sent], [tgt_sent])
        score = output.matches['LaBSE']['Similarity'][0]
        return score

    def _retrieve_total_term_dict(self, random_terms=False, local_proper_terms=False):
        """
        Retrieve the global terminology dictionary based on the current mode and specified conditions (whether to use random or proper terms).


        :param random_terms (bool): whether random terms should be used (for random mode). Default is False.
        :param local_proper_terms (bool): whether proper terms should be used from the same file. Default is False.

        :return: A global term dictionary.
        """
        if self.config['mode'] == 'proper':
            # use the proper terms from the same (proper) submission
            total_term_dict = self.data[0]['terms']

        else:
            # either use terms from outside or use non-proper terms
            if self.config['mode'] == 'random' and random_terms == True:
                # use the random terms for the random mode
                total_term_dict = self.data[0]['terms']

            elif local_proper_terms:
                # use proper terms from the same submission
                # TODO: maybe excessive given first if statement
                total_term_dict = self.data[0]['terms']

            else:
                # use the proper terms from the proper submission
                proper_doc_name = re.sub(r'\.(noterm|random)', '.proper', self.filename)
                with open(f'{self.filepath}{proper_doc_name}', 'r') as f:
                    proper_data = [json.loads(s) for s in f.readlines()]
                total_term_dict = proper_data[0]['terms']

        return total_term_dict

    def _normalize_en_paragraph(self, paragraph):
        """
        Normalize an English paragraph using stanza text processing tools.

        :param paragraph: str, The input English paragraph to be normalized.

        :return: norm_paragraph: str, The normalized English paragraph.
        """
        doc = self.stanza_en(paragraph)

        lemmas = [word.lemma.lower() for sent in doc.sentences for word in sent.words]
        norm_paragraph = ' '.join(lemmas)
        return norm_paragraph

    def _create_norm2term_dict(self):
        """
        Creates a dictionary mapping normalized terms to original terms.

        :return: norm2term_dict, dictionary containing the mapping from normalized terms to their original forms.
        """
        norm2term_dict = {}
        for term in self.total_term_dict.keys():
            norm_term = self._normalize_en_paragraph(term)
            norm2term_dict[norm_term] = term
        return norm2term_dict

    def _find_terms_in_paragraph(self, paragraph):
        """
        Finds the terms from a global dictionary in a given source paragraph (based on shallow matching).

        :param paragraph (str): The paragraph from which terms need to be extracted.

        :return: term_subset: a subset of a global dictionary containing the terms found in the paragraph.
        """
        term_subset = {}
        for term in self.norm2term_dict.keys():
            if self.config['src_lang'] == 'en':
                try:
                    if re.search(rf'(^|\W+){term}(\W+|$)', paragraph) is not None: # re.escape(rf'(^|\W+){term}(\W+|$)')
                        real_term = self.norm2term_dict[term]
                        term_subset[term] = self.total_term_dict[real_term]
                except:
                    pass
            else:
                #search_term = term
                if term in paragraph:
                    real_term = self.norm2term_dict[term]
                    term_subset[term] = self.total_term_dict[real_term]

        return term_subset

    def _clear_1tomany(self, term_dict):
        """
        Removes entries with a one-to-many mapping from the given dictionary and normalizes the values.

        :param term_dict (dict): A (filtered) dictionary where keys are terms and values are lists of mapped terms.

        :return clear_dict: A dictionary of format {str: [str]} with single entries in the lists.
        """
        clear_dict = {}
        for term in term_dict.keys():
            if len(term_dict[term]) > 1:
                # if there are multiple entries: firstly normalize
                norm_set = set([self._normalize_en_paragraph(w) for w in term_dict[term]])
                if len(norm_set) == 1:
                    # if the values became the same: add to the dictionary
                    clear_dict[term] = [list(norm_set)[0]]
                # else: exclude the entry from the dictionary
            else:
                clear_dict[term] = term_dict[term]
        return clear_dict