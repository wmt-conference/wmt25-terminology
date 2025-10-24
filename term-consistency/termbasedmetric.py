import pandas as pd
import json
from nltk.tokenize import word_tokenize
from openai import OpenAI
import ast
from transformers import AutoModel, AutoTokenizer
import itertools
import torch
import stanza
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score # TODO: generalize to input other metrics
from sklearn.metrics import confusion_matrix
from collections import defaultdict
import pymorphy3
import jieba


class TermBasedMetric():
    """
    Handles term-based evaluation metrics for machine translation models.

    Core methods:
    1. load: loads the submission file. Tailored to the WMT25 shared task format.
    2. extract_keywords: extracts terms from the submission file. Tailored to the explicitly predefined terminology dictionaries.
    3. align: aligns terms with the mt outputs. Current implementation uses ChatGPT + LaBSE postprocessing as an aligner.
    4. assign_pseudoreference: assigns pseudoreference terms (the terms from which the deviation would be counted) based on the selected strategy.
    5. compute_metric: computes the term-based metric based on the selected strategy (micro/macro accuracy of the aligned terms against the pseudoreference terms).

    Attributes:
        lang_src (str): Source language code (e.g., 'en').
        lang_tgt (str): Target language code (e.g., 'cs').
        stanza_src: A stanza pipeline object for processing the source language.
        stanza_tgt: A stanza pipeline object for processing the target language.
        ru_morph: A pymorphy3 MorphAnalyzer object for Russian text analysis.
        keyword_extractor (str): Specifies the keyword extraction method.
        aligner (str): Specifies the alignment algorithm.
        openai_client: OpenAI client object if aligner is 'llm'.
        aligner_model: Alignment model if aligner is 'llm' and requires models.
        aligner_tokenizer: Tokenizer for the alignment model used with 'llm'.
        failed_dummy (str): Placeholder string for failed alignments.
        pseudoref_dummy (str): Placeholder string for dummy pseudoreferences.
    """

    def __init__(self, src_lang: str, tgt_lang: str, keyword_extractor: str, aligner: str):
        # TODO: consider reallocating keyword-extractor and aligner params to extract_keywords and align methods
        # TODO: more consistent lemmatizers (stanza_src/tgt VS ru_morph for Russian VS jieba for Chinese)
        '''
        initializes the term-based metric instance.

        :param src_lang: Source language code.
        :param tgt_lang: Target language code.
        :param keyword_extractor: name of the keyword extraction principle. {'yake', 'regex', 'predefined'}, default: 'predefined'
        :param aligner: name of the automatic alignment algorithm. {'fastalign', 'awesomealign', 'llm'}, default: 'llm'
        '''
        self.lang_src, self.lang_tgt = src_lang, tgt_lang
        self.stanza_src = stanza.Pipeline(self.lang_src, processors='tokenize,lemma', lemma_pretagged=True, tokenize_pretokenized=False)
        self.stanza_tgt = stanza.Pipeline(self.lang_tgt, processors='tokenize,lemma', lemma_pretagged=True, tokenize_pretokenized=False)
        self.ru_morph = pymorphy3.MorphAnalyzer()
        self.keyword_extractor = keyword_extractor
        self.aligner = aligner
        if self.aligner == 'llm':
            with open('openai-api-key.txt', 'r', encoding='utf-8') as f:
                openai_key = f.read()
            self.openai_client = OpenAI(api_key=openai_key)
            self.aligner_model = AutoModel.from_pretrained("aneuraz/awesome-align-with-co")
            self.aligner_tokenizer = AutoTokenizer.from_pretrained("aneuraz/awesome-align-with-co")
        #self.pseudoreference_mode = pseudoreference_mode
        #self.statistical_metric = statistical_metric

        self.failed_dummy = '▔'
        self.pseudoref_dummy = '▁'


    def load(self, system: str, mode='proper', track=1, year=None,
                              file_path='data/submissions/', file_type='jsonl', return_random=False):

        """
        Loads and processes bitext data from a specified file source based on the given
        configuration, and returns the processed data as a DataFrame.
        Tailored for the WMT25 format, other formats like Moses (|||-separated) are not checked.

        Parameters:
        :param system: str
            Specifies the system name.
        :param mode: str, default 'proper'
            Specifies the terminology mode {proper, random, noterm}.
        :param track: int, default 1
            Specifies the track of the submission (1 or 2).
        :param year: Optional, default None
            Specifies the year corresponding to the data (2nd track only).
        :param file_path: str, default 'data/submissions/'
            Specifies the path to the directory containing the bitext files.
        :param file_type: str, default 'jsonl'
            Specifies the file type to be read; supported types are 'tsv', 'jsonl', or Moses format.
        :param return_random: bool, default False
            Determines whether for the random mode, the random terminology is returned
            (default is retrieving the proper terminology).

        :returns self.bitext_df: pd.DataFrame, containing bitext pairs aligned in columns `src_raw` and `mt_raw`.
        :returns self.file_config: dict, containing the configuration of the submitted file.

        """
        self.file_config = {'system': system, 'mode': mode, 'track': track, 'year': year, 'file_path': file_path, 'file_type': file_type, 'return_random': return_random}

        bitext_filepath = self._name_compiler_wmt25()

        #self.bitext_df_filename = bitext

        if file_type == 'tsv':
            bitext_df = self._open_tsv_file(bitext_filepath)
        elif file_type == 'jsonl':

            bitext_df = self._open_jsonl_file(bitext_filepath)#.head(20)

        else:
            # when it is in the moses format (src_sent ||| tgt_sent):

            with open(bitext_filepath, 'r', encoding='utf-8') as f:
                data = f.read()
            lines = data.split('\n')
            if lines[-1] == '':
                lines = lines[:-1]

            sent_pairs = []
            for line in lines: # TODO: make a list comprehension
                src, tgt = line.split(' ||| ')
                pair = [src, tgt]
                sent_pairs.append(pair)

            bitext_df = pd.DataFrame(sent_pairs, columns=['src_raw', 'mt_raw'])

        self.bitext_df = bitext_df#[['src', 'mt']]

        return self.bitext_df

    def extract_keywords(self, return_random=False):
        '''
        retrieves the list of the keywords either from the predefined file, or automatically.
        Tailored for the WMT25 format (predefined terminology), other methods like regex or YAKE are not checked.

        the keyword_extractor values are:
            - `predefined`: the manually predefined set of words in the
                            format {"src_lang term": "tgt_lang term"} format (WMT terminology shared task-2025)
            - `yake`: TODO
            - `regex`: TODO
        :return: 'src_terms' column for the self.bitext_df
        '''
        if self.keyword_extractor == 'predefined':

            terms, reference_dict = self._retrieve_predefined_terms(return_random=return_random)
            self.bitext_df['src_terms'] = terms
            if self.file_config['track'] == 2:
                self.bitext_df['terms'] = self.bitext_df['terms'].apply(lambda d: {k: v[0] for k, v in d.items()})

        elif self.keyword_extractor == 'yake':
            raise NotImplementedError
        elif self.keyword_extractor == 'regex':
            raise NotImplementedError

        return self.bitext_df

    def align(self, test=False):
        """
        Extracts the translations of the source terms from the target text, and stores them in the alignment dictionary.
        Current implementation is tailored for the WMT25 format (for languages in question, LLMs+post-filtering worked best),
        alternatives (like fast-align or awesome-align) are not implemented.

        IMPORTANT: the LLM method requires the few-shot prompt file specified by 'lang_src-lang_tgt-20.txt' in 'fewshot' folder,
                   otherwise raises IOError!

        :param test: bool  (default True), takes only last 50 rows from the bitext DataFrame for alignment.

        :return: column 'alg_terms' in self.bitext_df, dictionary containing source terms and translations extracted from the target text.
        :return: column 'over_aligned' in self.bitext_df: int (0 or 1), whether the aligned translated term includes excessive words. To be deprecated.

        """

        if test:
            df = self.bitext_df.tail(50)
            self.bitext_df = None
            self.bitext_df = df


        if self.aligner == 'llm':
            # TODO: generalize the fewshot path
            with open(f'fewshot/{self.lang_src}-{self.lang_tgt}-20.txt', 'r', encoding='utf-8') as f:
                fewshots_prompt = f.read()
            self.bitext_df['alg_terms'], self.bitext_df['over_aligned'] = zip(*self.bitext_df.apply(lambda row: self._llm_align_one_segment(row.src_raw, row.src_terms, row.mt_raw, fewshots_prompt, row.terms), axis=1))

    def assign_pseudoreferences(self, pseudoreference_mode):
        """
        Assign pseudoreferences (expected correct terms against which the real occurrences will be counted)
        to source terms based on a specified pseudoreference mode and create data structures mapping the source terms to the pseudoreferences.

        Parameters:
        :param pseudoreference_mode: str, three modes:
         - 'predefined': uses the predefined translations as pseudoreferences,
         - 'frequent': uses the most frequent translation of a term as pseudoreference,
         - 'first': uses the first translations of each term as pseudoreference,

         :return: 'norm_tgt_terms' column in self.bitext_df, list of normalized translated terms extracted for each segment.
         :return: 'pseudoref_terms' in self.bitext_df, list pseudoreferences for each segment.
         :return: df_first, df_freq, df_predef: 3 DataFrames, containing mappings of source terms to the pseudoreferences based on the specified mode.
                  (TODO: unnecessary, consider deleting)
         :return: 'pseudoreference_dict', dictionary containing ALL source terms and corresponding pseudoreferences.
        """


        self.bitext_df['norm_tgt_terms'] = self.bitext_df.apply(lambda row: self._normalize_pseudoreference_candidates(row.src_terms, row.alg_terms), axis=1)

        src_list_of_lists = self.bitext_df['src_terms'].tolist()
        tgt_list_of_lists = self.bitext_df['norm_tgt_terms'].tolist()

        # creating the pseudoreference tables for extracting the pseudoreferences
        df_first, df_freq, df_predef = self._create_pseudoreference_tables(src_list_of_lists, tgt_list_of_lists, self.bitext_df['terms'].tolist())
        if pseudoreference_mode == 'predefined':
            selected_df = df_predef
        elif pseudoreference_mode == 'frequent':
            selected_df = df_freq
        elif pseudoreference_mode == 'first':
            selected_df = df_first

        pseudoreference_dict, self.doublet_dict = self._select_pseudoreferences(selected_df, pseudoreference_mode)

        self.bitext_df['pseudoref_terms'] = self.bitext_df['src_terms'].apply(lambda s: self._assign_pseudoreference(s, pseudoreference_dict))

        return df_first, df_freq, df_predef, pseudoreference_dict

    def compute_metric(self, output_mode='macro'):
        """
        Computes the metric based on two columns:
         - 'norm_tgt_terms' (real occurrences of the translated terms, output of 'align' method)
         - 'pseudoref_terms' (pseudoreference terms, output of 'assign_pseudoreferences' method).
         First, the lists of lists (for both columns) are flattened,
            then the percentage of the translated terms that match the pseudoreference terms is calculated
            (micro or macro averages, depending on the specified output mode).

        Parameters:
        :param output_mode: str, Specifies the averaging type for metric (from 'micro', 'macro').


        :return: metric - float (0-1), the percentage of the terms matching pseudoreferences
        :return: flat_df - DataFrame object with two columns: normalized terms and pseudoreferences (one term per line)
        """

        metric, flat_df = self._compute_metric(output_mode)

        return metric, flat_df

    def _name_compiler_wmt25(self, enforce_proper_terms=False):
        """
        Compiles a filename based on the configuration properties and language details.

        This method constructs a string for the filename by utilizing the values in 
        the `file_config` attribute of the class. The filename format changes 
        based on the track specified in the configuration. It also supports enforcing 
        a 'proper' naming mode through the enforce_proper_terms argument.

        :param enforce_proper_terms: bool, If True, refers to the 'proper' mode of the submission (important for retrieval of proper terms). Defaults to False. 
        :return: filename - str, the full file name.
        """
        mode = 'proper' if enforce_proper_terms else self.file_config['mode']
        if self.file_config["track"] == 1:
            filename = f'{self.file_config["file_path"]}track{self.file_config["track"]}/{self.file_config["system"]}.{self.lang_src}{self.lang_tgt}.{mode}.{self.file_config["file_type"]}'
        elif self.file_config["track"] == 2:
            filename = f'{self.file_config["file_path"]}track{self.file_config["track"]}/{self.file_config["system"]}.{self.file_config["year"]}.{self.lang_src}{self.lang_tgt}.{mode}.{self.file_config["file_type"]}'
        return filename

    def _open_jsonl_file(self, bitext_filepath):
        """
        Opens a JSONL file and returns its content as a Pandas DataFrame. Used for 1 track submissions.

        :param bitext_filepath : str, path to the JSONL file to be opened.

        :return bitext_df : Pandas DataFrame, the Pandas DataFrame containing the content of the JSONL file.
        """
        with open(bitext_filepath, 'r', encoding='utf-8') as f:
            data = [json.loads(s) for s in f.readlines()]
        bitext_df = pd.DataFrame(data)
        bitext_df = bitext_df.rename(columns={self.lang_src: 'src_raw', self.lang_tgt: 'mt_raw'})
        return bitext_df

    def _open_tsv_file(self, bitext_filepath):
        """
        Reads a TSV file and returns its content as a Pandas DataFrame. Used for 2 tracks submissions.

        :param bitext_filepath : str, path to the JSONL file to be opened.

        :return bitext_df : Pandas DataFrame, the Pandas DataFrame containing the content of the JSONL file.

        """
        
        bitext_df = pd.read_csv(bitext_filepath, sep='\t')
        bitext_df = bitext_df.rename(columns={self.lang_src: 'src_raw', self.lang_tgt: 'mt_raw'})
        bitext_df['terms'] = bitext_df['terms'].apply(lambda s: ast.literal_eval(s))
        bitext_df[['src_raw', 'mt_raw']] = bitext_df[['src_raw', 'mt_raw']].fillna(value='')
        return bitext_df

    def _retrieve_predefined_terms(self, return_random=False):
        """
        Retrieves predefined terms from the submitted file based on the specified mode (tailored to WMT25 format).
        
        Depending on mode, it retrieves the predefined terms from the same submission (if mode is 'proper')
            or from a proper mode submission (if mode is 'random' or 'noterm'; 
            if parameter `return_random` is True, it retrieves random terms from the same submission).
            
        :param return_random: bool, whether to retrieve random terms from the 'random' mode submission. Defaults to False.

        :return: filtered_term_column - List[Dict], A list of term dictionaries 
        :return:  reference_dict - global dictionary mapping source terms to their target translations. Not used. 
        """
        
        # 1. retrieving the column with the terms - either from the current df, or from the `proper` mode
        if self.file_config['mode'] == 'proper':
            term_column = self.bitext_df['terms'].tolist()

        elif self.file_config['mode'] == 'random':
            if return_random:
                print(f'retrieving random terms from the same file')
                term_column = self.bitext_df['terms'].tolist()
            else:
                proper_filepath = self._name_compiler_wmt25(enforce_proper_terms=True)
                print(f'retrieving terms from the proper file: {proper_filepath}')
                if self.file_config['track'] == 1:
                    proper_bitext = self._open_jsonl_file(proper_filepath)
                elif self.file_config['track'] == 2:
                    proper_bitext = self._open_tsv_file(proper_filepath)
                self.bitext_df['terms'] = proper_bitext['terms']
                term_column = proper_bitext['terms'].tolist()
        elif self.file_config['mode'] == 'noterm':
            if proper_inside:
                term_column = self.bitext_df['terms'].tolist()
            else:
                proper_filepath = self._name_compiler_wmt25(enforce_proper_terms=True)
                print(f'retrieving terms from the proper file: {proper_filepath}')
                if self.file_config['track'] == 1:
                    proper_bitext = self._open_jsonl_file(proper_filepath)
                elif self.file_config['track'] == 2:
                    proper_bitext = self._open_tsv_file(proper_filepath)
                self.bitext_df['terms'] = proper_bitext['terms']
                term_column = proper_bitext['terms'].tolist()

        # 2.
        # format of the current dict: {src_term: tgt_term} or {src_term: [tgt_term_i]}
        reference_dict = {} #if return_translations else None

        filtered_term_column = [list(segment.keys()) for segment in term_column]
        for segment in term_column:
            reference_dict.update(segment)
        return filtered_term_column, reference_dict

    def _llm_align_one_segment(self, src_sent, src_terms, tgt_sent, fewshots, ref_term_dict):
        """
        Aligns source terms to a target sentence using few-shot prompted ChatGPT and post-filtering (with awesomealign).
        Applied to a single segment.

        :param src_sent: str, The source sentence from which terms are aligned.
        :param src_terms: list[str], A list of terms from the source sentence to be aligned with the target sentence.
        :param tgt_sent: str, The target sentence with which the alignment is performed.
        :param fewshots: str, A string containing the 20-shot prompt (read from an external file) for ChatGPT alignment.
        :param ref_term_dict: dict, A reference dictionary containing terms and reference translation mappings used
            to detect over-alignment (regular ChatGPT behavior when extracted translations are longer than the really aligned words).

        :return: alg_dict, A dictionary mapping source terms to their aligned terms in the target sentence
        :return: overly_aligned, bool, indicating if over-alignment was detected during the process.
        """
        
        alg_dict = {}
        for src_term in src_terms:
            alg_term = self._llm_prompt_alignment(src_sent, src_term, tgt_sent, fewshots)
            overly_aligned = self._detect_over_alignment(alg_term, ref_term_dict, src_term)
            if overly_aligned:
                alg_term = self._filter_over_alignment(src_sent, tgt_sent, alg_term, src_term, ref_term_dict)

            if alg_term == '' or alg_term == ' ':
                #print(f'empty translation: {alg_term}')
                notfound = True
            elif not all([t.lower() in tgt_sent.lower() for t in alg_term.split(' ')]):
                #print(f'not all terms are found in the target sentence: {alg_term} vs {tgt_sent}')
                notfound = True
            else:
                notfound = False
            if notfound:
                alg_term = self.failed_dummy
            alg_dict[src_term] = alg_term
        return alg_dict, overly_aligned

    def _llm_prompt_alignment(self, src_sent, term, tgt_sent, fewshots):
        """
        Aligns a source term given the translated sentence and a few-shot instruction with a translated term (using ChatGPT).

        :param src_sent: str, source sentence.
        :param term: str, source term within `src_sent` to be aligned (usually normalized).
        :param tgt_sent: str, translated sentence in which the source term will be searched.
        :param fewshots: str, A string containing the 20-shot prompt (read from an external file) for ChatGPT alignment.

        :return: str, The translated term retrieved from the target sentence.
        """

        completion = self.openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
                {"role": "system", "content": fewshots},
                {
                    "role": "assistant",
                    "content": f"""
                  English sentence: {src_sent}
                  English term: {term}
                  Russian translation: {tgt_sent}
                  Translated term: """
                }
            ]
        )
        # print(completion.usage.prompt_tokens_details.cached_tokens)
        return completion.choices[0].message.content

    def _detect_over_alignment(self, found_translation, ref_term_dict, src_term):
        """
        Detect if the extracted translation is overly aligned compared to the reference term (regular ChatGPT behavior).
        The criterion is checking whether the extracted translation is longer than the reference term.

        :param found_translation: str, The extracted term translation to be compared.
        :param ref_term_dict: dict, A dictionary mapping source terms to their reference translations.
        :param src_term: str, The source term key to retrieve the corresponding reference translation.

        :return: bool, True if the found translation is longer than the reference translation; otherwise, False.
        """
        gt_translation = ref_term_dict[src_term]
        #print(f'comparing {found_translation} VS {gt_translation}')
        if len(self._word_tokenize(found_translation, self.lang_tgt)) > len(self._word_tokenize(gt_translation, self.lang_tgt)):
            #print(f'found translation is longer: {len(word_tokenize(found_translation))} > {len(word_tokenize(gt_translation))}')
            return True
        return False

    def _filter_over_alignment(self, src_sent, tgt_sent, translation, src_term, gt_dict):
        """
        Filters the target translation to remove over-aligned words (not aligned with the source term)
            with the help of awesomealign word alignment.

        First, normalize words on both sides. Then, iterate over the translated term. Words that do not
        align directly to the source term (or are incorrectly aligned to elements outside the term)
        are excluded. The output is a filtered translation preserving only the aligned words.

        :param src_sent: str, The complete source sentence.
        :param tgt_sent: str, The complete target sentence.
        :param translation: str, The extracted term translation in the target language.
        :param src_term: str, The term in the source language.
        :param gt_dict: dict, A dictionary mapping source terms (str) to their reference translation (str).

        :return: str, translation that retains only those correctly aligned with the source term based on the alignment mapping.
        """

        translation_words = self._word_tokenize(translation, self.lang_tgt)
        translation_lemmas = [self._normalize_word(w, self.lang_tgt) for w in translation_words]
        filtered_translation_words = []
        src_term_words = self._word_tokenize(src_term, self.lang_src)
        src_term_lemmas = [self._normalize_word(w, self.lang_src) for w in src_term_words]

        gt_translation = self._word_tokenize(gt_dict[src_term], self.lang_tgt)
        gt_lemmas = [self._normalize_word(w, self.lang_tgt) for w in gt_translation] # TODO: not used; delete
        alg_dict = self._awesomealign_alignment(src_sent, tgt_sent, return_words=True)
        inv_alg_dict = {self._normalize_word(v, self.lang_tgt): self._normalize_word(k, self.lang_src) for k, v in alg_dict.items()}

        for translation_lemma in translation_lemmas:
            # iterating over each lemma in an extracted term
            if translation_lemma in inv_alg_dict.keys():
                # translated lemma is aligned with something on the source side; checking if it is aligned to the term
                if inv_alg_dict[translation_lemma] not in src_term_lemmas:
                    # translated lemma is aligned to a word OUTSIDE the source term -> exclude from aligned sequence
                    pass
                else:
                    # translated lemma is aligned to a part of the source term -> include in the filtered alignment
                    idx = translation_lemmas.index(translation_lemma)
                    filtered_translation_words.append(translation_words[idx])
            else:
                # translated lemma is not aligned with anything on the source side -> exclude from aligned sequence
                pass
        return ' '.join(filtered_translation_words)


    def _awesomealign_alignment(self, sent_src, sent_tgt, return_words=False):

        """
        Performs alignment between a source and target sentence using an alignment model.
        Main code copied from https://huggingface.co/aneuraz/awesome-align-with-co, with additional normalization.


        :param sent_src: str, The source sentence to be aligned.
        :param sent_tgt: str, The target sentence to be aligned.
        :param return_words: bool, Whether to return alignments in the form of word strings (default: False).

        Returns:
        - dict
            If `return_words` is False, a dictionary mapping source word indices to target word indices.
            If `return_words` is True, a dictionary mapping source word strings to target word strings.

        """
        sent_src, sent_tgt = self._word_tokenize(sent_src, self.lang_src), self._word_tokenize(sent_tgt, self.lang_tgt)
        token_src, token_tgt = [self.aligner_tokenizer.tokenize(word) for word in sent_src], [self.aligner_tokenizer.tokenize(word) for word in
                                                                                 sent_tgt]
        wid_src, wid_tgt = [self.aligner_tokenizer.convert_tokens_to_ids(x) for x in token_src], [self.aligner_tokenizer.convert_tokens_to_ids(x)
                                                                                     for x in token_tgt]
        ids_src, ids_tgt = self.aligner_tokenizer.prepare_for_model(list(itertools.chain(*wid_src)), return_tensors='pt',
                                                       model_max_length=self.aligner_tokenizer.model_max_length, truncation=True)[
            'input_ids'], \
        self.aligner_tokenizer.prepare_for_model(list(itertools.chain(*wid_tgt)), return_tensors='pt', truncation=True,
                                    model_max_length=self.aligner_tokenizer.model_max_length)['input_ids']
        sub2word_map_src = []
        for i, word_list in enumerate(token_src):
            sub2word_map_src += [i for x in word_list]
        sub2word_map_tgt = []
        for i, word_list in enumerate(token_tgt):
            sub2word_map_tgt += [i for x in word_list]

        # alignment
        align_layer = 8
        threshold = 1e-3
        self.aligner_model.eval()
        with torch.no_grad():
            out_src = self.aligner_model(ids_src.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]
            out_tgt = self.aligner_model(ids_tgt.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]

            dot_prod = torch.matmul(out_src, out_tgt.transpose(-1, -2))

            softmax_srctgt = torch.nn.Softmax(dim=-1)(dot_prod)
            softmax_tgtsrc = torch.nn.Softmax(dim=-2)(dot_prod)

            softmax_inter = (softmax_srctgt > threshold) * (softmax_tgtsrc > threshold)

        align_subwords = torch.nonzero(softmax_inter, as_tuple=False)
        # print(align_subwords)
        align_word_ids, align_words = {}, {}
        for i, j in align_subwords:
            # align_words.add( (sub2word_map_src[i], sub2word_map_tgt[j]) )
            align_word_ids[sub2word_map_src[i]] = sub2word_map_tgt[j]
            align_words[sent_src[sub2word_map_src[i]]] = sent_tgt[sub2word_map_tgt[j]]

        if not return_words:
            return align_word_ids
        else:
            return align_words

    def _word_tokenize(self, sequence, lang):
        """
        language-specific word tokenizer (nltk for most languages).

        Args:
        :param sequence: str, The input text sequence to tokenize.
        :param lang: str, language code for the input sequence.

        :return: A list of tokens (words) extracted from the input sequence.
        """
        if lang == 'zh':
            return jieba.lcut(sequence)
        else:
            return word_tokenize(sequence)

    def _normalize_word(self, word, lang):
        """
        language-specific word normalization (stanza for most languages, PyMorphy for Russian, no lemmatization for Chinese).

        Parameters:
        :param word: str, The word to be normalized.
        :param lang: str, language code representing the language of the provided word. Supported
            values include 'ru', 'en', 'de', 'es', and 'zh'.

        :return: str, normalized form of the input word.
        """
        if lang == 'ru':
            return self.ru_morph.parse(word)[0].normal_form
        elif lang in ['en', 'de', 'es']:
            if lang == self.lang_src:
                doc = self.stanza_src(word)
            elif lang == self.lang_tgt:
                doc = self.stanza_tgt(word)
            # print(f'output{doc}, type: {type(doc)}')
            lemmas = [word.lemma for sent in doc.sentences for word in sent.words]
            return lemmas[0].lower()
        elif lang == 'zh':
            return word

    def _normalize_pseudoreference_candidates(self, src_terms, alg_dict):
        """
        Wrapper for pseudoreference candidates normalization.

        Parameters:
        :param src_terms: List[Str], The list of source terms (needed for retrieving candidates).
        :param alg_dict: Dict[Str: Str] A dictionary mapping source terms to their aligned translations.

        :return: norm_list, List[Str], A list of normalized pseudoreference candidates.
        """
        norm_list = []
        for src_term in src_terms:
            norm_list.append(' '.join([self._normalize_word(w, self.lang_tgt) for w in word_tokenize(alg_dict[src_term])]))
        return norm_list


    
    def _create_pseudoreference_tables(self, s_list, t_list, r_list):
        """
        Creates tables with source terms as index and translation variants as columns.
        Then, depending on pseudoreference type (first occurrence, frequency count, and
        normalized predefined terms found in text), fills in the corresponding tables.

        :param s_list: List[List[Str]], source terms for a particular segment.
        :param t_list: List[List[Str]], target terms for a particular segment.
        :param r_list: List[Dict[Str: Str]], reference term dictionaries for a particular segment.

        :return: df_first: DataFrame indicating first occurrences (0 or 1).
        :return: df_freq: A frequency count DataFrame.
        :return: df_predef: A DataFrame with predefined term translations (found in text).
        """
        def flatten(l):
            return [item for sublist in l for item in sublist]

        def flatten_list_of_dicts(r_list):
            """
            Creates a new list of dictionaries, extending each dictionary in the input by duplicating it
            for as many times as the number of keys in the dictionary.

            :param r_list: List of dictionaries to process.
            :return: New extended list of dictionaries.
            """
            extended_r_list = []
            for r_dict in r_list:
                extended_r_list.extend([r_dict] * len(r_dict.keys()))
            return extended_r_list

        # flattening the dictionaries and lists (so that lists are grouped by terms, not by segments)
        s_list, t_list = flatten(s_list), flatten(t_list)
        r_list = flatten_list_of_dicts(r_list)
        s_ordered_list = sorted(set(s_list))
        t_unique_list = sorted(set(t_list))

        # creating 3 DataFrames for 3 types of pseudoreferences: first occurrence in text/most frequent/predefined
        df_first = pd.DataFrame(0, index=s_ordered_list, columns=t_unique_list)
        df_freq = pd.DataFrame(0, index=s_ordered_list, columns=t_unique_list)
        df_predef = pd.DataFrame(0, index=s_ordered_list, columns=t_unique_list)

        for s, t, r_dict in zip(s_list, t_list, r_list):
            # update df_first
            if df_first.loc[s].sum() == 0:
                df_first.at[s, t] = 1
            # update df_freq
            df_freq.at[s, t] += 1
            normalized_r = ' '.join([self._normalize_word(w, self.lang_tgt) for w in word_tokenize(r_dict[s])])
            # update df_predef
            if normalized_r == t:
                try:
                    df_predef.at[s, t] = 1
                except:
                    pass

        return df_first, df_freq, df_predef

    def _select_pseudoreferences(self, df, pseudoreference_mode):
        """
        Select pseudoreferences from a dataframe based on the specified mode.
            If pseudoreference not found, it is set to dummy value (unique for each term).

        :param df: DataFrame, rows as source terms and columns possible target terms.
        :param pseudoreference_mode: str, the mode of pseudoreference selection. Supported values are 'first', 'freq', and 'predef'.

        :return: pseudoref_dict, A dictionary mapping each row name to its selected pseudoreference target term.
        :return doublet_dict, A dictionary of many-to-one source-to-target mappings.
        """

        pseudoref_dict = {}
        for row_name, row in df.iterrows():
            if pseudoreference_mode in ['first', 'predefined']:
                try:
                    column_name = row[row == 1].index[0]  # Locate the column where the cell value is 1
                except IndexError:
                    column_name = self.pseudoref_dummy
            elif pseudoreference_mode == 'frequent':
                column_name = row.idxmax()
            column_name = self.pseudoref_dummy if column_name == self.failed_dummy else column_name
            pseudoref_dict[row_name] = column_name

        # post-factum logging of the many-to-one mapped terms:
        
        # Generate doublet_dict
        doublet_dict = defaultdict(list)
        for key, value in pseudoref_dict.items():
            doublet_dict[value].append(key)
        
        # Filter the dictionary to only include non-unique values
        doublet_dict = {key: value for key, value in doublet_dict.items() if len(value) > 1}

        print(f'WARNING: non-unique selected target terms: {doublet_dict}')
        # TODO: prohibit the many-to-one source-to-target pseudoreferences

        if self.pseudoref_dummy in doublet_dict.keys():
            dummy_terms = doublet_dict[self.pseudoref_dummy]
            for idx, term in enumerate(dummy_terms):
                pseudoref_dict[term] = f'{self.pseudoref_dummy}{idx}'

        return pseudoref_dict, doublet_dict
        
        

    def _assign_pseudoreference(self, src_terms: list[list[str]], pseudoreference_dict: dict):
        """
        Assign pseudoreference terms to a source list of terms.

        :param src_terms (list[list[str]]): A nested list of strings representing the source terms 
                to be mapped to pseudoreference terms.
        :param pseudoreference_dict (dict): A dictionary where the keys are individual source terms 
                and the values are their corresponding pseudoreference terms.

        :return: pseudoref_terms, A list containing the mapped pseudoreference terms.
        """
        pseudoref_terms = []
        for src_term in src_terms:
            pseudoref_terms.append(pseudoreference_dict[src_term])
        return pseudoref_terms

    def _compute_metric(self, metric):
        """
        Computes accuracy (either 'micro' or 'macro' averaging) for translated terms against pseudoreferences.
        
        First, flattens the segment-wise columns (to consider every term occurrence separately). 
        Then, computes the accuracy for each unique term. 
        Finally, computes the average accuracy.

        :param metric : str, The type of accuracy averaging to be computed. Options include 'micro' and 'macro' values.

        :return: metric_score (float), The computed score for the specified metric type 
        :return: flat_df (pd.DataFrame), DataFrame containing flattened target ('tgt') terms 
              and pseudoreference ('pseudoref') terms.
        """
        
        def flatten(list_of_lists):
            flattened_list = [item for sublist in list_of_lists for item in sublist]
            return flattened_list
        flat_tgt, flat_pseudoref = flatten(self.bitext_df['norm_tgt_terms'].tolist()), flatten(self.bitext_df['pseudoref_terms'].tolist())
        flat_df = pd.DataFrame({'tgt': flat_tgt, 'pseudoref': flat_pseudoref})

        if metric == 'micro':
            metric_score = flat_df[flat_df['pseudoref'] == flat_df['tgt']].shape[0] / flat_df.shape[0]
        elif metric == 'macro':
            metric_sum = 0
            for pseudoref_type in flat_df.pseudoref.unique():
                subset = flat_df[flat_df['pseudoref'] == pseudoref_type]
                corr_subset = subset[subset['tgt'] == subset['pseudoref']]
                metric_sum += len(corr_subset) / len(subset)
                metric_score = metric_sum / len(flat_df.pseudoref.unique())

        return metric_score, flat_df