# Terminology Consistency Metric (WMT2025 version)

Here you can find the metric for the term consistency evaluation and its results: 

* prerequisites: 
  * data: unzip the input files according to the readme files in [track1](data/submissions/track1/README.md) and [track2](data/submissions/track2/README.md) folders
  * libraries: see the [`requirements.txt`](requirements.txt) file. Core libraries are `polyfuzz`, `flair.embeddings`, `sentence_transformers` (for sentence alignment), `openai`, `transformers`, `torch`, (for term-based metric modules), `stanza`, `pymorphy3`, `jieba` (for language-specific normalization)
    * to reproduce the TermBasedMetric implementation, you'd need the openai API key. You can generate it [here](https://platform.openai.com/api-keys), and paste it into the [`openai-api-key.txt`](openai-api-key.txt) document.
* input data: [`data\submissions`](data\submissions) folder: 
  * track 1: same as in [the main metric](..\ranking\submissions\track1) (just flattened to a single folder)
  * track 2: preprocessed by the code from [`track2-sent-alignment.ipynb`](track2-sent-alignment.ipynb) (see below)
* codes: 
  * terminology-based metric: 
    * main module: [`termbasedmetric.py`](termbasedmetric.py)
    * scripts:
      * [`script_track1.py`](script_track1.py): run the following command: `python3 -m script_track1.py -s {source_lang} -t {target_lang} -m {terminology_mode}`
      * [`script_track2.py`](script_track2.py): run the following command: `python3 -m script_track2.py -s {source_lang} -t {target_lang} -m {terminology_mode}`
  * files for track 2 should be firstly tailored to the track1 format. The necessary codes are below:
    * main module: [`docpreprocessor.py`](docpreprocessor.py)
    * notebook with running the code: [`track2-sent-alignment.ipynb`](track2-sent-alignment.ipynb)
* results: 
  * metric outputs: 
    * [`processed`](processed) folder: aligned sentences with extracted term translations
    * [`pseudorefs`](pseudorefs) folder: term occurrences with respective pseudoreferences (for each possible pseudoreference choice: first occrrence/most frequent translation/predefined translation)
    * [`stats`](stats) folder: statistics for each file, split into language pair and terminology mode