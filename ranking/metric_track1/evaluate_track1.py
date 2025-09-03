import os
import json
from functools import partial

import stanza

from evaluate_track1_utils import extract_track1_data # data prep
from evaluate_track1_utils import get_bleu, get_chrf # metrics
from evaluate_track1_utils import get_shared_task_src, get_shared_task_ref, get_participant_hyp # read utils

VERBOSE=True
if VERBOSE:
    print = partial(print, flush=True)
else:
    print = lambda *args, **kwargs: None

# Get source, reference, and dictionary data ready for track 1
extract_track1_data()

def get_lemmatized_dict_lists(lang, dict_lists, nlp_pipelines):
    """
    lemmatize the source and target terms in the terminology dictionary list
    """
    lemmatized_dict_lists = []
    for dict_list in dict_lists:
        lemmatized_dict_list = []
        for k, v in dict_list:
            lemmatized_k = "|||".join([word.lemma for sent in nlp_pipelines["en"](k).sentences for word in sent.words]).lower()
            lemmatized_v = "|||".join([word.lemma for sent in nlp_pipelines[lang](v).sentences for word in sent.words]).lower()
            lemmatized_dict_list.append((lemmatized_k, lemmatized_v))
        lemmatized_dict_lists.append(lemmatized_dict_list)
        
    return lemmatized_dict_lists

# Start the evaluation
## list all the submissions
teams = sorted([d for d in os.listdir("../submissions/track1/") if os.path.isdir(os.path.join("../submissions/track1/", d))])

score_dict = {}
for lang in ["de", "es", "ru"]:
    # initialize the NLP pipelines for lemmatization.
    if lang == "ru":  # mwt is not supported for Russian
        nlp_pipeline = stanza.Pipeline(lang=lang, processors='tokenize,pos,lemma')
    else:
        nlp_pipeline = stanza.Pipeline(lang=lang, processors='tokenize,mwt,pos,lemma')
    nlp_pipelines = {
        "en": stanza.Pipeline(lang="en", processors='tokenize,mwt,pos,lemma'),
        lang: nlp_pipeline
    }

    print(f"Evaluating language: {lang}")
    score_dict[lang] = {}

    # get shared task's sources and references
    shared_task_src = get_shared_task_src(lang)
    shared_task_ref = get_shared_task_ref(lang)
    
    # Lemmatize the source texts
    lemmatized_shared_task_src = []
    for src_line in shared_task_src:
        lemmatized_shared_task_src.append(\
            "|||".join([word.lemma for sent in nlp_pipelines["en"](src_line).sentences for word in sent.words]).lower()) # pyright: ignore[reportAttributeAccessIssue]

    for mode in ["noterm", "proper", "random"]:
        score_dict[lang][mode] = {}
        dict_list = [] # list of terminology dictionaries
        with open(f"../references/track1/extracted_dict.{mode}.en{lang}.txt") as f:
            for line in f:
                dict_list.append(json.loads(line))
        # since the dictionaries have different sizes, turn the list of dict into a list of lists of tuples
        dict_lists = [list((k,v) for k,v in d.items()) for d in dict_list]
        # lemmatize the source-target items in the dictionaries
        lemmatized_dict_lists = get_lemmatized_dict_lists(lang, dict_lists, nlp_pipelines)
        
        # make sure the sizes are the same til now.
        assert len(shared_task_src) == len(lemmatized_shared_task_src) == len(dict_lists) == len(lemmatized_dict_lists) == 500

        # score each team.
        ## load the participant hypotheses every time,
        ## but re-use the shared_task_src, lemmatized_shared_task_src, dict_lists, and lemmatized_dict_lists
        for team in teams:
            print(f"Evaluating {team} for {lang} in {mode} mode")
            
            score_dict[lang][mode][team] = {}

            hyp_file = f"../submissions/track1/{team}/{team}.en{lang}.{mode}.jsonl"

            if not os.path.exists(hyp_file):
                continue  # skip if there is no hypothesis file in the folder

            participant_hyp = get_participant_hyp(hyp_file, lang)

            # BLEU score
            bleu_score = get_bleu(participant_hyp, shared_task_ref, max_ngram_order=4, verbose=False)
            score_dict[lang][mode][team]["bleu4"] = bleu_score.score

            # chrF2++ score
            chrf_score = get_chrf(participant_hyp, shared_task_ref, char_order=6, word_order=2, verbose=False)
            score_dict[lang][mode][team]["chrf2++"] = chrf_score.score

            # Now we compute the term matching statistics
            total_terms = 0
            matched_terms = 0
            
            if mode == "noterm":
                pass
            else:
                # Lemmatize the participant's hypotheses
                lemmatized_participant_hyp = []
                for hyp_line in participant_hyp:
                    lemmatized_participant_hyp.append(\
                        "|||".join([word.lemma for sent in nlp_pipelines[lang](hyp_line).sentences for word in sent.words]).lower()) # pyright: ignore[reportAttributeAccessIssue]
                assert len(participant_hyp) == len(lemmatized_participant_hyp) == 500
                
                for lemmatized_src_line, lemmatized_hyp_line, lemmatized_dict_list, src_line, hyp_line, dict_list in \
                    zip(lemmatized_shared_task_src, lemmatized_participant_hyp, lemmatized_dict_lists, shared_task_src, participant_hyp, dict_lists):
                    assert len(lemmatized_dict_list) == len(dict_list)

                    # we consider it a match as long as either the original or lemmatized item appears in the translation
                    for (lemmatized_k, lemmatized_v), (k, v) in zip(lemmatized_dict_list, dict_list):
                        if lemmatized_k in lemmatized_src_line or k.lower() in src_line.lower():
                            total_terms += 1 # increase the total count if the source item is found
                            if lemmatized_v in lemmatized_hyp_line or v.lower() in hyp_line.lower():
                                matched_terms += 1 # increase the match count if the target item is found
                        else:
                            continue
                            # print(f"{lemmatized_k} not found in source: {lemmatized_src_line} AND {k.lower()} not found in source: {src_line.lower()}")
                assert total_terms > 0, f"No terms found in {hyp_file} for {lang} in {mode} mode"

            score_dict[lang][mode][team]["term_success_rate"] = matched_terms / total_terms if total_terms > 0 else -1

os.makedirs("./scores", exist_ok=True)
with open("./scores/track1_score_dict.json", "w") as f_out:
    json.dump(score_dict, f_out, indent=4, ensure_ascii=False)

print("\n\n")
print("All job done!")