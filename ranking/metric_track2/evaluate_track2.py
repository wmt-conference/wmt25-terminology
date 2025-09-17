import os
import json
from functools import partial

from evaluate_track2_utils import get_bleu, get_chrf # metrics

VERBOSE=True
if VERBOSE:
    print = partial(print, flush=True)
else:
    print = lambda *args, **kwargs: None

# Known Caveat:
## If an input term is a sub-string of another input term (super-string), any incorrect terminology in the super-string's target will count towards the sub-string too. It is undefined whether this should be the case or not.


def get_term_success_rate(src_str: str, hyp_str: str, src_term: str, trg_terms: list, lowercase: bool) -> float:
    '''
    src_str: source/input string
    hyp_str: hypothesis/output string
    src_term: the source term
    trg_term: the corresponding target term(s)
    lowercase: whether to lowercase()

    Returns: ratio of the number of occurrences of the trg_term in the hyp_str
    to the number of occurrences of the src_term in the src_str, capped at 1.
    '''

    assert isinstance(trg_terms, list)
    trg_terms = [trg_terms.strip() for trg_terms in trg_terms]
    if lowercase:
        src_str = src_str.lower()
        hyp_str = hyp_str.lower()
        src_term = src_term.lower()
        trg_terms = [trg_terms.strip().lower() for trg_terms in trg_terms]

    input_term_count = src_str.count(src_term)
    # add up counts for each possible output_term
    output_terms_count = sum(hyp_str.count(t) for t in trg_terms)

    return min(1.0 * output_terms_count / input_term_count, 1.0)


# Start the evaluation for track 2

# list all the submissions
submission_folder_path = "../submissions/track2"
reference_folder_path = "../references/track2"
teams = sorted([d for d in os.listdir(submission_folder_path) if os.path.isdir(os.path.join(submission_folder_path, d))])

direction_map = {
    "zhen": [*range(2016, 2025, 2)],
    "enzh": [*range(2015, 2024, 2)],
}
score_dict = {k: {} for k in direction_map.keys()}
for direction, years in direction_map.items():
    src_lang = direction[:2]
    trg_lang = direction[2:]
    for mode in ["noterm", "proper", "random"]:
        score_dict[direction][mode] = {}
        for team in teams:
            submission_data = []
            reference_data = []
            for year in years:
                with open(f"{submission_folder_path}/{team}/{team}.{year}.{direction}.{mode}.jsonl", "r") as f:
                    year_data = [json.loads(line.strip()) for line in f]
                    submission_data.extend(year_data)

                with open(f"{reference_folder_path}/full_data_{year}.jsonl") as f:
                    year_data = [json.loads(line.strip()) for line in f]
                    reference_data.extend(year_data)

            # sanity check: participants got the same source data as our internal data
            for sub_d, ref_d in zip(submission_data, reference_data):
                assert sub_d[src_lang].strip() == ref_d[src_lang].strip()

            srcs = [ref_d[src_lang].strip() for ref_d in reference_data]
            hyps = [sub_d[trg_lang].strip() for sub_d in submission_data]
            refs = [ref_d[trg_lang].strip() for ref_d in reference_data]
            proper_term_dicts = [ref_d["proper"] for ref_d in reference_data]
            random_term_dicts = [ref_d["random"] for ref_d in reference_data]
            assert len(srcs) == len(hyps) == len(refs) == len(proper_term_dicts) == len(random_term_dicts)
            for term_dict in proper_term_dicts + random_term_dicts: # sanity check
                for _, v in term_dict.items():
                    assert isinstance(v, list), "target terms should be in a list"

            bleu_tokenizer = "13a" if trg_lang == "en" else "zh" # use the "zh" tokenizer for traditional Chinese (although this might not be ideal for the underlying tokenizer (jieba?))
            bleu_score = get_bleu(hyps, refs, max_ngram_order=4, tokenize=bleu_tokenizer)
            chrf_score = get_chrf(hyps, refs, char_order=6, word_order=2)

            score_dict[direction][mode][team] = {
                "bleu4": bleu_score.score,
                "chrf2++": chrf_score.score,
            }

            # compute with both proper and random dicts regardless of the mode
            for dict_mode in ["proper", "random"]:
                term_dicts = proper_term_dicts if dict_mode == "proper" else random_term_dicts
                # reset the stats
                valid_src_terms = 0.0
                aggregated_success_rate = 0.0
                for src, hyp, ref, term_dict in zip(srcs, hyps, refs, term_dicts):

                    for src_term, trg_terms in term_dict.items():
                        src_term = src_term.strip()
                        # only compute success rate for a source term if it appears in the source sentence and any of the corresponding target terms appear in the hypothesis.
                        # This is because the term dict has been extracted by GPT, and humans only reviewed the mapping.
                        # We could not guarantee that any target term will definitely appear in the hypothesis.
                        if src_term and src_term.lower() in src.lower() and any(t.lower() in ref.lower() for t in trg_terms):
                            valid_src_terms += 1
                            aggregated_success_rate += get_term_success_rate(src, hyp, src_term, trg_terms, lowercase=True) # we decided to measure success rate with everything lowercased

                score_dict[direction][mode][team][f"{dict_mode}_term_success_rate"] = aggregated_success_rate / valid_src_terms if valid_src_terms > 0 else -1.0

            print(f"Evaluated {team} for {direction} in {mode} mode: ", score_dict[direction][mode][team])

os.makedirs("./scores", exist_ok=True)
with open("./scores/track2_score_dict_new.json", "w") as f_out:
    json.dump(score_dict, f_out, indent=4, ensure_ascii=False)

print("\n\n")
print("All job done!")
