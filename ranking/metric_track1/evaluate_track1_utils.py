import os
import json

from sacrebleu.metrics.bleu import BLEU
from sacrebleu.metrics.chrf import CHRF


submission_dir = "../submissions/track1"
reference_dir = "../references/track1"


FULL_REF_DATA = {}
for langpair in ["ende", "enes", "enru"]:
    lang_data = []
    with open(f"{reference_dir}/full_data.{langpair}.jsonl", "r") as f:
        lang_data = [json.loads(line) for line in f]
    src_code, tgt_code = langpair[:2], langpair[2:]
    FULL_REF_DATA[langpair][src_code] = [item[src_code] for item in lang_data]
    FULL_REF_DATA[langpair][tgt_code] = [item[tgt_code] for item in lang_data]
    FULL_REF_DATA[langpair]["proper"] = [item["proper"] for item in lang_data]
    FULL_REF_DATA[langpair]["random"] = [item["random"] for item in lang_data]


def get_bleu(hyps, refs, max_ngram_order=4, tokenize="13a", verbose=False):
    assert len(hyps) == len(refs)

    bleu_metric = BLEU(max_ngram_order=max_ngram_order, tokenize=tokenize)
    bleu_score = bleu_metric.corpus_score(hyps, [refs])

    if verbose:
        print(f"\tBLEU score (max_ngram_order={max_ngram_order}): {bleu_score}")
        print("\t" + str(bleu_metric.get_signature()))

    return bleu_score


def get_chrf(hyps, refs, char_order=6, word_order=2, verbose=False):
    assert len(hyps) == len(refs)

    chrf_metric = CHRF(char_order=char_order, word_order=word_order)
    chrf_score = chrf_metric.corpus_score(hyps, [refs])

    if verbose:
        print(f"\tCHRF score (char_order={char_order}, word_order={word_order}): {chrf_score}")
        print("\t" + str(chrf_metric.get_signature()))

    return chrf_score


def get_shared_task_dict(lang, mode):
    shared_task_dict = FULL_REF_DATA[f"en{lang}"][mode]
    assert len(shared_task_dict) == 500
    return shared_task_dict


def get_shared_task_src(lang):
    shared_task_src = FULL_REF_DATA[f"en{lang}"]["en"]
    assert len(shared_task_src) == 500
    return shared_task_src


def get_shared_task_ref(lang):
    shared_task_ref = FULL_REF_DATA[f"en{lang}"][lang]
    assert len(shared_task_ref) == 500
    return shared_task_ref


def get_participant_hyp(filename, lang):
    participants_hyp = []
    with open(filename, "r") as f:
        for line in f:
            participants_hyp.append(json.loads(line)[lang].strip())

    assert len(participants_hyp) == 500
    return participants_hyp


if __name__ == "__main__":