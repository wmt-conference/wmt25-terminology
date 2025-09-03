import os
import json

from sacrebleu.metrics.bleu import BLEU
from sacrebleu.metrics.chrf import CHRF


submission_dir = "../submissions/track1"
reference_dir = "../references/track1"


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


def extract_track1_data():
    """
    Since the shared task data is a sampled version of the official data,
    - we extract source sentences for track 1 from two submissions.
    - we then extract the terminology dictionary from two submissions. We extract two versions, one "proper" and one "random"
    - we finally extract the reference translations from the official data based on the source sentences.
    """

    # list the submissions
    teams = [d for d in os.listdir(f"{submission_dir}/") \
                    if os.path.isdir(os.path.join(f"{submission_dir}/", d))]

    # consider the source from two teams to ensure the source lines are correct
    team0 = teams[0]
    team1 = teams[1]

    for lang in ["de", "es", "ru"]:

        hyp_file1 = f"{submission_dir}/{team0}/{team0}.en{lang}.proper.jsonl"
        hyp_file2 = f"{submission_dir}/{team1}/{team1}.en{lang}.proper.jsonl"

        src1, dict1 = [], []
        with open(hyp_file1, "r") as f_in:
            for line in f_in:
                data = json.loads(line)
                src1.append(data["en"].strip())
                dict1.append(data["terms"])

        src2, dict2 = [], []
        with open(hyp_file2, "r") as f_in:
            for line in f_in:
                data = json.loads(line)
                src2.append(data["en"].strip())
                dict2.append(data["terms"])

        # check that the items from both teams are the same
        ## length check
        assert len(src1) == len(src2) == len(dict1) == len(dict2) == 500
        ## each source sentence should match
        for s1, s2 in zip(src1, src2):
            assert s1 == s2, f"Source line mismatch: {s1} vs {s2}"
        ## each terminology dictionary entry should match
        for d1, d2 in zip(dict1, dict2):
            assert d1 == d2, f"Terminology mismatch: {d1} vs {d2}"
            
        # use the source sentences to find the corresponding references from the official release

        official_ref_file = f"{reference_dir}/term_postedits.test.en{lang}.pe"
        official_src_file = f"{reference_dir}/term_postedits.test.en{lang}.src"

        all_src_ref_map = {}
        with open(official_src_file, "r") as f1_in, open(official_ref_file, "r") as f2_in:
            for line1, line2 in zip(f1_in, f2_in):
                all_src_ref_map[line1.strip()] = line2.strip()

        ref = []
        for src in src1:
            ref.append(all_src_ref_map[src])

        with open(f"{reference_dir}/extracted_ref.en{lang}.txt", "w") as f:
            for src in ref:
                f.write(src + "\n")

        with open(f"{reference_dir}/extracted_src.en{lang}.txt", "w") as f:
            for src in src1:
                f.write(src + "\n")

        # write the proper dictionary
        with open(f"{reference_dir}/extracted_dict.proper.en{lang}.txt", "w") as f:
            for d in dict1:
                json.dump(d, f, ensure_ascii=False)
                f.write("\n")
                
        # additionally, we extract the random dictionary too
        rand_hyp_file1 = f"{submission_dir}/{team0}/{team0}.en{lang}.random.jsonl"
        rand_hyp_file2 = f"{submission_dir}/{team1}/{team1}.en{lang}.random.jsonl"

        rand_dict1, rand_dict2 = [], []
        with open(rand_hyp_file1, "r") as f_in:
            for line in f_in:
                rand_dict1.append(json.loads(line)["terms"])

        with open(rand_hyp_file2, "r") as f_in:
            for line in f_in:
                rand_dict2.append(json.loads(line)["terms"])

        # check that the items from both teams are the same
        ## length check
        assert len(rand_dict1) == len(rand_dict2) == 500

        ## each terminology dictionary entry should match
        for d1, d2 in zip(rand_dict1, rand_dict2):
            assert d1 == d2, f"Terminology mismatch: {d1} vs {d2}"

        # write the random dictionary
        with open(f"{reference_dir}/extracted_dict.random.en{lang}.txt", "w") as f:
            for d in rand_dict1:
                json.dump(d, f, ensure_ascii=False)
                f.write("\n")
        
        # create a dummy file for "noterm"
        with open(f"{reference_dir}/extracted_dict.noterm.en{lang}.txt", "w") as f:
            for _ in rand_dict1:
                json.dump({}, f)
                f.write("\n")


def get_shared_task_src(lang, src_filename_prefix="../references/track1/extracted_src.en"):
    shared_task_src = []

    with open(f"{src_filename_prefix}{lang}.txt", "r") as f:
        for line in f:
            shared_task_src.append(line.strip())

    assert len(shared_task_src) == 500
    return shared_task_src


def get_shared_task_ref(lang, src_filename_prefix="../references/track1/extracted_ref.en"):
    shared_task_ref = []

    with open(f"{src_filename_prefix}{lang}.txt", "r") as f:
        for line in f:
            shared_task_ref.append(line.strip())

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
    extract_track1_data()