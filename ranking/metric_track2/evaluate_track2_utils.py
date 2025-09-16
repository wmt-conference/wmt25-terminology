import os
import json

from sacrebleu.metrics.bleu import BLEU
from sacrebleu.metrics.chrf import CHRF


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


def prepare_source_reference_term_dict_data():
    raw_data_path_prefix = "../references/track2raw/document_v3"
    public_test_path_prefix = "../references/track2raw/public_test"
    
    # loop through year 2015 to 2024
    data = {}
    for year in range(2015, 2025, 1):
        data[year] = []
        
        # find all md files
        year_folder_path = f"{raw_data_path_prefix}/{year}/md"
        md_files = [f for f in os.listdir(year_folder_path) \
                        if os.path.isfile(f"{year_folder_path}/{f}") and f.lower().endswith(".md")]

        en_list = []
        zh_list = []

        for md_file in md_files:
            # MD files starting with a number is in English; those starting with "C" are in Chinese
            if md_file[0].isdigit():
                en_list.append(md_file)
            else:
                zh_list.append(md_file)

        en_list.sort(key=lambda x: int(x.split("_")[0]))
        zh_list.sort(key=lambda x: int(x.split("_")[0][1:]))
        
        # sanity check: en and zh files are of same size and same name.
        assert len(en_list) == len(zh_list)
        for en_md_file, zh_md_file in zip(en_list, zh_list):
            assert en_md_file == zh_md_file[1:]
            
            with open(f"{year_folder_path}/{en_md_file}", "r") as f_en, \
                 open(f"{year_folder_path}/{zh_md_file}", "r") as f_zh:
                en_lines = f_en.readlines()
                zh_lines = f_zh.readlines()
                # sanity check: en and zh files have same number of lines and are not empty
                assert len(en_lines) == len(zh_lines) and zh_lines and en_lines
                
                en_line = "".join(en_lines).strip()
                zh_line = "".join(zh_lines).strip()
                
                # add to data the en and zh test
                data[year].append({"en": en_line, "zh": zh_line})

        # read the public test release to get the random and proper terminology dictionaries
        translation_direction = "enzh" if year % 2 == 1 else "zhen"
        for term_type in ["proper", "random"]:
            str_list, term_dict_list = [], []
            with open(f"{public_test_path_prefix}/{year}.{translation_direction}.{term_type}.jsonl", "r") as f:
                for line in f:
                    dd = json.loads(line.strip())
                    if "en" in dd:  
                        str_list.append(dd["en"].strip())
                    else:
                        str_list.append(dd["zh"].strip())
                    term_dict_list.append(dd["terms"])
                # check that dictionaries are the same for each documents
                assert all(td == term_dict_list[0] for td in term_dict_list)
                # check that the published data has the same size as the raw data
                assert len(str_list) == len(term_dict_list) == len(data[year])

                for d, s in zip(data[year], str_list):
                    assert s == d["zh"] or s == d["en"]
                    d[term_type] = term_dict_list[0]
        
        # add a dummy noterm key to each data pair
        for d in data[year]:
            d["noterm"] = {}

    for year in data:
        with open(f"../references/track2/full_data_{year}.jsonl", "w") as f:
            for item in data[year]:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    prepare_source_reference_term_dict_data()