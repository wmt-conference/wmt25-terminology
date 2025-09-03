import json


def get_term_count(strings: list[str], term: str):
    '''
    count the number of occurences of a term in a list of strings. Used to obtain both input and output term counts.
    '''

    return sum(string.count(term.strip()) for string in strings)


def process_doc_data(data_dict: list[dict], input_language: str, output_language: str):
    """
    Extract the input strings, output strings, and the terminology dictionary for data read as jsonl.
    """

    input_strs = [item[input_language] for item in data_dict]
    output_strs = [item[output_language] for item in data_dict]
    assert len(input_strs) == len(output_strs)

    assert data_dict[0]["terms"] == data_dict[-1]["terms"] and data_dict[0]["terms"] is not None
    term_dict = data_dict[0]["terms"]

    return input_strs, output_strs, term_dict


def compute_doc_success_rate(data_dict: list[dict], input_language: str, output_language: str):

    """
    This is the function that computes document-level terminology succss rates.
    
    One challenge is the existance of telescopic terms in the terminology dictionary, e.g. "cat", "the cat", and "the cat is".
    Assuming that the target terms will also be telescopic if the source terms are telescopic, 
        we subtracts stats of super-strings from the stats of sub-strings.
    """

    input_strs, output_strs, term_dict = process_doc_data(data_dict, input_language, output_language)

    # convert the terminology dictionary to a list and sort by the length of the input term in descending order (longest to shortest)
    # we can process super-strings in the input terms first
    term_dict_list = [(k, v) for k, v in term_dict.items()]
    term_dict_list.sort(key=lambda x: len(x[0]), reverse=True)

    # keeps track of each input_term: [input_count, {output_term1: count1, output_term2: count2, ...}], 
    # so when we encouter sub-strings, we can subtract the stats for super-strings
    term_input_count_output_count_dict = {}

    sum_success_rate = 0
    valid_count = 0

    for input_term, output_terms in term_dict_list:
        input_term = input_term.strip()
        input_term_count = get_term_count(input_strs, input_term)
        # if an input term is not found in the input strings, skip the term
        if input_term_count == 0:
            continue
        else:
            term_input_count_output_count_dict[input_term] = [input_term_count, {}]

        if isinstance(output_terms, str):
            output_terms = [output_terms]
        output_terms = [output_term.strip() for output_term in output_terms]

        # count the number of occurences of the output term in the output strings
        output_terms_count = 0

        for potential_super_input_term, potential_list in term_input_count_output_count_dict.items():
            # print(potential_super_input_term, potential_list)
            potential_super_input_term_count, potential_super_output_terms_count_dict = potential_list
            # if the input term has a super-string input term computed already

            if input_term in potential_super_input_term and len(input_term) < len(potential_super_input_term):
                for output_term in output_terms:
                    output_term_count = get_term_count(output_strs, output_term)
                    for potential_super_output_term in potential_super_output_terms_count_dict:
                        if output_term in potential_super_output_term and len(output_term) < len(potential_super_output_term):
                            output_term_count -= potential_super_output_terms_count_dict[potential_super_output_term]
                    assert output_term_count >= 0
                    
                    term_input_count_output_count_dict[input_term][1][output_term] = output_term_count
                    output_terms_count += output_term_count

                input_term_count -= potential_super_input_term_count
                if input_term_count < 0:
                    continue 
                term_input_count_output_count_dict[input_term][0] = input_term_count

                assert output_terms_count >= 0
        
        # if a input term is not found in the input strings, skip the term
        if input_term_count == 0:
            continue
        else:
            success_rate = min(1.0 * output_terms_count / input_term_count, 1)
            sum_success_rate += success_rate
            valid_count += 1

    # returns the success rate averaged over all valid terms for the given document
    average_success_rate = sum_success_rate / valid_count
    return average_success_rate


if __name__ == "__main__":

    with open("2015.enzh.proper.gpt-4.1-nano-2025-04-14.translated.jsonl", "r") as f:
        data = []
        for line in f:
            data.append(json.loads(line))

    success_rate = round(compute_doc_success_rate(data, "en", "output"), 4)
    print(f"language pair: en-zh, year: 2015, success rate: {success_rate}")

    with open("2016.zhen.proper.gpt-4.1-nano-2025-04-14.translated.jsonl", "r") as f:
        data = []
        for line in f:
            data.append(json.loads(line))

    success_rate = round(compute_doc_success_rate(data, "zh", "output"), 4)
    print(f"language pair: zh-en, year: 2016, success rate: {success_rate}")
