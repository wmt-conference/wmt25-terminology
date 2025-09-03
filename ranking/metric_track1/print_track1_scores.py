import os
import json


with open("scores/track1_score_dict.json", "r") as f_in:
    score_dict = json.load(f_in)

# print (term_success_rate, chrf2++, team) tuples for each language and mode:
for lang in ["de", "es", "ru"]:
    for mode in ["noterm", "proper", "random"]:
        score_tuples = []
        print(f"\n\n---Language: {lang}, Mode: {mode}---")
        for team in score_dict[lang][mode]:
            if "chrf2++" not in score_dict[lang][mode][team]:
                continue

            chrf2pp = score_dict[lang][mode][team]["chrf2++"]
            term_acc = score_dict[lang][mode][team]["term_success_rate"]

            score_tuples.append((term_acc, chrf2pp, team))
            print(f"{team}: Term Acc = {term_acc:.4f}, chrF2++ = {chrf2pp:.2f}")

        score_tuples.sort(key=lambda x: (-x[0], -x[1], x[2]))

        # save the score tuples for plotting
        with open(f"./scores/track1_score_{lang}_{mode}_tuples.txt", "w") as f_out:
            for term_acc, chrf2pp, team in score_tuples:
                f_out.write(f"({term_acc}, {chrf2pp}, \"{team}\")\n")

print("\n\n")
print("All job done!")