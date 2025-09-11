# %%
import os
import json
import statistics
import collections
import utils

os.makedirs("../generated/", exist_ok=True)

LANGS = ["es", "de", "ru"]
with open("metric_track1/scores/track1_score_dict.json", "r") as f:
    data = json.load(f)

systems = list(data["de"]["proper"].keys())

# compute zscores of variables for system ranking
data_agg = collections.defaultdict(list)
for lang, lang_v in data.items():
    for task, task_v in lang_v.items():
        for sys, sys_v in task_v.items():
            if sys == "TranssionMT" and lang != "ru":
                task_v[sys] = {}
            for metric, val in sys_v.items():
                if val == -1:
                    continue
                # print(val)
                data_agg[(lang, task, metric)].append(val)
data_agg = {
    k: (statistics.mean(v), statistics.stdev(v))
    for k, v in data_agg.items()
}
for lang, lang_v in data.items():
    for task, task_v in lang_v.items():
        for sys, sys_v in task_v.items():
            for metric, val in list(sys_v.items()):
                if val == -1:
                    continue
                sys_v[metric+"_z"] = (
                    (val - data_agg[(lang, task, metric)][0]) /
                    data_agg[(lang, task, metric)][1]
                )

systems.sort(
    key=lambda sys: statistics.mean([
        (
            data[lang]["proper"][sys]["chrf2++"] +
            data[lang]["proper"][sys]["term_success_rate"]
        )
        for lang in LANGS
        if data[lang]["proper"][sys] != {}
    ]) + (-1000 if any(data[lang]["proper"][sys] == {} for lang in LANGS) else 0),
    reverse=True,
)

# %%


def color_cell_chrf(val):
    color = f"SeaGreen3!{max(0, min(95, (val-50)*4.5)):.0f}!Firebrick3!50"
    return f"\\cellcolor{{{color}}} {val:.1f}"


def color_cell_term(val):
    color = f"SeaGreen3!{max(0, min(95, (val-70)*3)):.0f}!Firebrick3!50"
    return f"\\cellcolor{{{color}}} {val:.1f}"


def nocolor_cell(val):
    return f"{val:.1f}"


with open("../generated/track1.tex", "w") as f:
    print(
        r"\begin{tabular}{l  c>{\tiny}c>{\tiny}c>{\tiny}c c>{\tiny}c>{\tiny}c>{\tiny}c |c    c>{\tiny}c>{\tiny}c>{\tiny}c c>{\tiny}c>{\tiny}c>{\tiny}c |c     c>{\tiny}c>{\tiny}c>{\tiny}c}",
        r"\toprule",
        r"& \multicolumn{4}{c}{\bf Proper, ChrF} & \multicolumn{4}{c|}{\bf Proper, Term.} & & \multicolumn{4}{c}{\bf Random, ChrF} & \multicolumn{4}{c|}{\bf Random, Term} & & \multicolumn{4}{c}{\bf NoTerm, ChrF}\\",
        r"\bf System  & \bf Avg & \bf EnEs & \bf EnDe & \bf EnRu   & \bf Avg & \bf EnEs & \bf EnDe & \bf EnRu  & & \bf Avg & \bf EnEs & \bf EnDe & \bf EnRu   & \bf Avg & \bf EnEs & \bf EnDe & \bf EnRu  & & \bf Avg & \bf EnEs & \bf EnDe & \bf EnRu \\",
        r"\midrule",
        sep="\n",
        file=f,
    )

    for sys in systems:
        print(
            utils.SYS_TO_NAME.get(sys, sys),
            # proper, chrf
            color_cell_chrf(statistics.mean([
                data[lang]["proper"][sys]["chrf2++"] for lang in LANGS
            ])) if all(data[lang]["proper"][sys] != {} for lang in LANGS) else "",
            *[
                nocolor_cell(data[lang]["proper"][sys]["chrf2++"])
                if data[lang]["proper"][sys] != {} else ""
                for lang in LANGS
            ],
            # proper, term
            color_cell_term(statistics.mean([
                data[lang]["proper"][sys]["term_success_rate"]*100 for lang in LANGS
            ])) if all(data[lang]["proper"][sys] != {} for lang in LANGS) else "",
            *[
                nocolor_cell(data[lang]["proper"][sys]
                             ["term_success_rate"]*100)
                if data[lang]["proper"][sys] != {} else ""
                for lang in LANGS
            ],
            "",
            # random, chrf
            color_cell_chrf(statistics.mean([
                data[lang]["random"][sys]["chrf2++"] for lang in LANGS
            ])) if all(data[lang]["random"][sys] != {} for lang in LANGS) else "",
            *[
                nocolor_cell(data[lang]["random"][sys]["chrf2++"])
                if data[lang]["random"][sys] != {} else ""
                for lang in LANGS
            ],
            # random, term
            color_cell_term(statistics.mean([
                data[lang]["random"][sys]["term_success_rate"]*100 for lang in LANGS
            ])) if all(data[lang]["random"][sys] != {} for lang in LANGS) else "",
            *[
                nocolor_cell(data[lang]["random"][sys]
                             ["term_success_rate"]*100)
                if data[lang]["random"][sys] != {} else ""
                for lang in LANGS
            ],
            "",
            # noterm, chrf
            color_cell_chrf(statistics.mean([
                data[lang]["noterm"][sys]["chrf2++"] for lang in LANGS
            ])) if all(data[lang]["noterm"][sys] != {} for lang in LANGS) else "",
            *[
                nocolor_cell(data[lang]["noterm"][sys]["chrf2++"])
                if data[lang]["noterm"][sys] != {} else ""
                for lang in LANGS
            ],
            sep=" & ",
            end="\\\\\n",
            file=f,
        )

    print(
        r"\bottomrule",
        r"\end{tabular}",
        sep="\n",
        file=f,
    )
