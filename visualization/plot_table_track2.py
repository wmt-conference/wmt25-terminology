# %%
import os
import json
import statistics
import collections
import utils

# we have some placeholder -1 values for some systems' proper term acc because they did not submit to this mode/task
# the placeholder is needed because system ranking is based on this entry.
# now just replacing -1.0 and -100.0 which should be quite safe

# TODO: maybe there is a cleaner way to do this, but this currently works fine XD
import builtins
def print_no_placeholder_value(*args, **kwargs):
    sep   = kwargs.get("sep", " ")
    end   = kwargs.get("end", "\n")
    file  = kwargs.get("file", None)
    flush = kwargs.get("flush", False)
    # now rebuild what print() would output
    text = sep.join(str(a) for a in args)
    # replace placeholders with empty string; we know it must be the lowest value so the colouring is known
    text = text.replace("& \cellcolor{SeaGreen3!0!Firebrick3!50} -1.0 &", "& &").replace("& \cellcolor{SeaGreen3!0!Firebrick3!50} -100.0 &", "& &").replace(" -1.0 &", " &").replace(" -100.0 &", " &")
    builtins.print(text, end=end, file=file, flush=flush)
print = print_no_placeholder_value


os.makedirs("generated/", exist_ok=True)

LANGS = ["enzh", "zhen"]
with open("ranking/metric_track2/track2_score_dict.json", "r") as f:
    data = json.load(f)

systems = list(set(list(data["enzh"]["proper"].keys()) + list(data["enzh"]["noterm"].keys()) + list(data["enzh"]["random"].keys())))
# systems = [sys for sys in systems if "chrf2++" in data["zhen"]["proper"][sys]]

# compute zscores of variables for system ranking
data_agg = collections.defaultdict(list)
for lang, lang_v in data.items():
    for task, task_v in lang_v.items():
        for sys, sys_v in task_v.items():
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

for lang in LANGS:
    for task in ["proper", "noterm", "random"]:
        for sys in data[lang][task]:
            if data[lang][task][sys] == {} or len(data[lang][task][sys]) <= 2:
                data[lang][task][sys]["chrf2++"] = -1
                data[lang][task][sys]["proper_term_success_rate"] = -1
                data[lang][task][sys]["consistency_frequent"] = -1

systems.sort(
    key=lambda sys: statistics.mean(
        [data[lang]["proper"][sys]["chrf2++"]+data[lang]["proper"][sys]["proper_term_success_rate"] for lang in LANGS]),
    reverse=True,
)

# %%

def color_cell_chrf(val):
    color = f"SeaGreen3!{max(0, min(95, (val-40)*4.5)):.0f}!Firebrick3!50"
    return f"\\cellcolor{{{color}}} {val:.1f}"


def color_cell_acc(val):
    color = f"SeaGreen3!{max(0, min(95, (val-60)*3)):.0f}!Firebrick3!50"
    return f"\\cellcolor{{{color}}} {val:.1f}"


def color_cell_cons(val):
    color = f"SeaGreen3!{max(0, min(95, (val-80)*10)):.0f}!Firebrick3!50"
    return f"\\cellcolor{{{color}}} {val:.1f}"

def nocolor_cell(val):
    return f"{val:.1f}"


with open("generated/track2.tex", "w") as f:
    print(
        r"\begin{tabular}{l  cvv cvv cvv |c cvv cvv |c cvv}",
        r"\toprule",
        r"& \multicolumn{3}{c}{\bf Proper, ChrF} & \multicolumn{3}{c}{\bf Proper, Acc.} & \multicolumn{3}{c|}{\bf Proper, Cons.} &",
        r"& \multicolumn{3}{c}{\bf Random, ChrF} & \multicolumn{3}{c|}{\bf Random, Acc.} &",
        r"& \multicolumn{3}{c}{\bf NoTerm, ChrF} \\",
        r"\bf System  ",
        r"& \bf Avg & \bf EnZh & \bf ZhEn   & \bf Avg & \bf EnZh & \bf ZhEn  & \bf Avg & \bf EnZh & \bf ZhEn  &",
        r"& \bf Avg & \bf EnZh & \bf ZhEn   & \bf Avg & \bf EnZh & \bf ZhEn  &",
        r"& \bf Avg & \bf EnZh & \bf ZhEn  \\",
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
            ])),
            *[
                nocolor_cell(data[lang]["proper"][sys]["chrf2++"])
                for lang in LANGS
            ],
            # proper, term
            color_cell_acc(statistics.mean([
                data[lang]["proper"][sys]["proper_term_success_rate"]*100 for lang in LANGS
            ])),
            *[
                nocolor_cell(data[lang]["proper"][sys]["proper_term_success_rate"]*100)
                for lang in LANGS
            ],
            # proper, cons
            color_cell_acc(statistics.mean([
                data[lang]["proper"][sys]["consistency_frequent"]*100 for lang in LANGS
            ])) if all("consistency_frequent" in data[lang]["proper"][sys] for lang in LANGS) else "",
            *[
                nocolor_cell(data[lang]["proper"][sys]["consistency_frequent"]*100)
                if "consistency_frequent" in data[lang]["proper"][sys] else ""
                for lang in LANGS
            ],
            "",
            # random, chrf
            color_cell_chrf(statistics.mean([
                data[lang]["random"][sys]["chrf2++"] for lang in LANGS
            ])),
            *[
                nocolor_cell(data[lang]["random"][sys]["chrf2++"])
                for lang in LANGS
            ],
            # random, term
            color_cell_acc(statistics.mean([
                data[lang]["random"][sys]["proper_term_success_rate"]*100 for lang in LANGS
            ])),
            *[
                nocolor_cell(data[lang]["random"][sys]["proper_term_success_rate"]*100)
                for lang in LANGS
            ],
            "",
            # noterm, chrf
            color_cell_chrf(statistics.mean([
                data[lang]["noterm"][sys]["chrf2++"] for lang in LANGS
            ])),
            *[
                nocolor_cell(data[lang]["noterm"][sys]["chrf2++"])
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


with open("generated/track2_ext.tex", "w") as f:
    print(
        r"\begin{tabular}{l  cvv cvv cvv |c cvv cvv cvv |c cvv cvv cvv}",
        r"\toprule",
        r"& \multicolumn{3}{c}{\bf Proper, ChrF} & \multicolumn{3}{c}{\bf Proper, Acc.} & \multicolumn{3}{c|}{\bf Proper, Cons.} &",
        r"& \multicolumn{3}{c}{\bf Random, ChrF} & \multicolumn{3}{c}{\bf Random, Acc.} & \multicolumn{3}{c|}{\bf Random, Cons.} &",
        r"& \multicolumn{3}{c}{\bf NoTerm, ChrF} & \multicolumn{3}{c}{\bf NoTerm, Acc.} & \multicolumn{3}{c}{\bf NoTerm, Cons.} \\",
        r"\bf System  ",
        r"& \bf Avg & \bf EnZh & \bf ZhEn   & \bf Avg & \bf EnZh & \bf ZhEn   & \bf Avg & \bf EnZh & \bf ZhEn  &",
        r"& \bf Avg & \bf EnZh & \bf ZhEn   & \bf Avg & \bf EnZh & \bf ZhEn   & \bf Avg & \bf EnZh & \bf ZhEn  &",
        r"& \bf Avg & \bf EnZh & \bf ZhEn   & \bf Avg & \bf EnZh & \bf ZhEn   & \bf Avg & \bf EnZh & \bf ZhEn  \\",
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
            ])),
            *[
                nocolor_cell(data[lang]["proper"][sys]["chrf2++"])
                for lang in LANGS
            ],
            # proper, term
            color_cell_acc(statistics.mean([
                data[lang]["proper"][sys]["proper_term_success_rate"]*100 for lang in LANGS
            ])),
            *[
                nocolor_cell(data[lang]["proper"][sys]["proper_term_success_rate"]*100)
                for lang in LANGS
            ],
            # proper, cons
            color_cell_acc(statistics.mean([
                data[lang]["proper"][sys]["consistency_frequent"]*100 for lang in LANGS
            ])) if all("consistency_frequent" in data[lang]["proper"][sys] for lang in LANGS) else "",
            *[
                nocolor_cell(data[lang]["proper"][sys]["consistency_frequent"]*100)
                if "consistency_frequent" in data[lang]["proper"][sys] else ""
                for lang in LANGS
            ],
            "",
            # random, chrf
            color_cell_chrf(statistics.mean([
                data[lang]["random"][sys]["chrf2++"] for lang in LANGS
            ])),
            *[
                nocolor_cell(data[lang]["random"][sys]["chrf2++"])
                for lang in LANGS
            ],
            # random, term
            color_cell_acc(statistics.mean([
                data[lang]["random"][sys]["proper_term_success_rate"]*100 for lang in LANGS
            ])),
            *[
                nocolor_cell(data[lang]["random"][sys]["proper_term_success_rate"]*100)
                for lang in LANGS
            ],
            # random, cons
            color_cell_acc(statistics.mean([
                data[lang]["random"][sys]["consistency_frequent"]*100 for lang in LANGS
            ])) if all("consistency_frequent" in data[lang]["random"][sys] for lang in LANGS) else "",
            *[
                nocolor_cell(data[lang]["random"][sys]["consistency_frequent"]*100)
                if "consistency_frequent" in data[lang]["random"][sys] else ""
                for lang in LANGS
            ],
            "",
            # noterm, chrf
            color_cell_chrf(statistics.mean([
                data[lang]["noterm"][sys]["chrf2++"] for lang in LANGS
            ])),
            *[
                nocolor_cell(data[lang]["noterm"][sys]["chrf2++"])
                for lang in LANGS
            ],
            # noterm, term
            color_cell_acc(statistics.mean([
                data[lang]["noterm"][sys]["proper_term_success_rate"]*100 for lang in LANGS
            ])),
            *[
                nocolor_cell(data[lang]["noterm"][sys]["proper_term_success_rate"]*100)
                for lang in LANGS
            ],
            # noterm, cons
            color_cell_acc(statistics.mean([
                data[lang]["noterm"][sys]["consistency_frequent"]*100 for lang in LANGS
            ])) if all("consistency_frequent" in data[lang]["noterm"][sys] for lang in LANGS) else "",
            *[
                nocolor_cell(data[lang]["noterm"][sys]["consistency_frequent"]*100)
                if "consistency_frequent" in data[lang]["noterm"][sys] else ""
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
