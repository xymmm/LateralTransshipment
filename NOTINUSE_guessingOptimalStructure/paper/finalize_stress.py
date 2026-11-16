"""
NOTINUSE_guessingOptimalStructure/paper/finalize_stress.py : writes the numbers of the stress test into the paper.

    python finalize_stress.py

Reads ../out/round5b.csv through fill_stress.py, fills section_stress_template.tex and inserts it into
policy_structure.tex before the subsection on the frequency of the properties that fail. It also fills the totals in
the abstract and in the introduction. The script stops if the stress test contradicts a proved statement or if the
threshold property fails at a reached state in an instance with K <= R + v, because the text must then be written by hand.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
out = subprocess.run([sys.executable, os.path.join(HERE, "fill_stress.py")], capture_output=True, text=True).stdout
N = json.loads(out[:out.rindex("}") + 1])
assert sum(N["V"].values()) == 0, ("a proved statement is contradicted", N["V"])
assert N["OUTSIDE"] == 0
assert N["Q_last"] == 0
assert N["NSENDER_KLT"] == 0
fmt = lambda x: f"{x:,}"

# The text below was written for the result of 12 November 2026: failures at reached states occur in instances with
# K > R + v only, all with the demand pattern of Proposition 3. The assertions stop the script if this changes.
assert N["Q1r_inst_kle"] == 0, "threshold property fails at a reached state with K <= R + v: write the text by hand"
bad = [l for l in out[out.rindex("}") + 1:].splitlines() if l.strip() and not l.strip().startswith("instance")]
assert all(" counter " in l for l in bad), bad
thr = (f"At the set of all post-transshipment states the property fails in {N['Q1_inst']} of the {N['NSTRESS']} instances, "
       f"in {N['Q1_ip']} instance-periods. Of these instances {N['Q1_inst_kle']} have $K\\le R+v$ and {N['Q1_inst_kgt']} have $K>R+v$. "
       f"Condition~\\eqref{{eq:Q}} fails in {N['Q2_inst']} instances. No failure occurs in the last period. "
       f"At the reached states the property fails in {N['Q1r_inst']} instances and holds in the other {N['NSTRESS'] - N['Q1r_inst']}. "
       f"Both have $K>R+v$ and normal demand with $\\rho=0.02$. Both have the demand pattern of Proposition~\\ref{{prop:threshold}}: "
       f"the expected demand of location~1 alternates between 1 and a large value, and that of location~2 between 1 and a smaller value. "
       f"They are further cases of the mechanism of Section~\\ref{{sec:threshold}}. "
       f"In the {N['NKLE']} instances with $K\\le R+v$ the property holds at every reached state. "
       f"In these tests the threshold property with $K\\le R+v$ therefore fails only at states that the optimal policy does not reach.")

s = open(os.path.join(HERE, "section_stress_template.tex"), encoding="utf-8").read()
vals = dict(NRANDOM=N["NRANDOM"], NSTRESS=N["NSTRESS"], NIP=fmt(N["NIP"]), NSTATESB=fmt(N["NSTATESB"]), NKLE=N["NKLE"],
            NTRANS=fmt(N["NTRANS"]), NBEYOND=fmt(N["NBEYOND"]), NREORDER=fmt(N["NREORDER"]), NBELOW=fmt(N["NBELOW"]),
            NNOTBELOW=fmt(N["NNOTBELOW"]), NSENDER=fmt(N["NSENDER"]), NKLT=N["NKLT"], NLAST=fmt(N["NLAST"]),
            THRESHOLDTEXT=thr, E1=N["E1"], E2=N["E2"], E3=N["E3"], E4=N["E4"], E5=N["E5"], E6=N["E6"], E7=N["E7"],
            EINT=N["EINT"], EKC=N["EKC"], EKCQ=N["EKCQ"], NTEN=N["NTEN"], NMINTYPE=N["NMINTYPE"])
for k, v in vals.items():
    assert f"%{k}%" in s, k
    s = s.replace(f"%{k}%", str(v))
assert "%" not in s.replace("\\%", ""), [l for l in s.splitlines() if "%" in l.replace("\\%", "")]

p = os.path.join(HERE, "policy_structure.tex")
t = open(p, encoding="utf-8").read()
a = t.index("\\subsection{Stress test on extreme instances}") if "\\subsection{Stress test on extreme instances}" in t else None
b = t.index("\\subsection{Frequency of the properties that fail}")
t = (t[:a] if a is not None else t[:b]) + s + t[b:]
total_inst = 336 + N["NSTRESS"]
total_states = 9260688 + N["NSTATESB"]
t = t.replace("%%NTOTAL%%", fmt(total_inst)).replace("%%NSTATES%%", fmt(total_states))
open(p, "w", encoding="utf-8").write(t)
print("inserted; totals", total_inst, total_states)
print(json.dumps({k: N[k] for k in N if k not in ("types_inst",)}, indent=1)[:3000])
