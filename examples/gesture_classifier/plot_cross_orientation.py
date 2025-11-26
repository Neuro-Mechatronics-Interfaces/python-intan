# plot_cross_orientation.py
# Creates a bar chart of macro-F1 across cross-orientation permutations for emg/imu/both.

import os, json, glob, re
import numpy as np
import matplotlib.pyplot as plt

MODEL_DIR = os.path.join("model")  # adjust if needed
PATTERN = os.path.join(MODEL_DIR, "hand_open_close_external_metrics_*.json")

# Heuristics to parse modality & permutation from save_tag
def parse_tag(path):
    name = os.path.splitext(os.path.basename(path))[0]
    tag = name.replace("hand_open_close_external_metrics_", "")
    t = tag.lower()
    # modality
    modality = "both" if "both" in t else ("emg" if "emg" in t else ("imu" if "imu" in t else "both"))
    # permutation (train->test shorthand)
    # expect things like: xori_both_np_ps, xori_emg_ns_pp, xori_imu_ps_nn, or your "xori_np_ps" older tag.
    # Normalize common forms: *_np_ps, *_ns_pp, *_ps_nn
    m = re.search(r'(_np_ps|_ns_pp|_ps_nn)$', t)
    if not m:
        # try to recover if the tag doesn't end with a code; search anywhere
        m = re.search(r'(np_ps|ns_pp|ps_nn)', t)
    perm = m.group(1).lstrip("_") if m else "np_ps"
    return tag, modality, perm

# Load all metric files
files = sorted(glob.glob(PATTERN))
if not files:
    print(f"[WARN] No metrics found at: {PATTERN}")
    print("Make sure you ran 2b_train_model.py with --save_tag so files like")
    print("'model/hand_open_close_external_metrics_xori_both_np_ps.json' exist.")
    raise SystemExit(0)

rows = []  # (perm, modality, tag, macroF1, acc, filepath)
for fp in files:
    try:
        with open(fp, "r", encoding="utf-8") as f:
            J = json.load(f)
        cr = J.get("classification_report", {})
        macro_f1 = cr.get("macro avg", {}).get("f1-score")
        acc = cr.get("accuracy")
        tag, modality, perm = parse_tag(fp)
        if macro_f1 is None or acc is None:
            print(f"[WARN] Missing metrics in {fp}; skipping.")
            continue
        rows.append((perm, modality, tag, float(macro_f1), float(acc), fp))
    except Exception as e:
        print(f"[WARN] Failed to parse {fp}: {e}")

if not rows:
    print("[WARN] No usable metric rows parsed. Check filenames and JSON structure.")
    raise SystemExit(0)

# Sort for stable plotting: permutations in fixed order; modality fixed order
perm_order = ["np_ps", "ns_pp", "ps_nn"]  # (train: N+P→test S), (N+S→P), (P+S→N)
mod_order = ["emg", "imu", "both"]
perm_labels = {
    "np_ps": "Train N+P → Test S",
    "ns_pp": "Train N+S → Test P",
    "ps_nn": "Train P+S → Test N",
}
mod_labels = {"emg": "EMG", "imu": "IMU", "both": "EMG+IMU"}

# Build table [perm][mod] -> macroF1
table = {p: {m: None for m in mod_order} for p in perm_order}
acc_table = {p: {m: None for m in mod_order} for p in perm_order}
for perm, modality, tag, f1, acc, fp in rows:
    if perm not in table:
        table[perm] = {m: None for m in mod_order}
        acc_table[perm] = {m: None for m in mod_order}
    table[perm][modality] = f1
    acc_table[perm][modality] = acc

# Print a summary table to console
print("\nMacro-F1 summary (higher is better):")
w = max(len(perm_labels[p]) for p in perm_order)
print(f"{'Permutation':<{w}}  EMG     IMU     BOTH")
for p in perm_order:
    emg = table[p]["emg"]
    imu = table[p]["imu"]
    both = table[p]["both"]
    print(f"{perm_labels[p]:<{w}}  "
          f"{(f'{emg:.4f}' if emg is not None else '----'):>6}  "
          f"{(f'{imu:.4f}' if imu is not None else '----'):>6}  "
          f"{(f'{both:.4f}' if both is not None else '----'):>6}")

# ---- Plotting (single chart, no style/colors set) ----
x = np.arange(len(perm_order))
width = 0.25

fig = plt.figure(figsize=(9, 5))
ax = plt.gca()

vals_emg  = [table[p]["emg"]  if table[p]["emg"]  is not None else np.nan for p in perm_order]
vals_imu  = [table[p]["imu"]  if table[p]["imu"]  is not None else np.nan for p in perm_order]
vals_both = [table[p]["both"] if table[p]["both"] is not None else np.nan for p in perm_order]

ax.bar(x - width, vals_emg,  width, label="EMG")
ax.bar(x,         vals_imu,  width, label="IMU")
ax.bar(x + width, vals_both, width, label="EMG+IMU")

ax.set_xticks(x)
ax.set_xticklabels([perm_labels[p] for p in perm_order], rotation=0)
ax.set_ylabel("Macro-F1")
ax.set_title("Cross-Orientation Performance by Modality (external test)")
ax.set_ylim(0.0, 1.0)
ax.legend()

os.makedirs(MODEL_DIR, exist_ok=True)
out_path = os.path.join(MODEL_DIR, "cross_orientation_summary.png")
plt.tight_layout()
plt.savefig(out_path, dpi=200)
print(f"\nSaved figure → {out_path}")

# Show the plot (remove if running headless)
plt.show()
