import json
import os
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# ===================== Parameters =====================
versions = []
beta_list = []
rmsd_threshold = 0.1  # Å
num_workers = min(8, cpu_count())
# =======================================================

# ---------- Resolve File Path ----------
def resolve_path(version, beta):
    candidates = [
        f"./sample/{version}_{beta}.jsonl",  # default naming
        f"./sample/{version}_Traj2Relax_{beta}.jsonl",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

# ---------- Space Group Detection ----------
def get_space_group_relaxed(cell, atomic_numbers, positions):
    try:
        struct = Structure(lattice=cell, species=atomic_numbers, coords=positions, coords_are_cartesian=True)
        for tol in [1e-3, 1e-2, 5e-2]:
            try:
                sga = SpacegroupAnalyzer(struct, symprec=tol, angle_tolerance=10)
                sg = sga.get_space_group_symbol()
                if sg != "P1":
                    return sg
            except Exception:
                continue
        sga = SpacegroupAnalyzer(struct, symprec=0.1, angle_tolerance=15)
        return sga.get_space_group_symbol()
    except Exception:
        return None


# ---------- Single Sample Evaluation (for multiprocessing) ----------
matcher = StructureMatcher(
    primitive_cell=False,
    scale=False,
    attempt_supercell=False,
    stol=0.5,
)

def evaluate_one_record(args):
    record, dft_dict = args
    key = str(record["key"])
    if key not in dft_dict:
        return [], [], []

    dft_pos = np.array(dft_dict[key]["pos"])
    cell = np.array(record["cell"])
    atomic_numbers = record["atomic_numbers"]
    s_ref = Structure(cell, atomic_numbers, dft_pos)

    rmsd_list, sgmatch_list, time_list = [], [], []
    for pred in record["pred_list"]:
        pred_pos = np.array(pred["pos"])
        s_pred = Structure(cell, atomic_numbers, pred_pos)

        rmsd_val = matcher.get_rms_dist(s_ref, s_pred)
        if isinstance(rmsd_val, (tuple, list)):
            rmsd_val = rmsd_val[0]
        if rmsd_val is None:
            continue

        sg_ref = get_space_group_relaxed(cell, atomic_numbers, dft_pos)
        sg_pred = get_space_group_relaxed(cell, atomic_numbers, pred_pos)
        sg_match = (sg_ref == sg_pred) if sg_ref and sg_pred else False

        rmsd_list.append(rmsd_val)
        sgmatch_list.append(sg_match)
        time_list.append(pred.get("time", np.nan))
    return rmsd_list, sgmatch_list, time_list


# ================== Main Script ==================
dft_dict = {}
with open(resolve_path(versions[0], beta_list[0]), "r", encoding="utf-8") as fin:
    for line in fin:
        record = json.loads(line)
        if "dft" in record:
            dft_dict[str(record["key"])] = {
                "atomic_numbers": record["atomic_numbers"],
                "cell": record["cell"],
                "dft": record["dft"],
            }

summary = {}

for beta in beta_list:
    print(f"\n================= β = {beta} =================")
    beta_summary = {}
    for version in versions:
        pred_path = resolve_path(version, beta)
        if not pred_path:
            print(f"⚠️ File for {version} (β={beta}) not found, skipping.")
            beta_summary[version] = {
                "mean_rmsd": np.nan,
                "sg_match_rate": np.nan,
                "recovery_rate": np.nan,
                "mean_time": np.nan,
            }
            continue

        with open(pred_path, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f]

        with Pool(processes=num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(evaluate_one_record, [(r, dft_dict) for r in records]),
                    total=len(records),
                    desc=f"Evaluating {version} (β={beta}) using {num_workers} workers",
                )
            )

        rmsd_all, sgmatch_all, time_all = [], [], []
        for r, s, t in results:
            rmsd_all.extend(r)
            sgmatch_all.extend(s)
            time_all.extend(t)

        if not rmsd_all:
            beta_summary[version] = {
                "mean_rmsd": np.nan,
                "sg_match_rate": np.nan,
                "recovery_rate": np.nan,
                "mean_time": np.nan,
            }
            continue

        mean_rmsd = np.mean(rmsd_all)
        sg_match_rate = np.mean(sgmatch_all)
        recovery_rate = np.mean(np.array(rmsd_all) < rmsd_threshold)
        mean_time = np.mean(time_all)

        beta_summary[version] = {
            "mean_rmsd": mean_rmsd,
            "sg_match_rate": sg_match_rate,
            "recovery_rate": recovery_rate,
            "mean_time": mean_time,
        }

    # ✅ Print summary for current β
    print(f"\n----- Summary for β = {beta} -----")
    print(f"{'Version':<10}{'Mean RMSD':>12}{'SG Match':>12}{'Recovery':>12}{'Time(s)':>12}")
    print("---------------------------------------------------------------")
    for version, stats in beta_summary.items():
        print(
            f"{version:<10}"
            f"{stats['mean_rmsd']:>12.4f}"
            f"{stats['sg_match_rate']:>12.3f}"
            f"{stats['recovery_rate']:>12.3f}"
            f"{stats['mean_time']:>12.2f}"
        )
    print("---------------------------------------------------------------")

    summary[beta] = beta_summary

print("\n✅ All β evaluations completed!\n")
