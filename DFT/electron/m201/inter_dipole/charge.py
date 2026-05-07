# from pymatgen.io.vasp import Chgcar
# chg0 = Chgcar.from_file("AECCAR0")
# chg2 = Chgcar.from_file("AECCAR2")
# chg_total = chg0 + chg2
# chg_total.write_file("CHGCAR_total")


#!/usr/bin/env python3


def parse_bader(filename="ACF.dat"):
    charges = []
    start_parsing = False
    with open(filename, 'r') as f:
        for line in f:
            if '----' in line:
                start_parsing = True
                continue
            if start_parsing:
                if line.strip() == '':
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    charge = float(parts[4])
                    charges.append(charge)
                except ValueError:
                    continue

    total_charge = sum(charges)
    print(f"Number of atoms: {len(charges)}")
    print(f"Total Bader charge: {total_charge:.4f}")
    print("First 10 atomic charges:")
    for i, c in enumerate(charges[:10]):
        print(f"Atom {i+1}: {c:.4f}")

# if __name__ == "__main__":
#     parse_bader()

import pandas as pd

# === Function to load and parse ACF.dat files ===
def load_bader_acf(filepath):
    df = pd.read_csv(filepath, sep=r"\s+", skiprows=2, header=None,
                     names=["#", "X", "Y", "Z", "CHARGE", "MIN_DIST", "ATOMIC_VOL"],
                     comment="-")
    df = df.dropna().reset_index(drop=True)  # Ensure clean rows
    return df

# === Load files ===
df_interface = load_bader_acf("ACF_interface.dat")
df_Ga2O3 = load_bader_acf("ACF_Ga2O3.dat")  # 30 atoms
df_SiC = load_bader_acf("ACF_SiC.dat")      # 39 atoms

# === Ga2O3 part: interface 1–30
df_ga2o3_part = df_interface.iloc[0:30].copy()
df_ga2o3_part["reference_charge"] = df_Ga2O3["CHARGE"].values[:30]
df_ga2o3_part["delta_charge"] = df_ga2o3_part["CHARGE"] - df_ga2o3_part["reference_charge"]
df_ga2o3_part["region"] = "Ga2O3"

# === SiC part: interface 31–69 (39 atoms)
df_sic_part = df_interface.iloc[30:69].copy()
df_sic_part["reference_charge"] = df_SiC["CHARGE"].values[:39]
df_sic_part["delta_charge"] = df_sic_part["CHARGE"] - df_sic_part["reference_charge"]
df_sic_part["region"] = "SiC"

# === Combine and save
df_combined = pd.concat([df_ga2o3_part, df_sic_part], ignore_index=True)
df_combined = df_combined[["#", "region", "CHARGE", "reference_charge", "delta_charge"]]

# Save
df_combined.to_excel("bader_charge_transfer.xlsx", index=False)
print("Saved to bader_charge_transfer.xlsx")

# === Example: calculate total charge transfer for specific atom indices ===
# Provide atom numbers as 1-based index (e.g., from ACF.dat)
target_atoms = [1,2,	3,	4,	13,	14,	15,	16,	17,	18,	40,	41,	42,	46,	47,	48,	55,	56,	57,	64,	65,	66]  # You can modify this list

# Convert to 0-based index for pandas
target_indices = [i - 1 for i in target_atoms]

# Compute sums
subset = df_combined.iloc[target_indices]
sum_charge = subset["CHARGE"].sum()
sum_ref = subset["reference_charge"].sum()
sum_delta = subset["delta_charge"].sum()

print("\n--- Charge Summary for Selected Atoms ---")
print(f"Atom indices: {target_atoms}")
print(f"Total Bader charge: {sum_charge:.4f}")
print(f"Total reference charge: {sum_ref:.4f}")
print(f"Total charge transfer (ΔQ): {sum_delta:.4f} e")


