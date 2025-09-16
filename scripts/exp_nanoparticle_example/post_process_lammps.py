"""
post_process_lammps.py

This script post-processes LAMMPS simulation outputs to:
1. Parse simulation details from the LAMMPS input file (temperature, timestep, species, etc.).
2. Read and plot the radial distribution function (RDF) from 'rdfs.dat'.
3. Read and plot the coordination numbers (CN) from 'rdfs.dat'.
4. Parse mean squared displacement (MSD) values from 'msd_final.dat', reporting:
      - Total MSD
      - Species-resolved MSD (e.g., Ag, Au)
   and issue a WARNING if any MSD exceeds a specified threshold (default: 0.5 Å²).
5. Save RDF and CN plots as PNG images for easy inspection.

Outputs:
    - rdfs_plot.png        (Radial distribution function plot)
    - coordination_plot.png (Coordination number plot)
    - Console output with MSD values and threshold warnings.

Usage:
    Ensure the following files are present in the same directory:
        - LAMMPS input file:         lammpsIn_constP
        - RDF output from LAMMPS:    rdfs.dat
        - MSD output from LAMMPS:    msd_final.dat

Author: Andy Duff
"""

import re
import numpy as np
import matplotlib.pyplot as plt

def parse_lammps_info(lmp_path):
    info = {
        'species': None,
        'temp': None,
        'timestep': None,
        'rdf_avg_steps': None,
        'pair_style': None
    }
    with open(lmp_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('variable TEMP equal'):
                info['temp'] = float(line.split()[-1])
            elif line.startswith('timestep'):
                info['timestep'] = float(line.split()[-1])
            elif line.startswith('variable NUMMDSTEPS equal'):
                info['rdf_avg_steps'] = int(line.split()[-1])
            elif line.startswith('pair_style'):
                info['pair_style'] = line.split()[1]
            elif line.startswith('pair_coeff') and '.json' in line:
                tokens = line.split()
                info['species'] = [f"{tokens[-2]}1", f"{tokens[-1]}2"]

    # Print out parsed info for debugging
    print("\nParsed LAMMPS input info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    return info

def read_rdfs(rdfs_path):
    with open(rdfs_path, 'r') as f:
        lines = f.readlines()

    data_lines = []
    for line in lines:
        if line.startswith('#'):
            continue
        tokens = line.strip().split()
        if len(tokens) != 10:
            continue
        try:
            [float(x) for x in tokens]
            data_lines.append(tokens)
        except ValueError:
            continue

    if not data_lines:
        raise ValueError("No valid RDF data found in file.")

    data = np.array(data_lines, dtype=float)
    r = data[:, 1]  # second column is r
    g11, cn11 = data[:, 2], data[:, 3]
    g12, cn12 = data[:, 4], data[:, 5]
    g21, cn21 = data[:, 6], data[:, 7]
    g22, cn22 = data[:, 8], data[:, 9]

    print(f"\nRDF data loaded: {len(r)} points")
    return r, (g11, g12, g21, g22), (cn11, cn12, cn21, cn22)

def plot_rdfs(r, g, labels, temp, avg_ps, pair_style):
    print("\nPlotting RDFs...")
    g11, g12, g21, g22 = g
    plt.figure()
    plt.plot(r[g11 > 0], g11[g11 > 0], label=f"g_{labels[0]}-{labels[0]}")
    plt.plot(r[g12 > 0], g12[g12 > 0], label=f"g_{labels[0]}-{labels[1]}")
    plt.plot(r[g21 > 0], g21[g21 > 0], label=f"g_{labels[1]}-{labels[0]}")
    plt.plot(r[g22 > 0], g22[g22 > 0], label=f"g_{labels[1]}-{labels[1]}")
    plt.xlabel("r (Å)")
    plt.ylabel("g(r)")
    plt.title(f"RDF at {temp:.0f} K (averaged over {avg_ps:.2f} ps)")
    plt.legend()
    plt.figtext(0.99, 0.01, f"pair_style {pair_style}", ha='right', fontsize=8, alpha=0.5)
    plt.tight_layout()
    plt.savefig("rdfs_plot.png", dpi=300)
    print("Saved RDF plot: rdfs_plot.png")

def plot_cns(r, cn, labels, temp, avg_ps, pair_style):
    print("\nPlotting coordination numbers...")
    cn11, cn12, cn21, cn22 = cn
    plt.figure()
    plt.plot(r, cn11, label=f"CN_{labels[0]}-{labels[0]}")
    plt.plot(r, cn12, label=f"CN_{labels[0]}-{labels[1]}")
    plt.plot(r, cn21, label=f"CN_{labels[1]}-{labels[0]}")
    plt.plot(r, cn22, label=f"CN_{labels[1]}-{labels[1]}")
    plt.xlabel("r (Å)")
    plt.ylabel("Coordination Number")
    plt.title(f"Coordination Number at {temp:.0f} K (averaged over {avg_ps:.2f} ps)")
    plt.legend()
    plt.figtext(0.99, 0.01, f"pair_style {pair_style}", ha='right', fontsize=8, alpha=0.5)
    plt.tight_layout()
    plt.savefig("coordination_plot.png", dpi=300)
    print("Saved coordination plot: coordination_plot.png")


def read_msd(msd_path, threshold=0.5):
    print("\nReading MSD data...")
    with open(msd_path, 'r') as f:
        lines = [line.strip() for line in f if not line.startswith('#') and line.strip()]

    if not lines:
        raise ValueError("No valid MSD data found in file.")

    last_line = lines[-1]
    timestep, msd_all, msd_Ag, msd_Au = map(float, last_line.split())

    msd_values = {
        "All": msd_all,
        "Ag": msd_Ag,
        "Au": msd_Au
    }

    print("MSD values read (Å²): " + ", ".join(f"{k}={v:.5f}" for k, v in msd_values.items()))

    # Report max vibration
    max_species = max(msd_values, key=msd_values.get)
    max_value = msd_values[max_species]
    print(f"Maximum MSD: {max_value:.5f} Å² ({max_species})")

    # Threshold warning
    if any(msd > threshold for msd in msd_values.values()):
        print(f"WARNING: One or more MSD values exceed {threshold} Å²!")


# --- Run ---
info = parse_lammps_info("lammpsIn_constP")
r, g, cn = read_rdfs("rdfs.dat")
avg_ps = info['rdf_avg_steps'] * info['timestep']
plot_rdfs(r, g, info['species'], info['temp'], avg_ps, info['pair_style'])
plot_cns(r, cn, info['species'], info['temp'], avg_ps, info['pair_style'])
read_msd("msd_final.dat", threshold=0.5)
