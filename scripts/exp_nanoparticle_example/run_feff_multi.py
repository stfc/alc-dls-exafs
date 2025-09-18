from larixite import cif2feffinp
from larch.xafs.feffrunner import feff8l
from larch.xafs import xftf
from larch.io import read_ascii
from larch import Group
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def plot_fourier_transform(exafs):
    fig, ax = plt.subplots()
    ax.plot(exafs.r, exafs.chir_mag)
    ax.set_xlabel(r'R ($\AA$)')
    ax.set_ylabel(r'$\vert\chi(\mathrm{R})\vert$')
    ax.set_title('Fourier Transform of EXAFS')
    fig.savefig('EXAFS_reference.pdf', format='pdf')
    fig.savefig('EXAFS_reference.svg', format='svg', transparent=True)
    plt.show()

def run_postprocessing(feff_directory):
    feff_data = read_ascii(f"{feff_directory}/chi.dat")
    g = Group()
    g.k = feff_data.k
    g.chi = feff_data.chi
    xftf(g, kweight=2, window='hanning', dk=1, kmin=2, kmax=14)
    return g

def run_feff_no_pymatgen(cifpath, absorber, feff_directory):
    inp = cif2feffinp(cifpath, absorber)
    with open(f"{feff_directory}/test.inp", "w") as f:
        f.write(inp)
    feff8l(folder=feff_directory, feffinp='test.inp', verbose=True)

def _load_chi_drop_k0(feff_dir):
    d = read_ascii(f"{feff_dir}/chi.dat")
    k = np.asarray(d.k, float)
    chi = np.asarray(d.chi, float)
    if k.size > 0 and abs(k[0]) < 1e-12:
        k = k[1:]
        chi = chi[1:]
    return k, chi

def _average_chi_same_grid(feff_dirs):
    k_ref = None
    chis = []
    for d in feff_dirs:
        k, chi = _load_chi_drop_k0(d)
        if k_ref is None:
            k_ref = k
        else:
            if len(k) != len(k_ref) or not np.allclose(k, k_ref, rtol=0, atol=1e-12):
                raise RuntimeError("k-grids differ beyond removable k=0; refusing to interpolate in minimal mode.")
        chis.append(chi)
    stack = np.vstack(chis)
    chi_mean = stack.mean(axis=0)
    g = Group()
    g.k = k_ref
    g.chi = chi_mean
    xftf(g, kweight=2, window='hanning', dk=1, kmin=2, kmax=14)
    return g, k_ref, chi_mean

if __name__ == "__main__":
    feff_root = Path('feff')
    cifs_dir = Path('cifs')
    absorbing_atom = 'Ag'
    if cifs_dir.exists():
        feff_root.mkdir(exist_ok=True)
        feff_dirs = []
        for cif in sorted(cifs_dir.glob("*.cif")):
            outdir = feff_root / cif.stem
            outdir.mkdir(parents=True, exist_ok=True)
            run_feff_no_pymatgen(cif, absorbing_atom, outdir)
            if (outdir / "chi.dat").exists():
                feff_dirs.append(outdir)
        if not feff_dirs:
            raise RuntimeError("No chi.dat files found after running FEFF.")
        g_avg, kgrid, chi_mean = _average_chi_same_grid(feff_dirs)
        np.savetxt(feff_root / "avg_chi_k.dat", np.column_stack([kgrid, chi_mean]), header="k chi_mean", fmt="%.6f")
        plot_fourier_transform(g_avg)
    else:
        feff_directory = Path('feff')
        cif_location = Path('first_structure.cif')
        feff_directory.mkdir(exist_ok=True)
        run_feff_no_pymatgen(cif_location, absorbing_atom, feff_directory)
        exafs = run_postprocessing(feff_directory)
        plot_fourier_transform(exafs)
