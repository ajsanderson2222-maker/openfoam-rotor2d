"""
Rotor2D dynamic mesh post-processing:
  - Torque and thrust time series
  - Wake velocity at 1D and 2D downstream probes
  - Vorticity snapshot at final time
"""

import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import pyvista as pv

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})

OUT = Path("postProcessing")
IMGDIR = Path(".")

# ── 1. Parse forces.dat ────────────────────────────────────────────────────────
# Format: time   ((Fpx Fpy Fpz) (Fvx Fvy Fvz))   ((Mpx Mpy Mpz) (Mvx Mvy Mvz))

def parse_forces(fpath):
    times, Fx, Fy, Mz = [], [], [], []
    with open(fpath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # extract all floats in order
            nums = [float(x) for x in re.findall(r"[+-]?\d+\.?\d*[eE][+-]?\d+|[+-]?\d+\.\d+|[+-]?\d+", line)]
            if len(nums) < 13:
                continue
            # nums: [t, Fpx,Fpy,Fpz, Fvx,Fvy,Fvz, Mpx,Mpy,Mpz, Mvx,Mvy,Mvz]
            t = nums[0]
            fx = nums[1] + nums[4]   # pressure + viscous x
            fy = nums[2] + nums[5]   # pressure + viscous y
            mz = nums[9] + nums[12]  # pressure + viscous z-moment
            times.append(t)
            Fx.append(fx)
            Fy.append(fy)
            Mz.append(mz)
    return np.array(times), np.array(Fx), np.array(Fy), np.array(Mz)

t, Fx, Fy, Mz = parse_forces(OUT / "rotorForces/0/forces.dat")

# ── 2. Parse wake probes U ─────────────────────────────────────────────────────
# Only probes 0 (1D, x=0.04) and 1 (2D, x=0.08) are inside the domain

def parse_vector_probes(fpath, n_probes):
    times, data = [], [[] for _ in range(n_probes)]
    with open(fpath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # extract all floats
            nums = [float(x) for x in re.findall(r"[+-]?\d+\.?\d*[eE][+-]?\d+|[+-]?\d+\.\d+", line)]
            # time + n_probes * 3 components
            expected = 1 + n_probes * 3
            if len(nums) < expected:
                continue
            times.append(nums[0])
            for i in range(n_probes):
                data[i].append(nums[1 + i*3 : 1 + i*3 + 3])
    return np.array(times), [np.array(d) for d in data]

t_wake, wake = parse_vector_probes(OUT / "wakeProbes/0/U", n_probes=3)
# probe 0: 1D downstream  probe 1: 2D downstream  probe 2: outside domain (skip)
Ux_1D = wake[0][:, 0]
Ux_2D = wake[1][:, 0]

# ── 3. Steady-state window (second half of simulation) ─────────────────────────
half = len(t) // 2

def pclim(arr, lo=1, hi=99, pad=0.3):
    vlo, vhi = np.percentile(arr, lo), np.percentile(arr, hi)
    span = max(vhi - vlo, 1e-12)
    return vlo - pad * span, vhi + pad * span

# ── 4. Figure 1: Torque and thrust time series ─────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
fig.suptitle("Rotor2D — Force and Torque Time Series", fontweight="bold")

ax = axes[0]
ax.plot(t, Mz * 1e6, color="#e11d48", lw=0.8, label="Torque")
ax.set_ylabel("Torque  [μN·m]")
ax.set_ylim(*pclim(Mz[half:] * 1e6))
Mz_mean = np.mean(Mz[half:])
ax.axhline(Mz_mean * 1e6, color="#e11d48", lw=1.2, ls="--", alpha=0.6, label=f"Mean = {Mz_mean*1e6:.3f} μN·m")
ax.legend(fontsize=9)

ax = axes[1]
# Thrust = net axial force (x-direction in freestream-aligned frame)
# For a rotor in a crossflow, drag = Fx; lateral = Fy
ax.plot(t, Fx * 1e3, color="#2563eb", lw=0.8, label="Thrust (Fx)")
ax.plot(t, Fy * 1e3, color="#16a34a", lw=0.8, alpha=0.7, label="Lateral (Fy)")
ax.set_ylabel("Force  [mN]")
ax.set_xlabel("Time  [s]")
Fx_mean = np.mean(Fx[half:])
ax.axhline(Fx_mean * 1e3, color="#2563eb", lw=1.2, ls="--", alpha=0.6, label=f"Fx mean = {Fx_mean*1e3:.3f} mN")
all_F = np.concatenate([Fx[half:] * 1e3, Fy[half:] * 1e3])
ax.set_ylim(*pclim(all_F))
ax.legend(fontsize=9)

fig.tight_layout()
fig.savefig(IMGDIR / "forces.png", bbox_inches="tight")
plt.close(fig)
print("Saved forces.png")

# ── 5. Figure 2: Wake velocity deficit ─────────────────────────────────────────
half_w = len(t_wake) // 2
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(t_wake, Ux_1D, color="#7c3aed", lw=0.8, label="1D downstream (x = 0.04 m)")
ax.plot(t_wake, Ux_2D, color="#db2777", lw=0.8, label="2D downstream (x = 0.08 m)")
ax.axhline(1.0, color="gray", lw=1, ls=":", label="U∞ = 1 m/s")
ax.set_xlabel("Time  [s]")
ax.set_ylabel("Axial velocity Ux  [m/s]")
ax.set_title("Wake Velocity Recovery — Downstream Probes", fontweight="bold")
all_u = np.concatenate([Ux_1D[half_w:], Ux_2D[half_w:]])
ax.set_ylim(*pclim(all_u))
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(IMGDIR / "wake_velocity.png", bbox_inches="tight")
plt.close(fig)
print("Saved wake_velocity.png")

# ── 6. Figure 3: KPI summary table ────────────────────────────────────────────
omega_rpm = 60.0
omega_rad = omega_rpm * 2 * np.pi / 60
r_tip = 0.02       # m
U_inf = 1.0
TSR = omega_rad * r_tip / U_inf
D = 2 * r_tip
nu = 1e-5
Re = U_inf * D / nu

Mz_ss_mean = np.mean(Mz[half:])
Mz_ss_std  = np.std(Mz[half:])
Fx_ss_mean = np.mean(Fx[half:])

rho = 1.225
A = 4e-4   # reference area (Aref in forceCoeffs)
q_inf = 0.5 * rho * U_inf**2
Cm = Mz_ss_mean / (q_inf * A * D / 2)
Ct = Fx_ss_mean / (q_inf * A)

col_labels = ["KPI", "Value", "Notes"]
rows = [
    ("Rotation rate", f"{omega_rpm:.0f} RPM", "solidBody"),
    ("Tip speed ratio (TSR)", f"{TSR:.3f}", "λ = ωR/U∞"),
    ("Reynolds number", f"{Re:.0f}", "based on D"),
    ("Mean torque", f"{Mz_ss_mean*1e6:.3f} μN·m", "ss average"),
    ("Torque fluctuation", f"{Mz_ss_std*1e6:.3f} μN·m", "1σ"),
    ("Moment coeff Cm", f"{Cm:.4f}", "|Mz|/(q A D/2)"),
    ("Mean thrust Fx", f"{Fx_ss_mean*1e3:.3f} mN", "ss average"),
    ("Thrust coeff Ct", f"{Ct:.4f}", "Fx/(q A)"),
]

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.axis("off")
tbl = ax.table(
    cellText=rows,
    colLabels=col_labels,
    cellLoc="center",
    loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.2, 1.6)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#1e3a5f")
        cell.set_text_props(color="white", fontweight="bold")
    elif r % 2 == 0:
        cell.set_facecolor("#f1f5f9")
    cell.set_edgecolor("#cbd5e1")
ax.set_title("Rotor2D — Performance Summary", fontweight="bold", pad=10)
fig.tight_layout()
fig.savefig(IMGDIR / "kpi_table.png", bbox_inches="tight")
plt.close(fig)
print("Saved kpi_table.png")

# ── 7. Vorticity snapshot (final time step VTK) ────────────────────────────────
vtk_file = Path("VTK/openfoam-rotor2d_10000.vtk")
if vtk_file.exists():
    mesh = pv.read(str(vtk_file))
    sl = mesh.slice(normal="z", origin=(0, 0, 0))

    # compute vorticity z-component if vorticity field present
    if "vorticity" in sl.array_names:
        sl["vz"] = sl["vorticity"][:, 2]
        scalar = "vz"
        label = "Vorticity ωz  [1/s]"
    elif "U" in sl.array_names:
        sl["Ux"] = sl["U"][:, 0]
        scalar = "Ux"
        label = "Ux  [m/s]"
    else:
        scalar = None
        label = ""

    if scalar:
        vals = sl[scalar]
        vlim = float(np.percentile(np.abs(vals), 98))
        if vlim < 1e-10:
            vlim = 1.0

        xmin, xmax = -0.15, 0.25
        ymin, ymax = -0.15, 0.15
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
        yscale = (ymax - ymin) / 2 * 1.05

        pl = pv.Plotter(off_screen=True, window_size=[1600, 800])
        pl.set_background("#0f172a")
        pl.add_mesh(
            sl, scalars=scalar, cmap="RdBu",
            clim=[-vlim, vlim], show_edges=False,
            scalar_bar_args={
                "vertical": False,
                "position_x": 0.2, "position_y": 0.02,
                "width": 0.6, "height": 0.06,
                "title": label,
                "title_font_size": 14,
                "label_font_size": 12,
                "color": "white",
            }
        )
        pl.camera.position = (cx, cy, 1.0)
        pl.camera.focal_point = (cx, cy, 0.0)
        pl.camera.view_up = (0, 1, 0)
        pl.camera.parallel_projection = True
        pl.camera.parallel_scale = yscale
        pl.screenshot("vorticity_snapshot.png")
        pl.close()
        print("Saved vorticity_snapshot.png")
else:
    print(f"VTK file not found: {vtk_file}")

print("\nPost-processing complete.")
print(f"  TSR        = {TSR:.4f}")
print(f"  Re         = {Re:.0f}")
print(f"  Mean torque = {Mz_ss_mean*1e6:.4f} μN·m")
print(f"  Cm         = {Cm:.5f}")
print(f"  Ct         = {Ct:.5f}")
