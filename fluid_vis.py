#!/usr/bin/env python3
"""
fluid_vis.py

Two modes:
  1) sim  - 2D Navier-Stokes / vorticity pseudo-spectral simulation
  2) data - real atmospheric/ocean data from NetCDF/GRIB using xarray + cartopy

Examples:
  python fluid_vis.py --mode sim
  python fluid_vis.py --mode data --input /path/to/file.nc --var sst
  python fluid_vis.py --mode data --input /path/to/file.grib --var t2m
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap


# ============================================================
# PATHS / OUTPUT
# ============================================================
def get_output_dir() -> Path:
    desktop = Path.home() / "Desktop"
    if not desktop.exists():
        desktop = Path.cwd()
    outdir = desktop / "fluid_dynamics_outputs"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def save_animation(anim, outfile_base: Path, fps: int = 30):
    """
    Save MP4 if ffmpeg is available, otherwise GIF.
    """
    ffmpeg_ok = shutil.which("ffmpeg") is not None
    if ffmpeg_ok:
        out = outfile_base.with_suffix(".mp4")
        writer = FFMpegWriter(fps=fps, bitrate=2200)
        anim.save(out, writer=writer, dpi=180)
        print(f"[saved] {out}")
    else:
        out = outfile_base.with_suffix(".gif")
        writer = PillowWriter(fps=fps)
        anim.save(out, writer=writer, dpi=140)
        print(f"[saved] {out}")


# ============================================================
# SHARED VISUAL STYLE
# ============================================================
DARK_BG = "#041018"
SIM_CMAP = LinearSegmentedColormap.from_list(
    "sim_glow",
    [
        "#02040a",
        "#081827",
        "#0d3b66",
        "#1266a6",
        "#16a6c9",
        "#9be7ff",
        "#ffffff",
    ],
)
TRAIL_RGBA = np.array([0.80, 0.97, 1.00, 1.0])


# ============================================================
# MODE 1: NAVIER-STOKES / VORTICITY SIMULATION
# ============================================================
def periodic_dist(a, center, L):
    return ((a - center + 0.5 * L) % L) - 0.5 * L


def build_spectral_grid(n=192, L=2 * np.pi):
    x = np.linspace(0.0, L, n, endpoint=False)
    y = np.linspace(0.0, L, n, endpoint=False)
    X, Y = np.meshgrid(x, y)

    k = 2 * np.pi * np.fft.fftfreq(n, d=L / n)
    KX, KY = np.meshgrid(k, k)
    K2 = KX**2 + KY**2

    invK2 = np.zeros_like(K2)
    invK2[K2 != 0] = 1.0 / K2[K2 != 0]

    kmax = np.max(np.abs(k))
    dealias = (np.abs(KX) < (2 / 3) * kmax) & (np.abs(KY) < (2 / 3) * kmax)

    return X, Y, KX, KY, K2, invK2, dealias, L


def initial_vorticity(X, Y, L):
    """
    A lively initial condition: shear + random vortices + noise.
    """
    omega = 1.8 * np.sin(2.0 * Y) + 0.9 * np.cos(3.0 * X) * np.sin(2.0 * Y)

    vortices = [
        (0.22 * L, 0.68 * L, +8.0, 0.22, 0.22),
        (0.62 * L, 0.34 * L, -7.0, 0.18, 0.18),
        (0.80 * L, 0.56 * L, +6.5, 0.25, 0.20),
        (0.43 * L, 0.22 * L, -5.0, 0.16, 0.22),
    ]
    for cx, cy, amp, sx, sy in vortices:
        dx = periodic_dist(X, cx, L)
        dy = periodic_dist(Y, cy, L)
        omega += amp * np.exp(-(dx**2 / (2 * sx**2) + dy**2 / (2 * sy**2)))

    rng = np.random.default_rng(7)
    omega += 0.12 * rng.standard_normal(size=X.shape)
    return omega


def spectral_velocity_and_rhs(omega, t, X, Y, KX, KY, K2, invK2, dealias, nu=2e-3):
    """
    Compute u, v from vorticity and return RHS of:
        dω/dt = -u·∇ω + ν∇²ω + forcing
    """
    omega_hat = np.fft.fft2(omega)

    # Streamfunction: ∇²ψ = -ω
    psi_hat = -omega_hat * invK2

    # Velocity from streamfunction
    u = np.fft.ifft2(1j * KY * psi_hat).real
    v = np.fft.ifft2(-1j * KX * psi_hat).real

    # Derivatives of vorticity
    omega_x = np.fft.ifft2(1j * KX * omega_hat).real
    omega_y = np.fft.ifft2(1j * KY * omega_hat).real

    # A gentle, time-dependent forcing that keeps the flow alive
    forcing = (
        0.18 * np.sin(4.0 * Y + 0.30 * t)
        + 0.10 * np.cos(2.0 * X - 0.17 * t)
        + 0.06 * np.sin(3.0 * X + 2.0 * Y + 0.11 * t)
    )

    adv = u * omega_x + v * omega_y
    diff = nu * np.fft.ifft2(-K2 * omega_hat).real

    rhs = -adv + diff + forcing

    # Dealias RHS
    rhs_hat = np.fft.fft2(rhs)
    rhs_hat *= dealias
    rhs = np.fft.ifft2(rhs_hat).real

    return u, v, rhs


def rk4_step(omega, t, dt, X, Y, KX, KY, K2, invK2, dealias, nu=2e-3):
    u1, v1, k1 = spectral_velocity_and_rhs(omega, t, X, Y, KX, KY, K2, invK2, dealias, nu)
    _, _, k2 = spectral_velocity_and_rhs(
        omega + 0.5 * dt * k1, t + 0.5 * dt, X, Y, KX, KY, K2, invK2, dealias, nu
    )
    _, _, k3 = spectral_velocity_and_rhs(
        omega + 0.5 * dt * k2, t + 0.5 * dt, X, Y, KX, KY, K2, invK2, dealias, nu
    )
    _, _, k4 = spectral_velocity_and_rhs(
        omega + dt * k3, t + dt, X, Y, KX, KY, K2, invK2, dealias, nu
    )

    omega_new = omega + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # Enforce dealiasing after the step
    omega_hat = np.fft.fft2(omega_new)
    omega_hat *= dealias
    omega_new = np.fft.ifft2(omega_hat).real

    return omega_new, u1, v1


def bilinear_sample_periodic(field, xp, yp, L):
    """
    Periodic bilinear sampling on a square [0, L) x [0, L) grid.
    """
    n = field.shape[0]
    xi = (xp / L) * n
    yi = (yp / L) * n

    xi = np.mod(xi, n)
    yi = np.mod(yi, n)

    i0 = np.floor(xi).astype(np.int32)
    j0 = np.floor(yi).astype(np.int32)
    i1 = (i0 + 1) % n
    j1 = (j0 + 1) % n

    tx = xi - i0
    ty = yi - j0

    f00 = field[j0, i0]
    f10 = field[j0, i1]
    f01 = field[j1, i0]
    f11 = field[j1, i1]

    return (
        (1 - tx) * (1 - ty) * f00
        + tx * (1 - ty) * f10
        + (1 - tx) * ty * f01
        + tx * ty * f11
    )


def advect_particles(particles, u, v, dt, L):
    """
    RK2 advection for tracer particles on the periodic domain.
    """
    up = bilinear_sample_periodic(u, particles[:, 0], particles[:, 1], L)
    vp = bilinear_sample_periodic(v, particles[:, 0], particles[:, 1], L)

    xm = np.mod(particles[:, 0] + 0.5 * dt * up, L)
    ym = np.mod(particles[:, 1] + 0.5 * dt * vp, L)

    um = bilinear_sample_periodic(u, xm, ym, L)
    vm = bilinear_sample_periodic(v, xm, ym, L)

    particles[:, 0] = np.mod(particles[:, 0] + dt * um, L)
    particles[:, 1] = np.mod(particles[:, 1] + dt * vm, L)
    return particles


def run_simulation(args, outdir: Path):
    X, Y, KX, KY, K2, invK2, dealias, L = build_spectral_grid(n=args.n, L=2 * np.pi)

    omega0 = initial_vorticity(X, Y, L)
    particles0 = np.column_stack(
        [
            np.random.default_rng(123).uniform(0.0, L, args.n_particles),
            np.random.default_rng(456).uniform(0.0, L, args.n_particles),
        ]
    )

    trail_len = args.trail_len
    trail0 = np.repeat(particles0[None, :, :], trail_len, axis=0)

    state = {
        "omega": omega0.copy(),
        "particles": particles0.copy(),
        "trail": trail0.copy(),
        "t": 0.0,
    }

    fig, ax = plt.subplots(figsize=(14, 8), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect("equal")
    ax.axis("off")

    omega_abs0 = max(2.0, float(np.percentile(np.abs(omega0), 98)))

    im_omega = ax.imshow(
        omega0,
        extent=[0, L, 0, L],
        origin="lower",
        cmap="RdBu_r",
        interpolation="bicubic",
        vmin=-omega_abs0,
        vmax=omega_abs0,
        alpha=0.96,
    )

    trail_lc = LineCollection([], linewidths=1.2, alpha=1.0, capstyle="round")
    ax.add_collection(trail_lc)

    scat = ax.scatter(
        particles0[:, 0],
        particles0[:, 1],
        s=12,
        c=np.linspace(0.15, 1.0, args.n_particles),
        cmap="turbo",
        edgecolors="none",
        alpha=0.96,
    )

    title = ax.text(
        0.02,
        0.965,
        "2D Navier–Stokes / Vorticity Turbulence",
        transform=ax.transAxes,
        fontsize=18,
        fontweight="bold",
        color="white",
        ha="left",
        va="top",
    )

    subtitle = ax.text(
        0.02,
        0.925,
        "pseudo-spectral RK4 • tracer advection • glowing trails",
        transform=ax.transAxes,
        fontsize=10,
        color=(0.93, 0.95, 1.0, 0.85),
        ha="left",
        va="top",
    )

    stats = ax.text(
        0.02,
        0.04,
        "",
        transform=ax.transAxes,
        fontsize=10,
        color=(0.9, 0.95, 1.0, 0.92),
        ha="left",
        va="bottom",
    )

    def update(frame):
        omega = state["omega"]
        particles = state["particles"]
        trail = state["trail"]
        t = state["t"]

        # Render current state
        u, v, _ = spectral_velocity_and_rhs(
            omega, t, X, Y, KX, KY, K2, invK2, dealias, nu=args.nu
        )

        speed = np.hypot(u, v)
        omega_abs = max(2.0, float(np.percentile(np.abs(omega), 98)))

        im_omega.set_data(omega)
        im_omega.set_clim(-omega_abs, omega_abs)

        # Update particles/trails on current velocity field
        particles = advect_particles(particles, u, v, args.dt, L)
        trail = np.roll(trail, -1, axis=0)
        trail[-1] = particles

        segments = np.stack([trail[:-1], trail[1:]], axis=2).reshape(-1, 2, 2)
        trail_lc.set_segments(segments)

        # Trail alpha by age
        age = np.linspace(0.05, 1.0, trail_len - 1)
        alpha_by_age = np.repeat(age**1.8, args.n_particles)
        colors = np.tile(TRAIL_RGBA, (segments.shape[0], 1))
        colors[:, 3] = 0.04 + 0.96 * alpha_by_age
        trail_lc.set_color(colors)

        scat.set_offsets(particles)
        p_speed = bilinear_sample_periodic(speed, particles[:, 0], particles[:, 1], L)
        scat.set_array(p_speed)
        scat.set_sizes(10.0 + 40.0 * p_speed / (p_speed.max() + 1e-8))

        stats.set_text(
            f"t = {t:6.2f}   "
            f"mean speed = {speed.mean():.3f}   "
            f"vorticity rms = {np.sqrt(np.mean(omega**2)):.3f}"
        )

        # Advance simulation
        omega_next, _, _ = rk4_step(
            omega, t, args.dt, X, Y, KX, KY, K2, invK2, dealias, nu=args.nu
        )
        state["omega"] = omega_next
        state["particles"] = particles
        state["trail"] = trail
        state["t"] = t + args.dt

        return im_omega, trail_lc, scat, stats, title, subtitle

    anim = FuncAnimation(
        fig,
        update,
        frames=args.frames,
        interval=1000 // args.fps,
        blit=False,
    )

    base = outdir / "navier_stokes_vorticity"
    save_animation(anim, base, fps=args.fps)

    poster = outdir / "navier_stokes_vorticity_poster.png"
    fig.savefig(poster, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[saved] {poster}")

    plt.close(fig)


# ============================================================
# MODE 2: REAL NETCDF / GRIB DATA WITH XARRAY + CARTOPY
# ============================================================
def run_data_mode(args, outdir: Path):
    import xarray as xr
    import cartopy.crs as ccrs

    path = Path(args.input).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    is_grib = suffix in {".grib", ".grb", ".grib2", ".grb2"}
    engine = "cfgrib" if is_grib else None

    ds = xr.open_dataset(path, engine=engine)

    try:
        # Pick variable
        if args.var is not None:
            if args.var not in ds:
                raise KeyError(f"Variable '{args.var}' not found in dataset.")
            da = ds[args.var]
        else:
            preferred = [
                "sst", "sea_surface_temperature", "t2m", "u10", "v10", "z",
                "msl", "tp", "ssh", "temp", "temperature", "chl", "salinity"
            ]
            chosen = None
            for name in preferred:
                if name in ds.data_vars:
                    chosen = name
                    break
            if chosen is None:
                # First non-scalar data variable
                candidates = [name for name, v in ds.data_vars.items() if v.ndim >= 2]
                if not candidates:
                    raise ValueError("No suitable 2D/3D data variable found.")
                chosen = candidates[0]
            da = ds[chosen]

        da = da.squeeze(drop=True)

        # Identify lat/lon coordinates
        def find_name(names):
            for n in names:
                if n in ds.coords or n in ds.variables:
                    return n
            return None

        lat_name = find_name(["lat", "latitude", "nav_lat"])
        lon_name = find_name(["lon", "longitude", "nav_lon"])

        if lat_name is None or lon_name is None:
            raise ValueError(
                "Could not find latitude/longitude coordinates. "
                "Expected names like lat/lon or latitude/longitude."
            )

        # Reduce any extra non-spatial dimensions to first slice
        spatial_names = {lat_name, lon_name}
        time_name = next((d for d in da.dims if "time" in d.lower()), None)

        for d in list(da.dims):
            if d not in spatial_names and d != time_name and da.sizes[d] > 1:
                da = da.isel({d: 0})

        # Try to order dims as (time, lat, lon) if possible
        order = []
        if time_name and time_name in da.dims:
            order.append(time_name)
        if lat_name in da.dims:
            order.append(lat_name)
        if lon_name in da.dims:
            order.append(lon_name)
        order.extend([d for d in da.dims if d not in order])
        da = da.transpose(*order)

        lat = ds[lat_name].values
        lon = ds[lon_name].values

        if lat.ndim == 1 and lon.ndim == 1:
            Lon, Lat = np.meshgrid(lon, lat)
            use_imshow = True
            extent = [float(np.nanmin(lon)), float(np.nanmax(lon)), float(np.nanmin(lat)), float(np.nanmax(lat))]
        else:
            Lon, Lat = lon, lat
            use_imshow = False
            extent = [float(np.nanmin(lon)), float(np.nanmax(lon)), float(np.nanmin(lat)), float(np.nanmax(lat))]

        if time_name is not None:
            ntime = da.sizes[time_name]
            indices = np.linspace(0, ntime - 1, min(args.frames, ntime), dtype=int)
        else:
            indices = np.array([0])

        # Sample first frame for color scaling
        if time_name is not None:
            first = np.asarray(da.isel({time_name: int(indices[0])}).values)
        else:
            first = np.asarray(da.values)

        vmin = float(np.nanpercentile(first, 2))
        vmax = float(np.nanpercentile(first, 98))
        if np.isclose(vmin, vmax):
            vmin, vmax = float(np.nanmin(first)), float(np.nanmax(first) + 1e-9)

        fig = plt.figure(figsize=(14, 8), facecolor=DARK_BG)
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_facecolor(DARK_BG)

        def draw_frame(field, title_text):
            ax.clear()
            ax.set_facecolor(DARK_BG)

            if use_imshow:
                ax.imshow(
                    field,
                    extent=extent,
                    origin="lower",
                    transform=ccrs.PlateCarree(),
                    cmap=args.cmap,
                    vmin=vmin,
                    vmax=vmax,
                    interpolation="bilinear",
                )
            else:
                ax.pcolormesh(
                    Lon,
                    Lat,
                    field,
                    transform=ccrs.PlateCarree(),
                    cmap=args.cmap,
                    vmin=vmin,
                    vmax=vmax,
                    shading="auto",
                )

            ax.coastlines(linewidth=0.8, color="white", alpha=0.85)
            gl = ax.gridlines(
                draw_labels=True,
                linewidth=0.4,
                color="white",
                alpha=0.25,
                linestyle="--",
            )
            gl.top_labels = False
            gl.right_labels = False

            ax.set_extent(extent, crs=ccrs.PlateCarree())
            ax.set_title(title_text, fontsize=15, color="white", pad=12)

        if time_name is None:
            title = f"{path.name} | {args.var or da.name}"
            draw_frame(np.asarray(first), title)

            png = outdir / f"{path.stem}_{args.var or da.name or 'field'}.png"
            fig.savefig(png, dpi=240, bbox_inches="tight", facecolor=fig.get_facecolor())
            print(f"[saved] {png}")
        else:
            def update(i):
                idx = int(indices[i])
                field = np.asarray(da.isel({time_name: idx}).values)
                time_val = ds[time_name].values[idx]
                draw_frame(field, f"{path.name} | {args.var or da.name} | {time_name} = {time_val}")
                return []

            anim = FuncAnimation(
                fig,
                update,
                frames=len(indices),
                interval=1000 // args.fps,
                blit=False,
            )

            base = outdir / f"{path.stem}_{args.var or da.name or 'field'}"
            save_animation(anim, base, fps=args.fps)

            png = outdir / f"{path.stem}_{args.var or da.name or 'field'}_poster.png"
            update(0)
            fig.savefig(png, dpi=240, bbox_inches="tight", facecolor=fig.get_facecolor())
            print(f"[saved] {png}")

        plt.close(fig)

    finally:
        ds.close()


# ============================================================
# ARGUMENTS / MAIN
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description="Advanced fluid dynamics visualizations.")
    p.add_argument("--mode", choices=["sim", "data"], default="sim")
    p.add_argument("--input", type=str, default=None, help="NetCDF/GRIB file for data mode.")
    p.add_argument("--var", type=str, default=None, help="Variable name to plot in data mode.")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--frames", type=int, default=360, help="Number of animation frames.")
    p.add_argument("--dt", type=float, default=0.012, help="Timestep for the Navier-Stokes simulation.")
    p.add_argument("--nu", type=float, default=2e-3, help="Viscosity for the Navier-Stokes simulation.")
    p.add_argument("--n", type=int, default=192, help="Grid resolution for the Navier-Stokes simulation.")
    p.add_argument("--n-particles", type=int, default=260)
    p.add_argument("--trail-len", type=int, default=40)
    p.add_argument("--cmap", type=str, default="turbo", help="Colormap for data mode.")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = get_output_dir()
    print(f"[output dir] {outdir}")

    if args.mode == "sim":
        run_simulation(args, outdir)
    else:
        if args.input is None:
            raise SystemExit("In --mode data, you must provide --input /path/to/file.nc or .grib")
        run_data_mode(args, outdir)


if __name__ == "__main__":
    main()