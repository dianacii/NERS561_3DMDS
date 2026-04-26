"""
nem_solver.py — 1-node NEM Response Matrix Solver
===================================================

Algorithm:
    1. Precompute response matrices R(6x6) and P(6x6) per node per group
       from XS and geometry (done once, before iteration)
    2. Outer power iteration:
       a. Compute source Q (0th moment only for order=2 without TL)
       b. Inner iteration:
          - Get Ji (incoming) from neighbors' Jo (outgoing) or BC
          - Jo = R @ Ji + P @ Q   (response matrix multiply)
          - Update flux from neutron balance
       c. Update fission source
       d. Update k_eff
       e. Check convergence

Face ordering: X+, X-, Y+, Y-, Z+, Z-
    jo[k,g,0] = X+ outgoing
    jo[k,g,1] = X- outgoing
    jo[k,g,2] = Y+ outgoing
    jo[k,g,3] = Y- outgoing
    jo[k,g,4] = Z+ outgoing
    jo[k,g,5] = Z- outgoing

Boundary conditions:
    bc=0: Zero flux (ji = -jo, not physical, ignore)
    bc=1: Vacuum (ji = 0)
    bc=2: Reflective (ji = jo)
"""

import numpy as np
import argparse
from main import (Mesh, MaterialXS, cell_idx,
                  build_mesh, parse_input, parse_layers,
                  parse_materials, load_xs_library, clean_txt)


# =============================================================================
# Response matrix computation
# For 2nd order NEM without transverse leakage
# =============================================================================

def compute_response_matrices(mesh, xs_data):
    """
    Precompute response matrices R (6x6) and P (6x1) for each node and group.

    For each node k and group g:
        dx = D / delx,  lx = 1/(sigma_r * delx)
        dy = D / dely,  ly = 1/(sigma_r * dely)
        dz = D / delz,  lz = 1/(sigma_r * delz)

    Matrix A (6x6): coupling of outgoing currents
    Matrix B (6x6): coupling of incoming currents
    Matrix C (6x1): coupling of source (0th moment only)

    R = A^{-1} * B
    P = A^{-1} * C

    jo = R @ ji + P * Q0

    Returns
    -------
    R : array (N, G, 6, 6)
    P : array (N, G, 6)     [only 0th moment source column]
    """
    nx, ny, nz = mesh.material.shape
    G = next(iter(xs_data.values())).G
    N = nx*ny*nz

    R = np.zeros((N, G, 6, 6))
    P = np.zeros((N, G, 6))

    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                k   = cell_idx(ix, iy, iz, ny, nz)
                xs  = xs_data[int(mesh.material[ix, iy, iz])]
                hx  = mesh.dx[ix]
                hy  = mesh.dy[iy]
                hz  = mesh.dz[iz]

                for g in range(G):
                    D     = xs.D[g]
                    sigr  = xs.sigma_r[g]

                    dx = D / hx;  lx = 1.0 / (sigr * hx)
                    dy = D / hy;  ly = 1.0 / (sigr * hy)
                    dz = D / hz;  lz = 1.0 / (sigr * hz)

                    # ── Matrix A  ──
                    A = np.zeros((6, 6))

                    # Diagonal blocks (same direction)
                    ax  = 1.0 + 8.0*dx + 6.0*dx*lx
                    ay  = 1.0 + 8.0*dy + 6.0*dy*ly
                    az  = 1.0 + 8.0*dz + 6.0*dz*lz
                    ax1 = 4.0*dx + 6.0*dx*lx
                    ay1 = 4.0*dy + 6.0*dy*ly
                    az1 = 4.0*dz + 6.0*dz*lz

                    A[0,0] = ax;  A[1,1] = ax
                    A[2,2] = ay;  A[3,3] = ay
                    A[4,4] = az;  A[5,5] = az
                    A[0,1] = ax1; A[1,0] = ax1
                    A[2,3] = ay1; A[3,2] = ay1
                    A[4,5] = az1; A[5,4] = az1

                    # Cross terms (different directions)
                    xy = 6.0*dx*ly;  xz = 6.0*dx*lz
                    yx = 6.0*dy*lx;  yz = 6.0*dy*lz
                    zx = 6.0*dz*lx;  zy = 6.0*dz*ly

                    A[0,2]=xy; A[0,3]=xy; A[1,2]=xy; A[1,3]=xy
                    A[0,4]=xz; A[0,5]=xz; A[1,4]=xz; A[1,5]=xz
                    A[2,0]=yx; A[2,1]=yx; A[3,0]=yx; A[3,1]=yx
                    A[2,4]=yz; A[2,5]=yz; A[3,4]=yz; A[3,5]=yz
                    A[4,0]=zx; A[4,1]=zx; A[5,0]=zx; A[5,1]=zx
                    A[4,2]=zy; A[4,3]=zy; A[5,2]=zy; A[5,3]=zy

                    # ── Matrix B (same as A but with negative terms) ──────
                    B = A.copy()
                    bx  = 1.0 - 8.0*dx + 6.0*dx*lx
                    by  = 1.0 - 8.0*dy + 6.0*dy*ly
                    bz  = 1.0 - 8.0*dz + 6.0*dz*lz
                    bx1 = -4.0*dx + 6.0*dx*lx
                    by1 = -4.0*dy + 6.0*dy*ly
                    bz1 = -4.0*dz + 6.0*dz*lz

                    B[0,0] = bx;  B[1,1] = bx
                    B[2,2] = by;  B[3,3] = by
                    B[4,4] = bz;  B[5,5] = bz
                    B[0,1] = bx1; B[1,0] = bx1
                    B[2,3] = by1; B[3,2] = by1
                    B[4,5] = bz1; B[5,4] = bz1

                    # ── Matrix C (source coupling, 0th moment only) ───────
                    # C[i,0] = source coupling for 0th moment
                    C0 = np.zeros(6)
                    ax1c = 6.0*dx*lx*hx
                    ay1c = 6.0*dy*ly*hy
                    az1c = 6.0*dz*lz*hz

                    C0[0] = ax1c; C0[1] = ax1c
                    C0[2] = ay1c; C0[3] = ay1c
                    C0[4] = az1c; C0[5] = az1c

                    # ── Compute R = A^{-1} B, P = A^{-1} C0 ──────────────
                    try:
                        Ainv = np.linalg.inv(A)
                    except np.linalg.LinAlgError:
                        print(f"WARNING: singular A at node ({ix},{iy},{iz}) g={g}")
                        Ainv = np.eye(6)

                    R[k, g] = Ainv @ B
                    P[k, g] = Ainv @ C0

    return R, P


# =============================================================================
# Boundary condition: get incoming current from neighbor or BC
# =============================================================================

def get_incoming(jo, k, g, face, neighbor_k, neighbor_face, bc_type):
    """
    Get incoming partial current at face of node k, group g.

    For interior nodes: ji = jo of neighbor at opposite face
    For boundary nodes:
        bc_type=1 (vacuum):     ji = 0
        bc_type=2 (reflective): ji = jo[k,g,face]

    face ordering: 0=X+, 1=X-, 2=Y+, 3=Y-, 4=Z+, 5=Z-
    """
    if neighbor_k is None:
        # Boundary
        if bc_type == 1:    # vacuum
            return 0.0
        else:               # reflective
            return jo[k, g, face]
    else:
        return jo[neighbor_k, g, neighbor_face]


# =============================================================================
# Fission source
# =============================================================================

def compute_fission_source(mesh, xs_data, f0):
    """
    fs[k] = sum_g nu_sigma_f[g] * f0[k,g]
    F_total = sum_k fs[k] * V[k]
    """
    nx, ny, nz = mesh.material.shape
    G = next(iter(xs_data.values())).G
    N = nx*ny*nz

    fs = np.zeros(N)
    F_total = 0.0

    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                k   = cell_idx(ix, iy, iz, ny, nz)
                xs  = xs_data[int(mesh.material[ix, iy, iz])]
                V   = mesh.dx[ix]*mesh.dy[iy]*mesh.dz[iz]
                for g in range(G):
                    c = xs.nu_sigma_f[g] * f0[k, g]
                    fs[k] += c
                    F_total += c * V

    return fs, F_total


# =============================================================================
# Source Q (0th moment: fission/k + in-scatter)
# =============================================================================

def compute_source(mesh, xs_data, f0, fs, keff):
    """
    Q[k,g] = chi[g]*fs[k]/keff + sum_{g'!=g} sigma_s[g',g]*f0[k,g']

    This is the 0th moment of the source.
    """
    nx, ny, nz = mesh.material.shape
    G = next(iter(xs_data.values())).G
    N = nx*ny*nz

    Q = np.zeros((N, G))

    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                k  = cell_idx(ix, iy, iz, ny, nz)
                xs = xs_data[int(mesh.material[ix, iy, iz])]
                for g in range(G):
                    # Fission
                    Q[k, g] = xs.chi[g] * fs[k] / keff
                    # In-scatter from other groups
                    for gp in range(G):
                        if gp != g:
                            Q[k, g] += xs.sigma_s[gp, g] * f0[k, gp]

    return Q


# =============================================================================
# Update flux from currents
# =============================================================================

def update_flux(mesh, xs_data, f0, jo, ji, Q):
    nx, ny, nz = mesh.material.shape
    G = next(iter(xs_data.values())).G
    N = nx*ny*nz

    # Build sigma_r and node dimensions arrays (vectorized)
    sigr = np.zeros((N, G))
    hx_arr = np.zeros(N)
    hy_arr = np.zeros(N)
    hz_arr = np.zeros(N)

    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                k  = cell_idx(ix, iy, iz, ny, nz)
                xs = xs_data[int(mesh.material[ix, iy, iz])]
                sigr[k, :] = xs.sigma_r
                hx_arr[k]  = mesh.dx[ix]
                hy_arr[k]  = mesh.dy[iy]
                hz_arr[k]  = mesh.dz[iz]

    # Compute L0 for all nodes and groups at once
    L0x = (jo[:,:,0] - ji[:,:,0]) + (jo[:,:,1] - ji[:,:,1])  # (N,G)
    L0y = (jo[:,:,2] - ji[:,:,2]) + (jo[:,:,3] - ji[:,:,3])
    L0z = (jo[:,:,4] - ji[:,:,4]) + (jo[:,:,5] - ji[:,:,5])

    # Neutron balance: phi = (Q - L0x/hx - L0y/hy - L0z/hz) / sigr
    hx_arr = hx_arr[:,np.newaxis]  # (N,1) for broadcasting
    hy_arr = hy_arr[:,np.newaxis]
    hz_arr = hz_arr[:,np.newaxis]

    phi_new = (Q - L0x/hx_arr - L0y/hy_arr - L0z/hz_arr) / sigr

    # Only update where result is positive and sigr > 0
    valid = (sigr > 1e-20) & (phi_new > 0)
    f0_new = f0.copy()
    f0_new[valid] = phi_new[valid]

    return f0_new


# =============================================================================
# Inner iteration sweep
# =============================================================================

def inner_sweep(mesh, xs_data, f0, jo, ji, Q, R, P, bc, n_inner=2):
    """
    Inner iteration: update Jo using response matrix, then update flux.
 
    Uses red-black (checkerboard) ordering:
        - Color each node by (ix+iy+iz) % 2
        - Update all red nodes (color=0) first, then all black nodes (color=1)
        - Red nodes only have black neighbors (unchanged this pass) so all
          incoming currents are from the previous sweep, no ordering dependency
        - Then black nodes use freshly updated red neighbors
 
    jo = R @ ji + P * Q0
 
    bc dict: {face: bc_type}
             bc_type: 1=vacuum, 2=reflective
    """
    nx, ny, nz = mesh.material.shape
    G = next(iter(xs_data.values())).G
 
    # Neighbor lookup: returns (neighbor_k, neighbor_face, bc_type)
    def neighbor(ix, iy, iz, face):
        if face == 0:   # X+
            if ix == nx-1: return None, None, bc['xhi']
            return cell_idx(ix+1,iy,iz,ny,nz), 1, None
        if face == 1:   # X-
            if ix == 0:    return None, None, bc['xlo']
            return cell_idx(ix-1,iy,iz,ny,nz), 0, None
        if face == 2:   # Y+
            if iy == ny-1: return None, None, bc['yhi']
            return cell_idx(ix,iy+1,iz,ny,nz), 3, None
        if face == 3:   # Y-
            if iy == 0:    return None, None, bc['ylo']
            return cell_idx(ix,iy-1,iz,ny,nz), 2, None
        if face == 4:   # Z+
            if iz == nz-1: return None, None, bc['zhi']
            return cell_idx(ix,iy,iz+1,ny,nz), 5, None
        if face == 5:   # Z-
            if iz == 0:    return None, None, bc['zlo']
            return cell_idx(ix,iy,iz-1,ny,nz), 4, None

    red_nodes   = [(cell_idx(ix,iy,iz,ny,nz), ix, iy, iz)
               for ix in range(nx) for iy in range(ny) for iz in range(nz)
               if (ix+iy+iz) % 2 == 0]
    black_nodes = [(cell_idx(ix,iy,iz,ny,nz), ix, iy, iz)
               for ix in range(nx) for iy in range(ny) for iz in range(nz)
               if (ix+iy+iz) % 2 == 1]
    red_ks   = np.array([k for k,_,_,_ in red_nodes])
    black_ks = np.array([k for k,_,_,_ in black_nodes])

    N = nx*ny*nz
    neighbor_idx  = np.full((N, 6), -1, dtype=int)
    neighbor_face = np.full((N, 6), -1, dtype=int)
    bc_face       = np.zeros((N, 6), dtype=int)
    
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                k = cell_idx(ix,iy,iz,ny,nz)
                for face in range(6):
                    nk, nf, bc_t = neighbor(ix,iy,iz,face)
                    if nk is None:
                        bc_face[k,face] = bc_t
                    else:
                        neighbor_idx[k,face]  = nk
                        neighbor_face[k,face] = nf
                        
    for _ in range(n_inner):
        for nodes, ks in [(red_nodes, red_ks), (black_nodes, black_ks)]:
            # Vectorized ji update
            for face in range(6):
                nks  = neighbor_idx[ks, face]
                nfs  = neighbor_face[ks, face]
                bcs  = bc_face[ks, face]
                is_interior = nks >= 0
    
                # Interior nodes: ji = jo of neighbor at opposite face
                interior = np.where(is_interior)[0]
                if len(interior) > 0:
                    ji[ks[interior], :, face] = jo[nks[interior], :, nfs[interior]]
    
                # Boundary nodes
                boundary = np.where(~is_interior)[0]
                for b in boundary:
                    k_b = ks[b]
                    if bcs[b] == 1:   # vacuum
                        ji[k_b, :, face] = 0.0
                    else:              # reflective
                        ji[k_b, :, face] = jo[k_b, :, face]
    
            # Vectorized response matrix multiply
            jo[ks] = (np.einsum('kgij,kgj->kgi', R[ks], ji[ks])
                      + P[ks] * Q[ks][:,:,np.newaxis])
    
        # Update flux after full red-black pass
        f0 = update_flux(mesh, xs_data, f0, jo, ji, Q)
 
    return f0, jo, ji


# =============================================================================
# NEM power iteration
# =============================================================================

def nem_power_iteration(mesh, xs_data, variables,
                        k_tol=1e-5, phi_tol=1e-5,
                        max_outer=500, n_inner=2):
    """
    Outer power iteration for NEM.

    Based on Outer subroutine in NEM.f90.
    """
    nx, ny, nz = mesh.material.shape
    G = next(iter(xs_data.values())).G
    N = nx*ny*nz

    # Boundary conditions: 1=vacuum, 2=reflective
    def to_bc(s):
        return 2 if s == 'r' else 1

    bc = {
        'xlo': to_bc(variables['x_min']),
        'xhi': to_bc(variables['x_max']),
        'ylo': to_bc(variables['y_min']),
        'yhi': to_bc(variables['y_max']),
        'zlo': to_bc(variables['z_min']),
        'zhi': to_bc(variables['z_max']),
    }

    # Precompute response matrices (done once)
    print("Precomputing response matrices...")
    R, P = compute_response_matrices(mesh, xs_data)
    print(f"  R shape: {R.shape}, P shape: {P.shape}")

    # Initialize
    f0 = np.ones((N, G))    # node-averaged flux
    jo = np.ones((N, G, 6)) # outgoing partial currents
    ji = np.ones((N, G, 6)) # incoming partial currents
    keff = 1.0

    print("Starting NEM power iteration...")
    print(f"  {'Iter':>4}  {'k':>10}  {'dk':>10}  {'df':>10}")

    for o in range(max_outer):

        f0_old = f0.copy()
        fs, F_old = compute_fission_source(mesh, xs_data, f0)

        # Loop over groups
        for g in range(G):
            # Compute total source Q for this group
            Q = compute_source(mesh, xs_data, f0, fs, keff)

            # Inner iteration
            f0, jo, ji = inner_sweep(
                mesh, xs_data, f0, jo, ji, Q, R, P, bc, n_inner)

        # Update fission source and k
        fs_new, F_new = compute_fission_source(mesh, xs_data, f0)
        keff_new = keff * F_new / F_old if F_old > 1e-20 else keff

        # Convergence check
        k_err  = abs(keff_new - keff)
        f0_err = np.max(np.abs(f0 - f0_old)) / max(f0_old.max(), 1e-20)

        print(f"  {o+1:>4}  {keff_new:>10.6f}  {k_err:>10.2e}  {f0_err:>10.2e}")

        if o > 0 and k_err < k_tol and f0_err < phi_tol:
            print(f"\nNEM converged in {o+1} iterations: k_eff = {keff_new:.6f}")
            # Normalize flux
            f0 /= f0.max()
            return f0, keff_new

        keff = keff_new

    print(f"\nWARNING: did not converge. k={keff:.6f}")
    return f0, keff


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)
    args = parser.parse_args()
    print(f"NEM Solver — input: {args.input_file}")
    print("="*50)

    with open(args.input_file,'r') as f:
        raw = f.read()
    input_data = clean_txt(raw)
    variables  = parse_input(input_data)

    # NEM requires 1 cell per region
    if any(variables[f"{d}_cells_per_region"] != 1
           for d in ['x','y','z']):
        print("WARNING: NEM requires 1 cell per region. Overriding.")
        for d in ['x','y','z']:
            variables[f"{d}_cells_per_region"] = 1

    layers    = parse_layers(input_data, variables)
    mat_names = parse_materials(input_data)
    mesh      = build_mesh(variables, layers)
    nx,ny,nz  = mesh.material.shape
    print(f"Mesh: {nx} x {ny} x {nz} nodes ({nx*ny*nz} total)")

    xs_library = load_xs_library(variables["XS_LIBRARY"])
    xs_data = {}
    for mat_id, mat_name in mat_names.items():
        key = mat_name.lower()
        if key not in xs_library:
            print(f"ERROR: '{mat_name}' not in XS library"); exit(1)
        xs = xs_library[key]
        xs.mat_id = mat_id
        xs_data[mat_id] = xs

    G = next(iter(xs_data.values())).G
    print(f"XS: {len(xs_data)} materials, {G} groups")
    print("="*50)

    f0, keff = nem_power_iteration(mesh, xs_data, variables)
    print(f"\nFinal k_eff = {keff:.6f}")
    print(f"FD reference  (18x18x19):  ~1.027463")
    print(f"Benchmark ref (9x9x10):  ~1.031760")
    print(f"Benchmark ref (17x17x19):  ~1.02913")
