import argparse
from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp
import re, io
import pandas as pd

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Mesh:
    material: np.ndarray  # 3D array of material IDs, shape (nx, ny, nz)
    dx: np.ndarray        # 1D array of cell widths in x, shape (nx,)
    dy: np.ndarray        # 1D array of cell widths in y, shape (ny,)
    dz: np.ndarray        # 1D array of cell widths in z, shape (nz,)


@dataclass
class MaterialXS:
    """
    Macroscopic multigroup cross sections for one material.
    Units: D in cm, all sigma in cm^-1.
    sigma_s[g_from, g_to] = scattering XS from group g_from into group g_to.
    """
    mat_id:     int
    G:          int
    D:          np.ndarray   # diffusion coefficients,  shape (G,)
    sigma_a:    np.ndarray   # absorption XS,           shape (G,)
    nu_sigma_f: np.ndarray   # nu * fission XS,         shape (G,)
    chi:        np.ndarray   # fission spectrum,        shape (G,)
    sigma_s:    np.ndarray   # scattering matrix,       shape (G, G)

    @property
    def sigma_t(self):
        """
        Total XS for each group:
            sigma_t[g] = sigma_a[g] + sum_{all g'} sigma_s[g, g']
            a_ii = Sigma_t - (D_tilde_{i+1/2} + D_tilde_{i-1/2})
        """
        return self.sigma_a + self.sigma_s.sum(axis=1)

# =============================================================================
# INPUT PARSING
# =============================================================================

def clean_txt(text):
    """Remove comments and blank lines from input file text."""
    text = re.sub(r'#.*', '', text)
    text = re.sub(r'\n\s*\n', '\n', text)
    return text.strip()


def read_variable(input_data, name, var_type=str):
    match = re.search(fr"^{name}\s+(.+)", input_data, re.MULTILINE)
    if match:
        value = match.group(1).strip()
        if var_type == str:
            return value.strip('"')
        elif var_type == int:
            return int(value)
        elif var_type == np.ndarray:
            return np.array(list(map(float, value.split())))
        else:
            print(f"ERROR! Unsupported variable type for '{name}'")
            exit(1)
    else:
        print(f"ERROR! '{name}' not found in input file")
        exit(1)


def parse_input(input_data):
    """Parse all scalar variables and boundary conditions from cleaned input."""
    variables = {
        "XS_LIBRARY":          read_variable(input_data, "XS_LIBRARY",          str),
        "assemblies":          read_variable(input_data, "assemblies",          int),
        "symmetry":            read_variable(input_data, "symmetry",            int),
        "x_planes":            read_variable(input_data, "x_planes",            np.ndarray),
        "y_planes":            read_variable(input_data, "y_planes",            np.ndarray),
        "z_planes":            read_variable(input_data, "z_planes",            np.ndarray),
        "x_min":               read_variable(input_data, "x_min",               str),
        "x_max":               read_variable(input_data, "x_max",               str),
        "y_min":               read_variable(input_data, "y_min",               str),
        "y_max":               read_variable(input_data, "y_max",               str),
        "z_min":               read_variable(input_data, "z_min",               str),
        "z_max":               read_variable(input_data, "z_max",               str),
        "x_cells_per_region":  read_variable(input_data, "x_cells_per_region",  int),
        "y_cells_per_region":  read_variable(input_data, "y_cells_per_region",  int),
        "z_cells_per_region":  read_variable(input_data, "z_cells_per_region",  int),
    }
    return variables


def parse_layers(input_data, variables):
    """Parse LAYER blocks and return a 3D array of material IDs (nx, ny, nz)."""
    layer_blocks = re.findall(r"LAYER\s+\d+\s+((?:[\d ]+\n?)+)", input_data)
    layers = []
    for block in layer_blocks:
        grid = []
        for row in block.strip().splitlines():
            grid.append(list(map(int, row.split())))
        layers.append(np.array(grid))

    n_z_regions = len(variables["z_planes"]) - 1
    if len(layers) != n_z_regions:
        print(f"ERROR! Found {len(layers)} LAYER blocks but expected {n_z_regions}")
        exit(1)

    return np.stack(layers, axis=-1)


def parse_materials(input_data):
    """Parse BEGIN MATERIALS block, return dict {mat_id: name}."""
    mat_block = re.search(r"BEGIN MATERIALS(.+?)END MATERIALS", input_data, re.DOTALL)
    if not mat_block:
        print("ERROR! MATERIALS block not found in input file")
        exit(1)
    materials = {}
    for line in mat_block.group(1).strip().splitlines():
        parts = line.split()
        if parts:
            materials[int(parts[1])] = parts[0]
    return materials


# ============================================================================
# CROSS SECTION DATA PARSING
# =============================================================================

def load_xs_library(file_path):
    """Load XS data, return dict {material_name: MaterialXS}."""
    with open(file_path, 'r') as f:
        content = f.read().replace('\r\n', '\n').replace('\r', '\n')
    blocks = content.split('---')


    xs_library = {}

    for block in blocks:
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if not lines:
            continue
    
        # First line is "MATERIAL_NAME"
        mat_name = lines[0].strip().lower()  # e.g. "axial_reflector"

        split_idx = lines.index('SCATTER')
        group_data   = "\n".join(lines[1:split_idx])
        scatter_data = "\n".join(lines[split_idx + 1:])

        df_g = pd.read_csv(io.StringIO(group_data))
        df_s = pd.read_csv(io.StringIO(scatter_data))

        G = len(df_g)
        sigma_s = np.zeros((G, G))
        for _, row in df_s.iterrows():
            sigma_s[int(row['from']) - 1, int(row['to']) - 1] = row['sigma_s']

        xs_library[mat_name] = MaterialXS(
            mat_id     = -1,   # resolved when called in main with mat_id from input file
            G          = G,
            D          = df_g['D'].values,
            sigma_a    = df_g['sigma_a'].values,
            nu_sigma_f = df_g['nu_sigma_f'].values,
            chi        = df_g['chi'].values,
            sigma_s    = sigma_s,
        )
    return xs_library

# =============================================================================
# MESH SETUP
# =============================================================================

def build_mesh(variables, layers):
    """
    Expand the coarse region layout into a fine mesh.
    Each region is subdivided into x/y/z_cells_per_region fine cells.
    Returns a Mesh object.
    """
    nx_cpr = variables["x_cells_per_region"]
    ny_cpr = variables["y_cells_per_region"]
    nz_cpr = variables["z_cells_per_region"]

    nx_regions = len(variables["x_planes"]) - 1
    ny_regions = len(variables["y_planes"]) - 1
    nz_regions = len(variables["z_planes"]) - 1

    mat_n_x = nx_regions * nx_cpr
    mat_n_y = ny_regions * ny_cpr
    mat_n_z = nz_regions * nz_cpr

    materials_in_mesh = np.zeros((mat_n_x, mat_n_y, mat_n_z), dtype=int)
    for x in range(mat_n_x):
        for y in range(mat_n_y):
            for z in range(mat_n_z):
                rx = x // nx_cpr
                ry = y // ny_cpr
                rz = z // nz_cpr
                materials_in_mesh[x, y, z] = layers[rx, ry, rz]

    dx = np.repeat(np.diff(variables["x_planes"]) / nx_cpr, nx_cpr)
    dy = np.repeat(np.diff(variables["y_planes"]) / ny_cpr, ny_cpr)
    dz = np.repeat(np.diff(variables["z_planes"]) / nz_cpr, nz_cpr)

    return Mesh(material=materials_in_mesh, dx=dx, dy=dy, dz=dz)


# =============================================================================
# MATRIX ASSEMBLY 
# =============================================================================

def cell_idx(ix, iy, iz, ny, nz):
    """
    Flat cell index from 3D indices. x is slowest varying, z is fastest.
    Matches mesh.material[ix, iy, iz] which has shape (nx, ny, nz).
    """
    return ix * (ny * nz) + iy * nz + iz


def conductance(D_i, h_i, D_j, h_j):
    """
    Harmonic-mean diffusion conductance between adjacent cells i and j.
        D_tilde_{i+1/2} = 2 * D_i * D_j / (D_i * h_j + D_j * h_i)
    """
    return 2.0 * D_i * D_j / (D_i * h_j + D_j * h_i)


def assemble_A(mesh, xs_data, variables):
    """
    Assemble A to later solve the linear system A * phi = b.

      A matrix (7-stripe in 3D):
        a_{ii}   = sigma_t_{g,i} * V_i + sum of D_tilde over all 6 faces
        a_{i,j}  = -D_tilde_{face between i and j}

    Parameters
    ----------
    mesh      : Mesh object
    xs_data   : dict {mat_id: MaterialXS}
    variables : parsed input variables dict

    Returns
    -------
    A : scipy CSR sparse matrix, shape (N*G, N*G)
    G : int — number of energy groups
    """

    nx, ny, nz = mesh.material.shape
    N = nx * ny * nz                        # Number of cells in mesh
    G = next(iter(xs_data.values())).G      # Energy groups
    size = N * G                            # Total number of unknowns

    A = sp.lil_matrix((size, size))

    bc = {
        'xmin': variables['x_min'], 'xmax': variables['x_max'],
        'ymin': variables['y_min'], 'ymax': variables['y_max'],
        'zmin': variables['z_min'], 'zmax': variables['z_max'],
    }

    # Each neighbor: (delta_ix, delta_iy, delta_iz, bc_face_name)
    neighbors = [
        (-1,  0,  0, 'xmin'), (+1,  0,  0, 'xmax'),
        ( 0, -1,  0, 'ymin'), ( 0, +1,  0, 'ymax'),
        ( 0,  0, -1, 'zmin'), ( 0,  0, +1, 'zmax'),
    ]

    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):

                i   = cell_idx(ix, iy, iz, ny, nz)
                xs  = xs_data[int(mesh.material[ix, iy, iz])]
                h   = {'x': mesh.dx[ix], 'y': mesh.dy[iy], 'z': mesh.dz[iz]} # Delta x/y/z for cell i
                V   = mesh.dx[ix] * mesh.dy[iy] * mesh.dz[iz]    # Volume of cell i (differential cube)

                for g in range(G):
                    row = i * G + g
                    D_i = xs.D[g]

                    # Diagonal: sigma_t * V
                    # a_ii = Sigma_t - (D_tilde sums)
                    # The D_tilde terms are added below in the neighbor loop
                    A[row, row] += xs.sigma_t[g] * V

                    # Diffusion coupling to all 6 neighbors
                    for (dix, diy, diz, face) in neighbors:
                        jx, jy, jz = ix + dix, iy + diy, iz + diz

                        if dix != 0:
                            direction = 'x'
                            A_face = mesh.dy[iy] * mesh.dz[iz]
                        elif diy != 0:
                            direction = 'y'
                            A_face = mesh.dx[ix] * mesh.dz[iz]
                        else:
                            direction = 'z'
                            A_face = mesh.dx[ix] * mesh.dy[iy]

                        out_of_bounds = (jx < 0 or jx >= nx or
                                         jy < 0 or jy >= ny or
                                         jz < 0 or jz >= nz)

                        if out_of_bounds: # at the end of mesh, apply BC
                            if bc[face] == 'v':
                                # Vacuum BC: adds D/(0.5*h) * A_face to diagonal
                                A[row, row] += (D_i / (0.5 * h[direction])) * A_face
                                # Reflective BC: zero net current, no contribution

                        else:
                            # Interior face: harmonic-mean coupling
                            # a_{i,j} = -D_tilde
                            j    = cell_idx(jx, jy, jz, ny, nz)
                            xs_j = xs_data[int(mesh.material[jx, jy, jz])]
                            D_j  = xs_j.D[g]
                            h_j  = {'x': mesh.dx[jx],
                                    'y': mesh.dy[jy],
                                    'z': mesh.dz[jz]}[direction]

                            coupling = conductance(D_i, h[direction], D_j, h_j) * A_face
                            col = j * G + g
                            A[row, row] += coupling   # +D_tilde on diagonal
                            A[row, col]  -= coupling  # -D_tilde off-diagonal

    return A.tocsr()


def compute_source(mesh, xs_data, phi, k_guess=1.0): # Source vector b = Q                    
    """
    Compute the source term, b.

      b vector (source Q, right-hand side):
        b_{g,i} = sum_{g'!=g} sigma_s_{g'->g,i} * V_i * phi_{g',i}  (in-scatter)
                + chi_g / k * sum_{g'} nu_sigma_f_{g',i} * V_i * phi_{g',i}  (fission)

    Since phi is unknown, b is built using phi=1 as an initial flat flux guess.
    
    To solve fine-mesh finite difference eqns, we use power iteration.
        - b depends on phi, so it is updated for each iteration.

    Parameters
    ----------
    mesh      : Mesh object
    xs_data   : dict {mat_id: MaterialXS}
    variables : parsed input variables dict
    k         : multiplication factor (default 1.0 for critical reactor)

    Returns
    -------
    b : numpy array, shape (N*G,)
    F : scalar. Fission source integrated over all cells (volume), used for updating k in power iteration
    """

    nx, ny, nz = mesh.material.shape
    N = nx * ny * nz                        # Number of cells in mesh
    G = next(iter(xs_data.values())).G      # Energy groups
    size = N * G                            # Total number of unknowns

    b = np.zeros(size)
    F = 0.0
    k = k_guess

    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):

                i   = cell_idx(ix, iy, iz, ny, nz)
                xs  = xs_data[int(mesh.material[ix, iy, iz])]
                h   = {'x': mesh.dx[ix], 'y': mesh.dy[iy], 'z': mesh.dz[iz]} # Delta x/y/z for cell i
                V   = mesh.dx[ix] * mesh.dy[iy] * mesh.dz[iz]    # Volume of cell i (differential cube)

                for group in range(G):
                    row_idx = i * G + group
                    # In-scatter from other groups g' -> g
                    for gp in range(G):
                        if gp != group:
                            b[row_idx] += xs.sigma_s[gp, group] * V * phi[i * G + gp]

                    # Fission: chi_g/k * sum_{g'} nu_sigma_f_{g'} * phi_{g'}
                    for gp in range(G):
                        b[row_idx] += (xs.chi[group] / k) * xs.nu_sigma_f[gp] * V * phi[i * G + gp]
                        F += xs.nu_sigma_f[gp] * V * phi[i * G + gp] # accumulate total fission source over all cells
                    
    return b, F


# =============================================================================
# Solve 3D fine mesh finite difference equations with power iteration for A* phi = b
# =============================================================================

def power_iteration(A, mesh, xs_data, phi_guess, k_guess=1.0, k_tol=1e-6, max_iter=500):
    # Initial guess
    phi_l = phi_guess   # phi_{l}
    k_l = k_guess       # k_{l}

    # Iteration loop
    for i in range(max_iter):
        # Update source term b with new flux guess
        b, F_l = compute_source(mesh, xs_data, phi_l, k_l)

        # Solve A * phi_new = b
        phi_l_1 = sp.linalg.spsolve(A, b) # phi_{l+1}

        # Update k estimate
        _, F_l_1 = compute_source(mesh, xs_data, phi_l_1, k_l) # F_{l+1}
        k_l_1 = k_l * (F_l_1 / F_l) # k = n's from fission {l+1} / n's from fission {l}

       # Normalize flux
        #phi_new /= np.linalg.norm(phi_new)
        phi_l_1 /= phi_l_1.max()

        # Check convergence
        if abs(k_l_1 - k_l) / abs(k_l) < k_tol:
            print(f"Power iteration converged in {i+1} iterations: k={k_l_1:.6f}")
            break

        phi_l = phi_l_1
        k_l = k_l_1

        print(f"  Iter {i+1}: k={k_l_1:.6f}")
    return phi_l_1, k_l_1

# =============================================================================
# Include a 1-node NEM kernel
# =============================================================================


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    
    #--------------------------------------------------------------------------
    # Parse input and build mesh
    #--------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)
    args = parser.parse_args()
    print(f"Using input file: {args.input_file}")

    # Read and clean input file
    with open(args.input_file, 'r') as f:
        raw = f.read()
    #print(f"Input data:\n{raw}")
    input_data = clean_txt(raw)

    # Parse input
    variables = parse_input(input_data)
    print("----------")
    print(f"Input variables: {variables}")

    layers = parse_layers(input_data, variables)
    print("----------")
    print(f"Layers shape (nx_regions, ny_regions, nz_regions): {layers.shape}") # number of regions in each direction (x,y,z)

    mat_names = parse_materials(input_data)
    print("----------")
    print(f"Materials: {mat_names}")

    # Build mesh
    mesh = build_mesh(variables, layers)
    print("----------")
    print(f"Fine mesh shape (nx, ny, nz): {mesh.material.shape}") # i.e. number of cells (n) in each region (x,y,z)
    # print(f"dx: {mesh.dx}")
    # print(f"dy: {mesh.dy}")
    # print(f"dz: {mesh.dz}")
    # print(f"First layer (z=0):\n{mesh.material[:,:,0]}")
    # print(f"Second layer (z=4):\n{mesh.material[:,:,-1]}")

    #--------------------------------------------------------------------------
    # Load XS data  
    #--------------------------------------------------------------------------
    xs_library_by_name = load_xs_library(variables["XS_LIBRARY"])
    # mat_names = {1: 'axial_reflector', 2: 'radial_reflector', 3: 'fuel_type1', ...}
    xs_data = {}
    for mat_id, mat_name in mat_names.items():
        key = mat_name.lower()
        if key not in xs_library_by_name:
            print(f"ERROR: material '{mat_name}' not found in XS library")
            exit(1)
        xs = xs_library_by_name[key]
        xs.mat_id = mat_id
        xs_data[mat_id] = xs
    G = next(iter(xs_data.values())).G # Number of energy groups
    print("----------")
    print(f"Loaded XS for {len(xs_data)} materials, {G} groups")
    print("----------")

    #--------------------------------------------------------------------------
    print("Assembling Finite Difference system A*phi = b ...")
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    # Assemble matrix A
    #--------------------------------------------------------------------------
    nx, ny, nz = mesh.material.shape
    N = nx * ny * nz        # Number of cells in mesh
    size = N * G

    A = assemble_A(mesh, xs_data, variables)

    print(f"  System size: {N} cells x {G} groups = {N*G} unknowns")
    print(f"\n  A: shape={A.shape}, nonzeros={A.nnz}")    

    # For small problems, dense view for inspection
    if N * G <= 100:
        print("\nA (dense view):")
        print(A.toarray().round(4))

    # diagonal of A should be strictly positive
    if np.any(A.diagonal() <= 0):
        print("WARNING: A has non-positive diagonal entries — check XS or BC setup")
    else:
        print("  Diagonal of A is all positive (good)")

    #--------------------------------------------------------------------------
    # Compute source vector b
    #--------------------------------------------------------------------------
    # Flat unit flux guess for computing Q -- TODO: Make sure flat flux is correct
    phi_guess = np.ones(size) 

    # if only forming matrix:
    b, F = compute_source(mesh=mesh, xs_data=xs_data, phi=phi_guess)

    print(f"\n  b: shape={b.shape}, nonzero entries={np.count_nonzero(b)}")
    
    if N * G <= 100:     # For small problems, dense view for inspection
        print("\nb:")
        print(b.round(6))

    #--------------------------------------------------------------------------
    # Solve 3D fine mesh finite difference equations with power iteration
    #--------------------------------------------------------------------------
    print("\nSolving A*phi = b with power iteration...")

    phi, k = power_iteration(A=A, mesh=mesh, xs_data=xs_data, phi_guess=phi_guess)

    # print(f"  phi: shape={phi.shape}, nonzero entries={np.count_nonzero(phi)}")
    # print(f"  k_eff={k:.6f}")

    # if solving NEM: