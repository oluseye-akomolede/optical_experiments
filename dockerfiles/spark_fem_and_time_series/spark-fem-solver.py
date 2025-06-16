import os
import torch
import xml.etree.ElementTree as ET
#os.system(' pip uninstall --quiet numpy pyvista tetgen scipy -y')
#os.system(' pip install  --upgrade --quiet pyvista tetgen scipy')
#os.system(' pip install --upgrade --quiet numpy')
#os.system(' pip install --upgrade --quiet numba scikit-learn scikit-image  ')

import urllib.request
import pyvista as pv
import tetgen
import numpy as np
import torch
from scipy.sparse import coo_matrix
from pyspark.sql import SparkSession
from argparse import ArgumentParser, Namespace
import boto3
from botocore.client import Config
import psycopg2


def average_edge_length(mesh):
    edge_lengths = []
    edges = mesh.extract_all_edges()
    lines = edges.lines.reshape(-1, 3)
    for _, p0_idx, p1_idx in lines:
        p0 = edges.points[p0_idx]
        p1 = edges.points[p1_idx]
        edge_lengths.append(np.linalg.norm(p0 - p1))
    return np.mean(edge_lengths) if edge_lengths else float('inf')

def adaptive_subdivide(mesh, min_points=3000, max_edge_length=0.2, max_subdivisions=4, verbose=True):
    subdivisions = 0
    while (
        (mesh.n_points < min_points or average_edge_length(mesh) > max_edge_length)
        and subdivisions < max_subdivisions
    ):
        if verbose:
            print(f"Subdivision {subdivisions}: {mesh.n_points} points, avg edge = {average_edge_length(mesh):.4f}")
        mesh = mesh.subdivide(1, subfilter='loop')
        subdivisions += 1

    if verbose:
        print(f"Final: {mesh.n_points} points, avg edge = {average_edge_length(mesh):.4f} after {subdivisions} subdivisions")
    return mesh

#Download STL file from minio server
def download_stl(filename):
    # MinIO config
    BUCKET_NAME = "mse-meshes"
    MINIO_ENDPOINT = "http://minio.minio.svc.cluster.local:9000"
    ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
    SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
    
    s3 = boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
        region_name='us-east-1'  # MinIO doesn't use this, but boto3 expects it
    )
    # Download the STL file from MinIO
    save_path = f"/tmp/{filename}"
    url = f"{MINIO_ENDPOINT}/{BUCKET_NAME}/{filename}"
    print(f"Downloading STL file from {url} to {save_path}")
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Use urllib to download the file
    try:
        s3.download_file(BUCKET_NAME, filename, save_path)  
        return save_path
    except Exception as e:
        print(f"Error downloading file: {e}")
        return None
    
def compute_fem_matrices(tet_data):
    #This function computes stiffness and mass matrices. It is run in parallel on pyspark
    from numpy import array, dot, linalg
    A_entries = []
    B_entries = []

    for tet_indices, tet_coords in [tet_data]:
        v0, v1, v2, v3 = array(tet_coords)
        J = np.column_stack((v1 - v0, v2 - v0, v3 - v0))
        det_J = np.linalg.det(J)
        if abs(det_J) < 1e-10:
            continue

        volume = abs(det_J) / 6.0
        inv_JT = np.linalg.inv(J).T
        grad_ref = array([[-1, -1, -1],
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]])
        grads = inv_JT @ grad_ref.T
        grads = grads.T

        for i in range(4):
            for j in range(4):
                stiffness_entry = dot(grads[i], grads[j]) * volume
                A_entries.append((tet_indices[i], tet_indices[j], stiffness_entry))
                mass_entry = volume /10.0 if i == j else volume /20.0
                B_entries.append((tet_indices[i], tet_indices[j], mass_entry))

    return A_entries, B_entries

def coo_list_to_matrix(coo_list, n):
    from scipy.sparse import coo_matrix
    rows, cols, data = zip(*[(i, j, v) for ((i, j), v) in coo_list])
    return coo_matrix((data, (rows, cols)), shape=(n, n))

def create_pvd_file(output_dir: str, base_name: str, timesteps: int, dt: float) -> str:
    """
    Create a ParaView .pvd file that references all time-step .vtu files.

    Parameters:
    - output_dir: Directory where files are saved
    - base_name: Prefix for .vtu files (e.g., "impulse_mode0")
    - timesteps: Number of time steps
    - dt: Time step size

    Returns:
    - Full path to the created .pvd file
    """
    root = ET.Element("VTKFile", type="Collection", version="0.1", byte_order="LittleEndian")
    collection = ET.SubElement(root, "Collection")

    for t in range(timesteps):
        time = t * dt
        filename = f"{base_name}_t{t:03d}.vtu"
        ET.SubElement(collection, "DataSet", {
            "timestep": f"{time:.6f}",
            "group": "",
            "part": "0",
            "file": filename
        })

    pvd_path = os.path.join(output_dir, f"{base_name}.pvd")
    tree = ET.ElementTree(root)
    tree.write(pvd_path, encoding='UTF-8', xml_declaration=True)
    return pvd_path

def run_fem_solver(stl_file: str):
    TARGET_BUCKET_FEM = "mse-fem-modes"
    TARGET_BUCKET_FDTD = "mse-time-series"
    MIN_MESH_POINTS = 3000
    max_subdivisions = 4
    
    stl_file = stl_file

    # Step 1: Download STL file from MinIO
    stl_file =download_stl(stl_file)
    if stl_file is None:
        print("Failed to download STL file.")
        return
    os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-17-openjdk-amd64"
    from scipy.spatial import Delaunay

    # Step 2: Load STL file using PyVista
    surface_mesh = pv.read(stl_file)

    surface_mesh = surface_mesh.clean(tolerance=1e-5)
    surface_mesh = surface_mesh.extract_surface()
    surface_mesh = surface_mesh.triangulate()
    surface_mesh = surface_mesh.connectivity(largest=True)
    print("repaired mesh")
    #surface_mesh = surface_mesh.subdivide(nsub=4, subfilter='linear')
    surface_mesh = adaptive_subdivide(surface_mesh, min_points=MIN_MESH_POINTS, max_edge_length=0.2, max_subdivisions=max_subdivisions)

    # Step 2.1: Ensure the mesh is closed
    if not surface_mesh.is_all_triangles:
        print("Mesh is not triangulated, triangulating now...")
        surface_mesh = surface_mesh.triangulate()

    # Step 3: Generate tetrahedral mesh using TetGen
    tet = tetgen.TetGen(surface_mesh)
    tet.tetrahedralize(order=1, mindihedral=20, minratio=2.0)
    mesh = tet.grid

    # Step 4: Extract cells and points
    cells = mesh.cells_dict[10]  # VTK_TETRA = 10
    points = mesh.points

    spark = SparkSession.builder.appName("MeshFEM").getOrCreate()
    sc = spark.sparkContext

    tetra_data = []
    for tet in cells:
        coords = [points[i].tolist() for i in tet]
        tetra_data.append((tet.tolist(), coords))

    tetra_rdd = sc.parallelize(tetra_data)


    A_and_B_entries = tetra_rdd.map(compute_fem_matrices)
    A_entries_rdd = A_and_B_entries.flatMap(lambda x: x[0])
    B_entries_rdd = A_and_B_entries.flatMap(lambda x: x[1])

    A_coo = A_entries_rdd.map(lambda x: ((x[0], x[1]), x[2])) \
                        .reduceByKey(lambda x, y: x+ y) \
                        .collect()
    B_coo = B_entries_rdd.map(lambda x: ((x[0], x[1]), x[2])) \
                        .reduceByKey(lambda x, y: x + y) \
                        .collect()



    n = points.shape[0]
    A_global = coo_list_to_matrix(A_coo, n)
    B_global = coo_list_to_matrix(B_coo, n)

    # Step 1: Identify boundary nodes using PyVista
    surf = mesh.extract_surface()
    boundary_point_ids = np.unique(surf.point_data['vtkOriginalPointIds'])

    # Step 2: Apply Dirichlet BCs to stiffness and mass matrices
    from scipy.sparse import lil_matrix

    A_mod = A_global.tolil()
    B_mod = B_global.tolil()

    for idx in boundary_point_ids:
        A_mod[idx, :] = 0
        A_mod[:, idx] = 0
        A_mod[idx, idx] = 1e16   # Enforces ϕ = 0 at this node

        B_mod[idx, :] = 0
        B_mod[:, idx] = 0
        B_mod[idx, idx] = 1.0   # Optional: Makes B non-singular

    # Convert back to usable format
    A_bc = A_mod.tocoo()
    B_bc = B_mod.tocoo()

    # --- Load matrices from COO data ---
    # These should come from the PySpark aggregation step
    # Assume A_global and B_global are scipy.sparse.coo_matrix

    # Step 1: Convert to dense NumPy arrays
    A_dense_np = A_bc.toarray()
    B_dense_np = B_bc.toarray()

    # Step 2: Convert to PyTorch tensors (CPU for now)
    A = torch.tensor(A_dense_np, dtype=torch.float32)  # or float64 for precision
    B = torch.tensor(B_dense_np, dtype=torch.float32)

    # --- 💡 FUTURE: Enable this when you have GPU ---
    # A = A.to("cuda")
    # B = B.to("cuda")

    # Step 3: Solve the generalized eigenvalue problem
    # A x = λ B x
    # → Convert to standard problem: B⁻¹ A x = λ x
    B_inv = torch.linalg.inv(B)
    eigen_matrix = B_inv @ A

    # Solve (on CPU)
    eigvals, eigvecs = torch.linalg.eig(eigen_matrix)

    # Step 4: Extract real part (imaginary should be near-zero)
    eigvals = eigvals.real
    eigvecs = eigvecs.real

    # --- Sort and interpret ---
    sorted_indices = eigvals.argsort()
    eigvals_sorted = eigvals[sorted_indices]
    eigvecs_sorted = eigvecs[:, sorted_indices]

    # Step 5: Inspect top modes
    print("All eigenvalues (ω² or λ):")

    for x in eigvals_sorted.numpy():
        print(x)

    # --- 💡 FUTURE GPU Notes ---
    # Once you get your RTX 3050 server:
    # - Change all tensor initializations to `.to("cuda")`
    # - Or run this as a remote API job on the GPU server

    # Optional: Return eigenvectors if you want to plot the fields

    # Make sure your mesh and eigenvectors are ready
    output_dir = "eigenmodes"
    os.makedirs(output_dir, exist_ok=True)

    # If needed: convert PyTorch tensor to numpy
    eigvecs_np = eigvecs_sorted.cpu().numpy()

    eigenmode_type = stl_file.split('/')[-1].split('.')[0]  # Use STL filename as type
    # Export the first N modes
    N = min(100,len(eigvecs_sorted))  # you can change this
    for i in range(N):
        mode = eigvecs_np[:, i]
        mesh.point_data.clear()  # Clear previous fields
        mesh.point_data[f"mode_{i}"] = mode
        mesh.save(os.path.join(output_dir, f"{eigenmode_type}_eigenmode_{i:02d}.vtu"))

    # Now upload to MinIO
    s3 = boto3.client(
        "s3",
        endpoint_url="http://minio.minio.svc.cluster.local:9000",
        aws_access_key_id=os.getenv("MINIO_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("MINIO_SECRET_KEY"),
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
        region_name='us-east-1'  # MinIO doesn't use this, but boto3 expects it
    )

    for i in range(N):
        file_path = os.path.join(output_dir, f"{eigenmode_type}_eigenmode_{i:02d}.vtu")
        s3.upload_file(file_path, TARGET_BUCKET_FEM, f"{eigenmode_type}_eigenmode_{i:02d}.vtu")
        print(f"Uploaded {file_path} to bucket {TARGET_BUCKET_FEM}")

    # Now we update a table in the database with the results
    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT")
    )
    cursor = conn.cursor()
    for i in range(N):
        cursor.execute(f"""
            INSERT INTO mse_fem_eigenmodes (filename, fem_filename, mode_index, eigenvalue)
                       VALUES ('{args.stl}', '{eigenmode_type}_eigenmode_{i:02d}.vtu', {i}, {eigvals_sorted[i].item()})
        """)
    

    print(f"✅ Exported {N} eigenmodes to '{output_dir}/' for ParaView.")

    # Choose time steps
    timesteps = 1000
    dt = 1e-10  # seconds
    frequencies = np.sqrt(eigvals_sorted.numpy()) / (2 * np.pi)  # Hz
    omega = 2 * np.pi * frequencies

    # Pick an initial mode shape (e.g., unit excitation in mode 0)
    coeffs = np.zeros(N)
    coeffs[0] = 1.0

    response_dir = "impulse_response"
    os.makedirs(response_dir, exist_ok=True)

    for t in range(timesteps):
        time = t * dt
        field = np.zeros_like(eigvecs_np[:, 0])
        for i in range(N):
            field += coeffs[i] * np.cos(omega[i] * time) * eigvecs_np[:, i]
        mesh.point_data.clear()
        mesh.point_data[f"field_t{t:03d}"] = field
        mesh.save(os.path.join(response_dir, f"{eigenmode_type}_response_t{t:03d}.vtu"))

    pvd_path = create_pvd_file(response_dir, eigenmode_type, timesteps, dt)

    for t in range(timesteps):
        file_path = os.path.join(response_dir, f"{eigenmode_type}_response_t{t:03d}.vtu")
        s3.upload_file(file_path, TARGET_BUCKET_FDTD, f"{eigenmode_type}_response_t{t:03d}.vtu")

    # 🔁 Upload .pvd
    s3.upload_file(pvd_path, TARGET_BUCKET_FDTD, os.path.basename(pvd_path))


    cursor.execute(f"""
        INSERT INTO mse_time_series (filename, mode_index, pvd_path, dt, num_steps, created_at)
        VALUES ('{args.stl}', 0, '{pvd_path}', {dt}, {timesteps}, NOW())
    """)
    conn.commit()
    cursor.close()
    conn.close()

if __name__ == "__main__":
    parser = ArgumentParser(add_help=True)
    parser.add_argument("--stl", type=str, dest="stl",required=True)
    args = parser.parse_args()
    run_fem_solver(args.stl)