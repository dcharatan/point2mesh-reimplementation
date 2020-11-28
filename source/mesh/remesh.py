import os
import uuid
import trimesh
import numpy as np

MANIFOLD_SOFTWARE_DIR = "Manifold/build"


def remesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    num_faces=2000,
):
    # Write the original mesh as OBJ.
    original_file = random_file_name("obj")
    with open(original_file, "w") as f:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        f.write(trimesh.exchange.obj.export_obj(mesh))

    # Create a manifold of the original file.
    manifold_file = random_file_name("obj")
    manifold_script_path = os.path.join(MANIFOLD_SOFTWARE_DIR, "manifold")
    cmd = f"{manifold_script_path} {original_file} {manifold_file}"
    os.system(cmd)

    # Simplify the manifold.
    simplified_file = random_file_name("obj")
    simplify_script_path = os.path.join(MANIFOLD_SOFTWARE_DIR, "simplify")
    cmd = (
        f"{simplify_script_path} -i {manifold_file} -o {simplified_file} -f {num_faces}"
    )
    os.system(cmd)

    # Read the simplified manifold.
    with open(simplified_file, "r") as f:
        mesh = trimesh.exchange.obj.load_obj(f)

    # Prevent file spam.
    os.remove(original_file)
    os.remove(manifold_file)
    os.remove(simplified_file)

    return mesh["vertices"], mesh["faces"]


def random_file_name(ext, prefix="tmp_"):
    return f"{prefix}{uuid.uuid4()}.{ext}"