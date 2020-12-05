import argparse
import numpy as np
import trimesh
from ..mesh.remesh import remesh
from ..mesh.Mesh import Mesh

# Get the input file path.
parser = argparse.ArgumentParser(description="Remesh an obj.")
parser = argparse.ArgumentParser()
parser.add_argument(
    "--input", type=argparse.FileType("r", encoding="UTF-8"), required=True
)
parser.add_argument("--output", type=str, required=True)
args = parser.parse_args()

# Load the obj.
mesh = trimesh.exchange.obj.load_obj(args.input)
vertices = np.float32(mesh["vertices"])
faces = mesh["faces"]

# create mesh file
mesh_to_sample = Mesh(vertices, faces)

points, norms = mesh_to_sample.sample_surface(vertices, 25000)

points = points.numpy()

np.savetxt(args.output, points)
