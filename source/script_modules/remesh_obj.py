import argparse
import numpy as np
import trimesh
from ..mesh.remesh import remesh

# Get the input file path.
parser = argparse.ArgumentParser(description="Remesh an obj.")
parser = argparse.ArgumentParser()
parser.add_argument(
    "--input", type=argparse.FileType("r", encoding="UTF-8"), required=True
)
parser.add_argument(
    "--output", type=argparse.FileType("w", encoding="UTF-8"), required=True
)
args = parser.parse_args()

# Load the obj.
mesh = trimesh.exchange.obj.load_obj(args.input)
vertices = mesh["vertices"]
faces = mesh["faces"]

# Remesh the obj.
remeshed_vertices, remeshed_faces = remesh(vertices, faces)

# Save an OBJ file.
mesh = trimesh.Trimesh(vertices=remeshed_vertices, faces=remeshed_faces)
obj = trimesh.exchange.obj.export_obj(mesh)
args.output.write(obj)
