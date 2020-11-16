import argparse
import numpy as np
import trimesh

# Get the input file path.
parser = argparse.ArgumentParser(description="Create a convex hull.")
parser = argparse.ArgumentParser()
parser.add_argument(
    "--input", type=argparse.FileType("r", encoding="UTF-8"), required=True
)
parser.add_argument(
    "--output", type=argparse.FileType("w", encoding="UTF-8"), required=True
)
args = parser.parse_args()

# Load the points into NumPy.
points = np.loadtxt(args.input)
convex_hull = trimesh.convex.convex_hull(points[:, :3])

# Save an OBJ file.
obj = trimesh.exchange.obj.export_obj(convex_hull)
args.output.write(obj)
