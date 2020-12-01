import trimesh
import numpy as np
from typing import Tuple


class Obj:
    """This just makes it easier to save and load obj files."""

    @staticmethod
    def save(file_name: str, vertices: np.ndarray, faces: np.ndarray) -> None:
        with open(file_name, "w") as f:
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            f.write(trimesh.exchange.obj.export_obj(mesh))

    @staticmethod
    def load(file_name: str) -> Tuple[np.ndarray, np.ndarray]:
        with open(file_name, "r") as f:
            mesh = trimesh.exchange.obj.load_obj(f)
        return np.float32(mesh["vertices"]), mesh["faces"]