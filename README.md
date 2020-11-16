# Point2Mesh Reimplementation

## Project Setup

### Creating a Virtual Environment

To create a virtual environment, run `python3 -m venv venv`. Then, do `source venv/bin/activate` (or equivalent) and `pip3 install -r requirements.txt` to install the project's dependencies.

### Running Scripts

The scripts in `script_modules` can be run as Python modules. To run `create_convex_hull.py`, run the command `python3 -m source.script_modules.create_convex_hull` from the project's root directory. Several VS Code run configurations are included in `.vscode/launch.json`.

## Acknowledgements

The files `elephant.pwn`, `hand.pwn` and `sphere.pwn` were taken from Alex Jacobson's [mesh reconstruction project](https://github.com/alecjacobson/geometry-processing-mesh-reconstruction).
