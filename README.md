# Point2Mesh Reimplementation

## Project Setup

### Creating a Virtual Environment

To create a virtual environment, run `python3 -m venv venv`. Then, do `source venv/bin/activate` (or equivalent) and `pip3 install -r requirements.txt` to install the project's dependencies.

### Running Scripts

The scripts in `script_modules` can be run as Python modules. To run `create_convex_hull.py`, run the command `python3 -m source.script_modules.create_convex_hull` from the project's root directory. Several VS Code run configurations are included in `.vscode/launch.json`.

### Running Unit Tests

To run the unit tests in VS Code, open the command menu (<kbd>⌘ Command</kbd> <kbd>⇧ Shift</kbd> <kbd>P</kbd> on MacOS) and run `Python: Discover Tests` and then `Python: Run All Tests`. I recommend discovering and running tests through Python Test Explorer for Visual Studio Code (`littlefoxteam.vscode-python-test-adapter`) if you add more tests, since it's hard to diagnose broken tests that aren't being discovered with the default test explorer.

## Acknowledgements

The files `elephant.pwn`, `hand.pwn` and `sphere.pwn` were taken from Alex Jacobson's [mesh reconstruction project](https://github.com/alecjacobson/geometry-processing-mesh-reconstruction).
