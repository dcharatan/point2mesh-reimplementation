# Point2Mesh Reimplementation

## Project Setup

### Creating a Virtual Environment

To create a virtual environment, run `python3 -m venv venv`. Then, do `source venv/bin/activate` (or equivalent) and `pip3 install -r requirements.txt` to install the project's dependencies.

### Installing OpenEXR

#### Windows

Specifically there seems to be some sort of issues with some Windows machines installing OpenEXR (a critical component of tensorflow-graphics) through `pip3 install OpenEXR` or `pip3 install tensorflow-graphics`. Instead, the best option I found is downloading a precompiled wheel file from `https://www.lfd.uci.edu/~gohlke/pythonlibs/`, moving it to the workspace folder, then running `python -m pip install SomePackage-1.0-py2.py3-none-any.whl`. Then `pip3 install tensorflow-graphics` should work as you have already installed OpenEXR.

#### MacOS

Do the following:

```
brew install openexr
export CFLAGS="-I/Users/USERNAME/homebrew/include/OpenEXR -std=c++11"
export LDFLAGS="-L/Users/USERNAME/homebrew/lib"
pip3 install openexr
pip3 install tensorflow-graphics
```

### Running Scripts

The scripts in `script_modules` can be run as Python modules. To run `create_convex_hull.py`, run the command `python3 -m source.script_modules.create_convex_hull` from the project's root directory. Several VS Code run configurations are included in `.vscode/launch.json`.

### Running Unit Tests

To run the unit tests in VS Code, open the command menu (<kbd>⌘ Command</kbd> <kbd>⇧ Shift</kbd> <kbd>P</kbd> on MacOS) and run `Python: Discover Tests` and then `Python: Run All Tests`. I recommend discovering and running tests through Python Test Explorer for Visual Studio Code (`littlefoxteam.vscode-python-test-adapter`) if you add more tests, since it's hard to diagnose broken tests that aren't being discovered with the default test explorer.

## Acknowledgements

The files `elephant.pwn`, `hand.pwn` and `sphere.pwn` were taken from Alex Jacobson's [mesh reconstruction project](https://github.com/alecjacobson/geometry-processing-mesh-reconstruction).
