import json

DEFAULTS = {
    # The point cloud that's fitted.
    "point_cloud": "data/point_clouds/elephant.pwn",
    # The number of times remeshing/subdivision happens.
    "num_subdivisions": 3,
    # The number of iterations between each remeshing/subdivision.
    "num_iterations": 1000,
}


def load_options(argv: list):
    try:
        with open(argv[1], "r") as f:
            options = json.load(f)
    except:
        return DEFAULTS

    for key in options:
        if key not in DEFAULTS:
            raise Exception(f'Unknown setting "{key}"')
    return {**DEFAULTS, **options}