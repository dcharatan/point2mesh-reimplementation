import json

DEFAULTS = {
    # The point cloud that's fitted.
    "point_cloud": "data/point_clouds/giraffe.pwn",
    # The number of times remeshing/subdivision happens.
    "num_subdivisions": 6,
    # The number of iterations between each remeshing/subdivision.
    "num_iterations": 1000,
    # Each subdivision multiplies the number of faces by this.
    "subdivision_multiplier": 1.5,
    # The maximum number of faces that subdivision is allowed to yield.
    "max_num_faces": 10000,
    # The initial number of faces used for optimization.
    "initial_num_faces": 1000,
    # An optional initial mesh.
    "initial_mesh": "data/objs/giraffe_initmesh.obj",
    # The folder where the results are saved.
    "save_location": "results",
    # how often to run beamgap loss if -1 then no beam gap loss
    "beamgap_modulo": -1,
    # how often to save objs
    "obj_save_modulo": 5,
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