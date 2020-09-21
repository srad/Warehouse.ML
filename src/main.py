import sys
from warehouse import tasks


def info():
    print("Use following parameters:")
    print("python main.py <task> <input-folder> <cam>\n")
    print("<task> = full        Run through the full pipeline")
    print("         boxes:      Generate contour boxes based on the annotation files")
    print("         index:      create a data.json file which converts file names to structured information about the files")
    print("         feature:    Extract feature from the images in the folder and write them to <output-dir>")
    print("         template:   Compute P_xy feature template and write")
    print("         estimate:   First parameter pxy numpy stored matrix, second the input feature image to compute the estimation")
    print("         plot:       Draws histograms of <template> computed data (P_xy)")
    exit(0)


if __name__ == "__main__":
    args = sys.argv[1:]
    argc = len(args)

    if args == 0:
        info()

    task = args[0]

    if task == "index" and argc == 2:
        in_dir = args[1]
        tasks.index(in_dir)
    elif task == "boxes" and argc >= 2:
        json_dir = args[1]
        cam = None
        if argc > 2:
            cam = args[2]
        tasks.box_threaded(json_dir, cam)
    elif task == "extract" and argc == 2:
        in_dir = args[1]
        tasks.extract_features(in_dir)
    elif task == "template" and argc == 2:
        in_dir = args[1]
        tasks.feature_template(in_dir)
    elif task == "full" and argc >= 2:
        in_dir = args[1]
        tasks.full(**dict({'in_dir': in_dir, 'cam': args[2] if argc == 3 else None}))
    elif task == "plot" and argc == 2:
        in_dir = args[1]
        tasks.load_plot(in_dir)
    else:
        info()
