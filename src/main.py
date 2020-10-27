import sys
from warehouse import tasks
from warehouse.tex_sampler import mc_sample
from os import mkdir, path

def info():
    print("Use following parameters:")
    print("python main.py <task> <input-folder> <cam>\n")
    print("<task> = pipeline:   Run through the full pipeline")
    print("         boxes:      Generate contour boxes based on the annotation files")
    print("         index:      create a data.json file which converts file names to structured information about the files")
    print("         extract:    Extract feature from the images in the folder and write them to <output-dir>")
    print("         template:   Compute P_xy feature template and write")
    print("         estimate:   First parameter pxy numpy stored matrix, second the input feature image to compute the estimation")
    print("         plot:       Draws histograms of <template> computed data (P_xy)")
    print("         samplemc <input-image> <width> <height>: Generates a histogram and a sample texture via monte carlo sampling of the histogram")
    exit(0)


if __name__ == "__main__":
    if len(sys.argv[1:]) <= 1:
        info()
    
    args = sys.argv[1:]
    task = args[0]
    args = args[1:]
    argc = len(args)

    if task == "index" and argc == 2:
        in_dir = args[1]
        tasks.index(in_dir)
    elif task == "boxes" and argc >= 2:
        json_dir = args[1]
        cam = None
        if argc > 2:
            cam = args[2]
        tasks.box(json_dir, cam)
    elif task == "extract" and argc == 3:
        in_dir = args[1]
        template_path = args[2]
        tasks.extract_features(in_dir, template_path)
    elif task == "diff" and argc == 3:
        p = "output"
        if not path.exists(p):
            mkdir(p)
        tasks.diff(args[0], args[1], args[2])
    elif task == "template" and argc == 2:
        in_dir = args[1]
        tasks.feature_template(in_dir, "edge_")
        tasks.feature_template(in_dir, "corner_")
    elif task == "pipeline" and argc >= 3:
        in_dir = args[0]
        template_path = args[1]
        cam = args[2] if argc == 3 else None
        tasks.pipeline(in_dir, template_path, cam)
    elif task == "plot" and argc == 2:
        in_dir = args[0]
        name = args[1]
        tasks.load_plot(in_dir, name)
    elif task == "diffnoise":
        p = "output"
        if not path.exists(p):
            mkdir(p)
        tasks.diff_noise(args[0], args[1], p)
    elif task == "addnoise":
        tasks.add_noise(args[0], float(args[1]))
    elif task == "match" and argc == 3:
        tasks.match_templates_load(args[1], 0.9, args[2])
    elif task == "shadow" and argc == 2:
        tasks.find_shadows_load(args[1])
    elif task == "testextract" and argc == 3:
        tasks.extract_feature_load(args[1], args[2])
    elif task == "samplemc" and argc == 3:
        file = args[0]
        w = args[1]
        h = args[2]
        p = "output"
        if not path.exists(p):
            mkdir(p)
        mc_sample(file, p, int(w), int(h))
    else:
        info()
