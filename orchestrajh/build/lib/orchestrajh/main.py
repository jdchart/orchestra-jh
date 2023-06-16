from .preprocessing import *
from .staff_removal import *
from .helper_methods import *
from .utils import *

import os
import datetime

params = {
    "inputfolder" : "",
    "outputfolder" : "",
    "threshold" : 0.8,
    "filename" : os.path.join(os.getcwd(), "model.sav"),
    "accidentals" : ['x', 'hash', 'b', 'symbol_bb', 'd']
}

def preprocessing(inputfolder, fn, f):
      # Get image and its dimensions #
    height, width, in_img = preprocess_img('{}/{}'.format(inputfolder, fn))
    
    # Get line thinkness and list of staff lines #
    staff_lines_thicknesses, staff_lines = get_staff_lines(width, height, in_img, params["threshold"])

    # Remove staff lines from original image #
    cleaned = remove_staff_lines(in_img, width, staff_lines, staff_lines_thicknesses)
    
    # Get list of cutted buckets and cutting positions #
    cut_positions, cutted = cut_image_into_buckets(cleaned, staff_lines)
    
    # Get reference line for each bucket #
    ref_lines, lines_spacing = get_ref_lines(cut_positions, staff_lines)

    return cutted, ref_lines, lines_spacing, cut_positions

def get_target_boundaries(label, cur_symbol, y2):
    if label == 'b_8':
        cutted_boundaries = cut_boundaries(cur_symbol, 2, y2)
        label = 'a_8'
    elif label == 'b_8_flipped':
        cutted_boundaries = cut_boundaries(cur_symbol, 2, y2)
        label = 'a_8_flipped'
    elif label == 'b_16':
        cutted_boundaries = cut_boundaries(cur_symbol, 4, y2)
        label = 'a_16'
    elif label == 'b_16_flipped':
        cutted_boundaries = cut_boundaries(cur_symbol, 4, y2)
        label = 'a_16_flipped'
    else: 
        cutted_boundaries = cut_boundaries(cur_symbol, 1, y2)

    return label, cutted_boundaries

def get_label_cutted_boundaries(boundary, height_before, cutted, model):
    # Get the current symbol #
    x1, y1, x2, y2 = boundary
    cur_symbol = cutted[y1-height_before:y2+1-height_before, x1:x2+1]

    # Clean and cut #
    cur_symbol = clean_and_cut(cur_symbol)
    cur_symbol = 255 - cur_symbol

    # Start prediction of the current symbol #
    feature = extract_hog_features(cur_symbol)
    label = str(model.predict([feature])[0])

    return get_target_boundaries(label, cur_symbol, y2)

def process_image(inputfolder, fn, f, model, root, exportConfig):
    cutted, ref_lines, lines_spacing, cut_positions = preprocessing(inputfolder, fn, f)

    if exportConfig["lineCutPositions"]:
        export_data_to_file(cut_positions, os.path.join(root, "data/cut-positions.txt"))
    main_matrix = {"lines" : []}
    
    last_acc = ''
    last_num = ''
    height_before = 0

    if len(cutted) > 1:
        f.write('{\n')

    for it in range(len(cutted)):
        f.write('[')
        is_started = False
        
        symbols_boundaries = segmentation(height_before, cutted[it])
        symbols_boundaries.sort(key = lambda x: (x[0], x[1]))

        current_regions = []
        matrix = []

        for i in range(len(symbols_boundaries)):
            boundary = symbols_boundaries[i]


            label, cutted_boundaries = get_label_cutted_boundaries(boundary, height_before, cutted[it], model)

            if exportConfig["singleBoundariesData"]:
                export_data_to_file(boundary, os.path.join(root, "data/boundary-" + str(it) + "-" + str(i) + "-" + str(label) + ".txt"))
            reg = [boundary[0], boundary[1], boundary[2], boundary[3]]
            if exportConfig["singleBoundariesImage"]:
                export_image_region(os.path.join(inputfolder, fn), os.path.join(root, "data/boundary-" + str(it) + "-" + str(i) + "-" + str(label) + ".png"), reg)

            current_regions.append(reg)

            if label == 'clef':
                is_started = True
            
            for cutted_boundary in cutted_boundaries:
                _, y1, _, y2 = cutted_boundary
                if is_started == True and label != 'barline' and label != 'clef':
                    text = text_operation(label, ref_lines[it], lines_spacing[it], y1, y2)
                    
                    if (label == 't_2' or label == 't_4') and last_num == '':
                        last_num = text
                    elif label in params["accidentals"]:
                        last_acc = text
                    else:
                        if last_acc != '':
                            text = text[0] + last_acc + text[1:]
                            last_acc=  ''
                            
                        if last_num != '':
                            text = f'\meter<"{text}/{last_num}">'
                            last_num =  ''
                        
                        not_dot = label != 'dot'
                        f.write(not_dot * ' ' + text)

                        if not_dot:
                            matrix.append({
                                "item" : text,
                                "regions" : current_regions
                            })
                            current_regions = []
            
        height_before += cutted[it].shape[0]
        f.write(' ]\n')

        main_matrix["lines"].append(matrix)
        
    if len(cutted) > 1:
        f.write('}')

    writeJson(main_matrix, os.path.join(root, "data/matrix.json"))

def main_process(model, exportConfig):
    try: 
        os.mkdir(params["outputfolder"]) 
    except OSError as error:
        pass

    list_of_images = os.listdir(params["inputfolder"])
    for _, fn in enumerate(list_of_images):
        # Open the output text file #
        file_prefix = fn.split('.')[0]
        root = os.path.join(params["outputfolder"] + "/" + file_prefix)
        makeDirsRecustive([os.path.join(root, "data")])
        f = open(os.path.join(root, file_prefix + ".txt"), "w")

        # Process each image separately #
        try:
            print("\nProcessing " + file_prefix)
            process_image(params["inputfolder"], fn, f, model, root, exportConfig)
        except Exception as e:
            print(e)
            print(f'{params["inputfolder"]}-{fn} has been failed !!')
            pass
        
        f.close()  
    print('Finished !!') 

def process(inputDir: str, outputDir: str, **kwargs):

    params["inputfolder"] = inputDir
    params["outputfolder"] = outputDir
    params["threshold"] = kwargs.get("threshold", 0.8)
    params["filename"] = kwargs.get("filename", os.path.join(os.getcwd(), "model.sav"))
    params["accidentals"] = kwargs.get("accidentals", ['x', 'hash', 'b', 'symbol_bb', 'd'])

    exportConfig = {
        "lineCutPositions" : kwargs.get("export_line_cut_positions", False),
        "singleBoundariesData" : kwargs.get("export_single_boundaries_data", False),
        "singleBoundariesImage" : kwargs.get("export_single_boundaries_image", True) 
    }

    print(params)

    model = pickle.load(open(params["filename"], 'rb'))

    with open(params["outputfolder"] + "/Output.txt", "w") as text_file:
        text_file.write("Input Folder: %s" % params["inputfolder"])
        text_file.write("Output Folder: %s" % params["outputfolder"])
        text_file.write("Date: %s" % datetime.datetime.now())
    
    main_process(model, exportConfig)