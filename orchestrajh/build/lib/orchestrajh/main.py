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

            if len(cutted_boundaries) > 1:
                print("\nCutted boundaries was len: " + str(len(cutted_boundaries)))
                print("Boundary was: " + str(reg))
                for cutted_boundary in cutted_boundaries:
                    print("\t" + str(cutted_boundary))
                
            
            for count in range(len(cutted_boundaries)):
                cutted_boundary = cutted_boundaries[count]
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

                            if len(cutted_boundaries) == 1:
                                matrix.append({
                                    "item" : text,
                                    "regions" : current_regions
                                })
                                current_regions = []
                            else:
                                print("\tText: " + str(text))
                                print("\tCurrent regions: " + str(current_regions))
                                current_regions = []
                                current_regions.append([
                                    reg[0] + int(cutted_boundary[0]) + (int(cutted_boundary[2]) * count),
                                    int(cutted_boundary[1]),
                                    reg[0] + int(cutted_boundary[0]) + (int(cutted_boundary[2]) * count) + int(cutted_boundary[2]),
                                    int(cutted_boundary[3]),
                                ])
                                print("\tCurrent regions converted: " + str(current_regions))
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

    if exportConfig["exportCompositeImages"]:
        makeDirsRecustive([os.path.join(root, "data/composite-images")])
        export_composite_images(main_matrix, os.path.join(inputfolder, fn), os.path.join(root, "data/composite-images"))

    writeJson(main_matrix, os.path.join(root, "data/matrix.json"))

def export_composite_images(data, img, root_path):
    for i in range(len(data["lines"])):
        currentArray = data["lines"][i]
        for j in range(len(currentArray)):
            thisItem = currentArray[j]

            imageFileName = "composite-" + str(i) + "-" + str(j) + ".png"
            data["lines"][i][j]["composite_image_filename"] = imageFileName

            compositeRegion = get_composite_region(thisItem["regions"])
            data["lines"][i][j]["composite_image_region"] = compositeRegion
            
            export_image_region(img, os.path.join(root_path, imageFileName), compositeRegion)

def get_composite_region(regionList):
    if len(regionList) > 0:
        currentX1 = regionList[0][0]
        currentY1 = regionList[0][1]
        currentX2 = regionList[0][2]
        currentY2 = regionList[0][3]

        for region in regionList:
            if region[0] < currentX1:
                currentX1 = region[0]
            if region[1] < currentY1:
                currentY1 = region[1]
            if region[2] > currentX2:
                currentX2 = region[2]
            if region[3] > currentY2:
                currentY2 = region[3]
        
        return [currentX1, currentY1, currentX2, currentY2]
    else:
        return [0, 0, 1, 1]

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
        "lineCutPositions" : kwargs.get("export_line_cut_positions", True),
        "singleBoundariesData" : kwargs.get("export_single_boundaries_data", True),
        "singleBoundariesImage" : kwargs.get("export_single_boundaries_image", True),
        "exportCompositeImages" : kwargs.get("export_composite_images", True)
    }

    print(params)

    model = pickle.load(open(params["filename"], 'rb'))

    with open(params["outputfolder"] + "/Output.txt", "w") as text_file:
        text_file.write("Input Folder: %s" % params["inputfolder"])
        text_file.write("Output Folder: %s" % params["outputfolder"])
        text_file.write("Date: %s" % datetime.datetime.now())
    
    main_process(model, exportConfig)