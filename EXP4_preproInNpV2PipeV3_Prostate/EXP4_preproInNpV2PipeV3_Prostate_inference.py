import cv2
import numpy as np
import os
import time
# need to change
from glob import glob
# from keras.preprocessing import image as krs_images
from EXP4_preproInNpV2PipeV3_Prostate.EXP4_preproInNpV2PipeV3_Prostate_Train import YOLO, \
    get_anchors, my_get_random_data2, NUM_ANGLES, max_boxes, krs_image, warm_up_funcV2 #or "import poly_yolo_lite as yolo" for the lite version  ### need to change for different model design


import sys
import nibabel as nib

saved_model_name =  sys.argv[1]
best_h5_path =  sys.argv[2]
output_folder =  sys.argv[3]
FPS_txt = sys.argv[4]


def get_classes(classes_path):
    """loads the classes"""
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

print("cwd:", os.getcwd())
current_file_dir_path = os.path.dirname(os.path.realpath(__file__))
print("current file dir:", current_file_dir_path)
classes_path = current_file_dir_path+'/yolo_classesTongue.txt'
class_names = get_classes(classes_path)
print("class names:", class_names)

# # EXP BASE_1
# modelname =  "Exp_base1"
# h5_name = "ep087-loss21.695-val_loss22.960.h5"
# # EXP BASE_2
# modelname =  "Exp_base2"
# h5_name = "ep061-loss19.139-val_loss20.301.h5"
# # EXP BASE_2
# modelname =  "Exp_base3"
# h5_name = "ep077-loss22.085-val_loss23.323.h5"

# saved_model_folder = "F:\\TonguePolyYOLOLOGS\\MYAugGenerator/" + modelname

output_root = output_folder + "/" + saved_model_name

if not os.path.exists(output_root):
    os.makedirs(output_root)
# inference txt name
inferTXTName = output_root+ '/predictResult_{}.txt'.format(saved_model_name)
LabelTXTName = output_root+ '/labelResult_{}.txt'.format(saved_model_name)

file = open(inferTXTName, "w")
label_out = open(LabelTXTName, 'w')

#if you want to detect more objects, lower the score and vice versa
trained_model = YOLO(model_path=best_h5_path,  ## need to change
                          classes_path=current_file_dir_path+'/yolo_classesTongue.txt', # this need to specified for your model used classes
                          anchors_path = current_file_dir_path+'/yolo_anchorsTongue.txt',
                          iou=0.5, score=0.5)

#helper function
def translate_color(cls):
    if cls == 0: return (230, 25, 75)
    if cls == 1: return (60, 180, 75)
    if cls == 2: return (255, 225, 25)
    if cls == 3: return (0, 130, 200)
    if cls == 4: return (245, 130, 48)
    if cls == 5: return (145, 30, 180)
    if cls == 7: return (70, 240, 240)
    if cls == 8: return (240, 50, 230)
    if cls == 9: return (210, 245, 60)
    if cls == 10: return (250, 190, 190)
    if cls == 11: return (0, 128, 128)
    if cls == 12: return (230, 190, 255)
    if cls == 13: return (170, 110, 40)
    if cls == 14: return (255, 250, 200)
    if cls == 15: return (128, 0, 128)
    if cls == 16: return (170, 255, 195)
    if cls == 17: return (128, 128, 0)
    if cls == 18: return (255, 215, 180)
    if cls == 19: return (80, 80, 128)


def predict_for_list(test_list, image_count, box_count, fps_list, FP_name, FP=0,flag ="Nonzero"):
    for i in range(len(test_list)):
            temp_dict_data = test_list[i]
            # print("temp_dict_data:", temp_dict_data)
            # temp_mask_path = data_list[count]
            # each_input_series + " " + each_mask_series + " " + str(each_data_mean) + " " + str(each_data_std)
            # key, value =  list(temp_dict_data)[0].split(" ")
            splitted_data, nonzero_index = temp_dict_data
            splitted_data = splitted_data.split(" ")
            full_name = os.path.basename(splitted_data[0]) + str(nonzero_index)
            # splitted_data, nonzero_index =  list(temp_dict_data.values())[0]
            print("splitted_data:", splitted_data)
            print("nonzero index:", nonzero_index)
            # for each_slice_index in test_nonzeros_indexs_series:
            #     # 2. read the data and random select one index from nonzero index series
            #     print("For test:")
            #     # selected_slice_index = test_nonzeros_indexs_series[0]
            #     print("selected nonzero slice index:", each_slice_index)
            #
            #
            #     selected_slice_input = input_data[:, :, each_slice_index]
            #     selected_slice_mask = mask_data[:, :, each_slice_index]

            # start to encode and decode the polygons

            img, _, myPolygon, aug_mask, annotation_line = my_get_random_data2(splitted_data[0], splitted_data[1],
                                                                                   nonzero_index,
                                                                                   float(splitted_data[2]),
                                                                                   float(splitted_data[3]),
                                                                                   None, None,
                                                                                   train_or_test="Test")

            print("flag:",flag)
            # print("myPolygon", myPolygon)
            # print("annotation_line:",annotation_line)
            # count image +1 for nonzeros
            image_count += 1
            boxes = []
            scores = []
            classes = []

            # realize prediciction using poly-yolo
            # decode from polar to xy
            polygon_xy = np.zeros([max_boxes, 2 * NUM_ANGLES])
            startx = time.perf_counter()
            box, score, classs, polygons = trained_model.detect_image(img, input_shape, polygon_xy)
            print("len(box):", len(box))

            # out_boxes, out_scores, out_classes, polygons  = trained_model.detect_image(input_img,input_shape, polygon_xy)
            # # get
            # startx = time.perf_counter()
            # box, score, classs, polygons = trained_model.detect_image(input_img,input_shape)
            endtx = time.perf_counter()
            print("startx:", startx)
            print("endtx:", endtx)

            tmp_fps = 1.0 / (endtx - startx)
            print('Prediction speed: ', tmp_fps, 'fps')
            fps_list.append(tmp_fps)
            # example, hw to reshape reshape y1,x1,y2,x2 into x1,y1,x2,y2

            if flag == "Nonzero":
                print("writing annotations for nonzeros")
                print("annoataion_line", annotation_line)
                label_out.write(annotation_line)

                label_out.write("\n")
                if len(box) > 0:
                    print("there is a box prediction for nonzero slice")
            else:
                if len(box) > 0:
                    print("there is a false box prediction for zero slice")
                    FP += 1
                    FP_name.append(full_name)



            # draw:
            background = img.copy()*255
            background= cv2.cvtColor(background, cv2.COLOR_BGR2RGB)



            overlay = img.copy()

            for c in myPolygon:
                # compute the center of the contour
                # M = cv2.moments(c)
                # cX = int(M["m10"] / M["m00"])
                # cY = int(M["m01"] / M["m00"])
                # draw the contour and center of the shape on the image
                cv2.drawContours(background, [c], -1, (0, 255, 0), 2)
            for k in range(0, len(box)):
                boxes.append((box[k][1], box[k][0], box[k][3], box[k][2]))
                scores.append(score[k])
                classes.append(classs[k])

                cv2.rectangle(background, (box[k][1], box[k][0]), (box[k][3], box[k][2]), translate_color(classes[k]), 3, 1)
                cv2.putText(background, "{}:{:.2f}".format(class_names[classs[k]], score[k]), (int(box[k][1]), int(box[k][0])-3 ),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            translate_color(classes[k]), 1)
            box_count += len(boxes)

            if len(boxes) == 0:
                continue

            if flag == "Nonzero":
                file.write(full_name + " ")
            print("write image path")
            # browse all boxes
            for b in range(0, len(boxes)):

                # draw box and masks on the raw images:-------->
                f = translate_color(classes[b])
                points_to_draw = []
                offset = len(polygons[b]) // 2  # this = NUM_ANGLES
                # offset = NUM_ANGLES
                # filter bounding polygon vertices
                print("polygons len:", len(polygons[b]))
                print("offset")
                for dst in range(0, offset):   # this = NUM_ANGLES LOOP TO GET (X,Y) pairs
                    # if polygons[b][dst + offset] > 0.3:
                    points_to_draw.append([int(polygons[b][dst]), int(polygons[b][dst + offset])])

                points_to_draw = np.asarray(points_to_draw)
                points_to_draw = points_to_draw.astype(np.int32)
                if points_to_draw.shape[0] > 0:
                    cv2.polylines(background, [points_to_draw], True, f, thickness=2)
                    cv2.fillPoly(overlay, [points_to_draw], f)

                # cv2.polylines(img, [myPolygon], True, f, thickness=2)
                # cv2.fillPoly(overlay, [myPolygon], f)

                # write into txt:-------->
                if flag == "Nonzero":
                    str_to_write = ''

                    str_to_write += str(float(boxes[b][0])) + "," + str(float(boxes[b][1])) + "," + str(
                        float(boxes[b][2])) + "," + str(float(boxes[b][3])) + ","
                    str_to_write += str(scores[b]) + ","
                    str_to_write += str(int(classes[b]))

                    offset = len(polygons[b]) // 2  # 72 for 24 vertexes. offset = 24
                    vertices = 0
                    for dst in range(0, len(polygons[b]) // 2):  # 下取整
                        # if polygons[b][dst + offset] > 0.2:

                        str_to_write += "," + str(float(polygons[b][dst])) + "," + str(float(polygons[b][dst + offset]))
                        vertices += 1
                    str_to_write += " "
                    if vertices < 3:
                        print("No mask found")
                        print('found not correct polygon with ', vertices, ' vertices')
                        continue
                    # print(str_to_write)
                    file.write(str_to_write)
            file.write("\n")

            img = cv2.addWeighted(overlay, 0.4, background, 1 - 0.4, 0)
            cv2.imwrite(out_path + full_name + '.jpg', img)

    return image_count, box_count, fps_list, FP, FP_name

# def extract_series(series_path):
#     img_obj = nib.load(series_path)
#     # print("series:", img_obj.shape)
#     # print("img.get_data_dtype() == np.dtype(np.int16):", img_obj.get_data_dtype() == np.dtype(np.int16))
#     # print("img.affine.shape:", img_obj.affine.shape)
#     data = img_obj.get_fdata()
#     # print("data in numpy:", data.shape)
#     # print("data in numpy range: [{}, {}]".format(data.min(), data.max()))
#     return data


# def warm_up_func(train_input_series, train_mask_series):
#     train_mean_list_series = []
#     train_std_list_series = []
#     train_nonzeros_indexs_series = []
#     train_nonzeros_slices_num = 0
#     for each_input_series, each_mask_series in zip(train_input_series, train_mask_series):
#
#         non_zeros_indexes = []
#
#         data_input = extract_series(each_input_series)
#         data_mask = extract_series(each_mask_series)
#         each_v = data_input.reshape([-1])
#         print("each_v shape:", each_v.shape)
#         each_data_mean = np.mean(each_v, axis=0)
#         each_data_std = np.std(each_v, axis=0)
#         train_mean_list_series.append(each_data_mean)
#         train_std_list_series.append(each_data_std)
#         print("each train series data shape:", data_input.shape)
#
#         assert data_input.shape[-1] == data_mask.shape[-1]
#         for i in range(data_mask.shape[-1]):
#             temp_mask = data_mask[:, :, i]
#             # print("each slice mask shape:", temp_mask.shape)
#             nonzeros_in_mask = np.count_nonzero(temp_mask)
#             if nonzeros_in_mask > 5:
#                 # print("find non zeros in the mask at {}: {}".format(i, nonzeros_in_mask))
#                 non_zeros_indexes.append(i)
#                 train_nonzeros_slices_num += 1
#             else:
#                 pass
#         train_nonzeros_indexs_series.append(non_zeros_indexes)
#
#     return train_mean_list_series, train_std_list_series, train_nonzeros_indexs_series, train_nonzeros_slices_num



# dir_imgs_name = 'E:\\dataset\\Tongue\\mytonguePolyYolo\\test\\test_inputs' #path_where_are_images_to_clasification
# dir_imgs_name = 'E:\\dataset\\Tongue\\mytonguePolyYolo\\test\\test_inputs' #path_where_are_images_to_clasification
# test_txt_path = current_file_dir_path+'/myTongueTestLab.txt'
# FOR THE LAB
# test_txt_path = current_file_dir_path+'/myTongueTestLab.txt'   # need to change
out_path       = output_root+'/PredOut/' #path, where the images will be saved. The path must exist
if not os.path.exists(out_path):
    os.makedirs(out_path)
MAX_VERTICES = 1000 #that allows the labels to have 1000 vertices per polygon at max. They are reduced for training

# # read test lines from txt file
# with open(test_txt_path) as f:
#     text_lines = f.readlines()
#     print("total {} test samples read".format(len(text_lines)))
#
# # print(text_)
# for i in range(0, len(text_lines)):
#
#     text_lines[i] = text_lines[i].split()
#     #     print(text_lines[i])
#     for element in range(1, len(text_lines[i])):
#         for symbol in range(text_lines[i][element].count(',') - 4, MAX_VERTICES * 2, 2):
#             text_lines[i][element] = text_lines[i][element] + ',0,0'
#         # print(text_lines)
#
# # %%
#
# # browse all images
# print("cwd:", os.getcwd())
# cwd = os.getcwd()
#
# # os.chdir("E:\\Projects\\poly-yolo\\simulator_dataset\\imgs")
classes_path = current_file_dir_path + '/yolo_classesTongue.txt'
anchors_path = current_file_dir_path + '/yolo_anchorsTongue.txt'
class_names = get_classes(classes_path)
num_classes = len(class_names)
anchors = get_anchors(anchors_path)

# raw_input_shape = (416,832) # multiple of 32, hw
input_shape = (256, 256)  # multiple of 32, hw

# # for validation dataset  # we need or label and masks are the same shape
# test_input_paths = glob('E:\\dataset\\Tongue\\mytonguePolyYolo\\test\\test_inputs/*')
# test_mask_paths = glob('E:\\dataset\\Tongue\\mytonguePolyYolo\\test\\testLabel\\label512640/*.jpg')

# test_input_paths = glob('C:\\MyProjects\\data\\tonguePoly\\test\\input/*')
# test_mask_paths = glob('C:\\MyProjects\\data\\tonguePoly\\test\\label/*.jpg')

# for laptop
train_total_series = glob('E:\\dataset\\Prostate\\myDownload\\trainData/*.mhd')
# val_total_series = glob('E:\\dataset\\Prostate\\myDownload\\trainData/*')

total_input_series_paths = []
total_labels_series_paths = []
for each_train_path in train_total_series:  # find input and label
    # print(each_train_path)
    if "seg" in each_train_path:
        total_labels_series_paths.append(each_train_path)
    else:
        total_input_series_paths.append(each_train_path)

print("total len of total input series:", len(total_input_series_paths))
print("total len of total mask series:", len(total_labels_series_paths))

# warm up ~~~---------------->
# split train dataset:
print("splitting the training dataset into train and val")
train_rate = 0.8

val_rate = 0.5  # half of the remainning
# print("raw total series paths:", train_input_series_paths)
num_train_cases = int(train_rate * len(total_input_series_paths))
print("splitted num_train_cases:", num_train_cases)

# num_val_cases = int(val_rate * len(train_input_series_paths))
# print("splitted num_train_cases:", num_train_cases)

# for train ->
train_input_series = total_input_series_paths[:num_train_cases]
print("splitted train_input_series # :", len(train_input_series))
train_mask_series = total_labels_series_paths[:num_train_cases]

# train_input_series = total_input_series_paths[0:1]
# print("splitted train_input_series # :", len(train_input_series))
# train_mask_series = total_labels_series_paths[0:1]
# for val ->
# val_input_series = total_input_series_paths[
#                    num_train_cases:num_train_cases+1]   # for check purpose only
# val_mask_series = total_labels_series_paths[num_train_cases:num_train_cases+1] # for check purpose only
remaining_input_series = total_input_series_paths[
                         num_train_cases:]  ## split training data into real train and validation
remaining_mask_series = total_labels_series_paths[num_train_cases:]

num_val_cases = int(val_rate * len(remaining_input_series))
val_input_series = remaining_input_series[:num_val_cases]
print("splitted val_input_series # :", len(val_input_series))
val_mask_series = remaining_mask_series[:num_val_cases]
# # for test ->
# test_input_series = remaining_input_series[num_val_cases:]
# print("splitted test_input_series # :", len(test_input_series))
# test_mask_series = remaining_mask_series[num_val_cases:]

# # for the laptop:
# val_input_series = remaining_input_series[0:1]
# print("splitted val_input_series # :", len(train_input_series))
# val_mask_series = remaining_mask_series[0:1]

# # for test ->
test_input_series = remaining_input_series[num_val_cases:]
# print("splitted test_input_series # :", len(test_input_series))
test_mask_series = remaining_mask_series[num_val_cases:]
# #
# print("splitted val_input_series # :", len((val_input_series)))
# anaylize the training CTs
# print("train series len:", len(train_input_series))
print("test series len:", len(test_input_series))

# warm up for training needed

print("train warm up ---------------------------->")
# for each_input_series, each_mask_series in zip(train_input_series, train_mask_series):
# train_mean_list_series, train_std_list_series, train_nonzeros_indexs_series, train_nonzeros_slices_num = \
#     warm_up_func(train_input_series, train_mask_series)

nonzero_test_data_list, zero_test_data_list = \
    warm_up_funcV2(test_input_series, test_mask_series)

print("nonzeros test_data_list:", nonzero_test_data_list)
print("zeros test_data_list:", zero_test_data_list)

assert len(test_input_series) == len(test_mask_series), "test imgs and mask are not the same"
print("total {} testsamples read".format(len(test_input_series)))

test_data_list =  zero_test_data_list + nonzero_test_data_list
# # package the path and warm up info.
# ziped_series_mask_list = list(zip(val_input_series, val_mask_series,
#                                       test_mean_list_series, test_std_list_series,
#                                       test_nonzeros_indexs_series))

# create data_generator
#
# test_Gen = my_Gnearator(test_input_paths, test_mask_paths, batch_size=4, input_shape=[256, 256],
#                        anchors=anchors, num_classes=num_classes,
#                        train_flag="test")
nonzero_total_boxes = 0
nonzero_imgs = 0
zero_total_boxes = 0
zero_imgs = 0
fps_list=[]
input_shape=[256, 256]
FP= 0
FP_name = []



# for nonzero list test
nonzero_image_count, nonzero_box_count, fps_list, FP, FP_name = predict_for_list(nonzero_test_data_list, nonzero_imgs,
                                                                 nonzero_total_boxes,fps_list, FP_name,FP, flag ="Nonzero")

# for zero list test
zero_image_count, zero_box_count, fps_list, FP, FP_name = predict_for_list(zero_test_data_list, zero_imgs,
                                                                 zero_total_boxes,fps_list, FP_name,FP, flag ="zero")

total_detected_box =  nonzero_box_count+ zero_box_count
imgs = nonzero_image_count +  zero_image_count
file.close()
label_out.close()
print('total detected boxes: ', total_detected_box)
print('imgs: ', imgs)
print("avg fps:", sum(fps_list)/len(fps_list))


with open(FPS_txt, 'a') as f:
    f.write("saved_model_name {}, total_num_imgs {}, num_nonzeros_detected {}/{}, zeros_dtected {}/{}, avg_fps {}, std_fps {}\n".
            format (saved_model_name,imgs, nonzero_box_count,nonzero_image_count, zero_box_count, zero_image_count,np.array(fps_list).mean(), np.array(fps_list).std()))
    f.write("# of FP:{}\n".format(FP))
    if len(FP_name)>0:
        for i in range(len(FP_name)):
            f.write("FP names:{}\n".format(FP_name[i]))
    else:
        f.write("NoFP\n")
