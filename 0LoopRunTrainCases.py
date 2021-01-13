import os
from glob import glob
try:
    def run_case(file_dir, max_run=1, fileIdx=1):
        """
        # rotation_range = 45
        # width_shift_range = 0.3
        # height_shift_range = 0.3
        # zoom_range = 0.2
        # shear_range=0.35
        # horizontal_flip=True
        # brightness_range=[0.5, 1.3]

        :param file_dir:
        :param rotation_rangemax_run:
        :return:
        """

        # for root, dirs, files in os.walk(file_dir):
        #     print(files, "to be run")
        #     for i in files:
        all_files = glob(file_dir + '/*')
        print("all files:", all_files)
        for file in all_files:
            if file.endswith('Train.py'):
                for i in range(max_run):
                    print("run file: {}  {} ".format(file, fileIdx))
                    cmd = 'python ' + file + ' {}'.format(fileIdx)
                    print(cmd)
                    os.system(cmd)
                    fileIdx += 1
                    if fileIdx > max_run:
                        break
        print("total run {} training scripts!".format(fileIdx-1))
except Exception as e:
    print("Has some error", e)


# Initial deisign  no augmentaion
# case_list = ['C:\\MyProjects\\LungSegmentaiton\\EXP1']
# for laptop
# case_list = ['E:\\Projects\\LungSegmentaiton\\EXP4_preproInNpV2PipeV3']
case_list = ['E:\\Projects\\LungSegmentaiton\\EXP4_preproInNpV2PipeV3_Prostate']
for i, each_case in enumerate(case_list):
    print("i:", i)
    print(each_case)
    if i < 0:
        continue
    elif i ==0:
        run_case(each_case, fileIdx=1)
    else:
        run_case(each_case, fileIdx=1)
