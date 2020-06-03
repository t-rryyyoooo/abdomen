import numpy as np
import SimpleITK as sitk
import os
import sys
import argparse
from functions import DICE
from pathlib import Path
from tqdm import tqdm

args = None

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('true_directory', help = '~/Desktop/data/KIDNEY')
    parser.add_argument('predict_directory',help = '~/Desktop/data/hist/segmentation')
    parser.add_argument('patientID_list',help = '000 001 002', nargs="*")
    parser.add_argument("--classes", help="3", default=3, type=int) 
    parser.add_argument("--class_label", help="bg kidney cancer", nargs="*") 
    parser.add_argument("--true_name", help="segmentation.nii.gz", default="segmentation.nii.gz") 
    parser.add_argument("--predict_name", help="label.mha", default="label.mha") 
    args = parser.parse_args()

    return args


def main(args):
    if args.class_label is not None:
        if len(args.class_label) != args.classes:
            print("[ERROR] You have to equate the length of class_label with the classes")
            sys.exit()
    else:
        args.class_label = ["Class {}".format(c) for c in range(args.classes)]

    whole_DICE = []
    DICE_per_class = [[] for _ in range(args.classes)]
    total = len(args.patientID_list) * args.classes
    with tqdm(desc="Caluculating DICE...", ncols=60, total=total) as pbar:
        for x in args.patientID_list:
            true_directory = Path(args.true_directory) / ("case_00" + x) / args.true_name
            predict_directory = Path(args.predict_directory) / ("case_00" + x) / args.predict_name

            true = sitk.ReadImage(str(true_directory))
            predict = sitk.ReadImage(str(predict_directory))

            true_array = sitk.GetArrayFromImage(true)
            predict_array = sitk.GetArrayFromImage(predict)
         
            dice = DICE(true_array, predict_array)
            whole_DICE.append(dice)

            print('case_00' + x)
            print("Whole DICE : {}".format(dice))
            for c in range(args.classes):
                true_c_array = (true_array == c).astype(int)
                predict_c_array  = (predict_array == c).astype(int)
                
                dice = DICE(true_c_array, predict_c_array)
                DICE_per_class[c].append(dice)
                print("{} DICE : {}".format(args.class_label[c], dice))
                pbar.update(1)
            print()


    avg_dice = np.mean(whole_DICE)
    print("Average whole DICE : {}".format(avg_dice))
    for c, dice_c in zip(range(args.classes), DICE_per_class):
        avg_dice = np.mean(dice_c)
        print("Average {} DICE : {}".format(args.class_label[c], avg_dice))
    print()

if __name__ == '__main__':
    args = parseArgs()
    main(args)
