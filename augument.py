import albumentations as A
from utils import  *
import matplotlib.pyplot as plt 
import argparse
from pathlib import Path
import glob
import os
import shutil
import tqdm
from transform import  Augumentation
###
### this code woeks only open image with multiple bbox
###

## TODO for now it is hardcode for the size
aug = Augumentation()
if __name__ == '__main__':
    
    parser  = argparse.ArgumentParser()
    parser.add_argument("-s",default=Path,required=True)
    parser.add_argument("-b",default=Path,required=True)
    parser.add_argument("-n",type=int,default=10)
    parser.add_argument("-o",default=Path,required=True)

    opt = parser.parse_args()
    source = os.listdir(opt.s)
    bbox_source = os.listdir(opt.b)
    bboxes = {}
    for file in source:
        name = str(file.split(".")[:-1][0])+".txt"
        # print(name)
        # print(bboxes)
        out = []
        if name in bbox_source:
            #TODO this only process one ine
            with open(Path(opt.b)/name,"r") as f:
                for line in f:
                    line_list = line.rstrip("\n").split(" ")[1:]
                    float_list = list(map(float,line_list))
                    out.append(float_list)
                bboxes[file] = out
    
    if  os.path.exists(opt.o):
        shutil.rmtree(opt.o)
    print(f"Creating dir {opt.o}")
    os.mkdir(opt.o)
    os.mkdir(opt.o+"/images")
    os.mkdir(opt.o+"/labels")
    # os.mkdir(opt.o+"/eval")
    # #select the image source and correspoinding bbox
    index = 0
    for file in tqdm.tqdm(source):
        file_pth = Path(opt.s)/file
        file_pth = file_pth.as_posix()
        image = cv2.imread(file_pth)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,(1920,1080))

        
        count = 0
        fail_count = 0
        while count < opt.n:
            bbox_result = bboxes[file]
            augumentation = aug.transform( image,bbox_result)
            # print(augumentation['bboxes'])
            if len(augumentation['bboxes'])==0:
                fail_count += 1
                #don't wast time on keep failing image
                if fail_count > 100:
                    print(f"{file} keep fail for 100 times skip this image,change mode")
                    #resize to solve issue
                    aug.changeStatus()
                continue
            au_img = augumentation['image']
            box = augumentation['bboxes']
            cv2.imwrite(opt.o+"/images/"+str(index)+".jpg",au_img)

            strings = []
            for bbox in box:

                string_box = [str(i) for i in bbox]
                string_box = ["0"]+string_box
                strings.append(string_box)

            with open(opt.o+"/labels/"+str(index)+".txt","w") as f:
                for s in strings:
                    f.writelines(" ".join(s)+"\n")
                    
        
            count+=1
            index+=1
        #change back to the original tranform
        aug.changeStatus(0)
            # break




            

    #generat