import albumentations as A
from utils import  *
import matplotlib.pyplot as plt 
import argparse
from pathlib import Path
import glob
import os
import shutil
import tqdm
###
### this code woeks only open image with only one bbox
###
transform = A.Compose(
    [
        # A.Resize(640,480),
        # A.RandomCrop(1920,1080),
        A.OneOf(
            [A.CenterCrop(1080, 1920,p=0.5),
            A.RandomCrop(1080, 1920,p=0.5)],p=1
        ),
        # A.CenterCrop(1080, 1920),
        A.Rotate(limit=45,p=0.9,border_mode=cv2.BORDER_CONSTANT),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3), 
        # A.ToGray(p=1), 
        A.Blur(blur_limit=3,p=0.3),
        A.GaussNoise(p=0.3),
        A.ISONoise(p=0.3),

    ],
    bbox_params=A.BboxParams(format="yolo",min_visibility=0.7,label_fields = [])
)

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
        if name in bbox_source:
            #TODO this only process one ine
            with open(Path(opt.b)/name,"r") as f:
                out = f.read()[0:-1].split(" ")[1:]
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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        count = 0
        while count < opt.n:
            bbox_result = [list(map(float,bboxes[file]))]
            augumentation = transform(image = image,bboxes =bbox_result)
            # print(augumentation['bboxes'])
            if len(augumentation['bboxes'])==0:
                continue
            au_img = augumentation['image']
            box = augumentation['bboxes'][0]
            cv2.imwrite(opt.o+"/images/"+str(index)+".jpg",au_img)

            string_box = [str(i) for i in box]
            string_box = ["0"]+string_box

            with open(opt.o+"/labels/"+str(index)+".txt","w") as f:
                f.writelines(" ".join(string_box))

            count+=1
            index+=1

            # break




            

    #generat