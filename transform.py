import albumentations as A
import cv2
class Augumentation():
    def __init__(self):
        self.default = A.Compose(
            [
                # A.Resize(640,480),
                # A.RandomCrop(1920,1080),
                A.OneOf(
                    [A.CenterCrop(480, 640,p=0.8),
                    A.RandomCrop(480, 640,p=0.2)],p=1
                ),

                # A.CenterCrop(1080, 1920),
                A.Rotate(limit=180,p=0.9,border_mode=cv2.BORDER_CONSTANT),
                # A.HorizontalFlip(p=0.5),
                # A.VerticalFlip(p=0.3), 
                # A.ToGray(p=1), 
                A.Affine(shear=30,p=0.3),
                A.Blur(blur_limit=3,p=0.3),
                A.GaussNoise(p=0.3),
                A.ISONoise(p=0.3),

            ],
            bbox_params=A.BboxParams(format="yolo",min_visibility=0.7,label_fields = [])
        )
        # self.s_option = A.Compose(
        #     [
        #         # A.Resize(640,480),
        #         # A.RandomCrop(1920,1080),
        #         # A.OneOf(
        #         #     [A.CenterCrop(480, 640,p=0.8),
        #         #     A.RandomCrop(480, 640,p=0.2)],p=1
        #         # ),

        #         # A.CenterCrop(1080, 1920),
        #         A.Rotate(limit=180,p=0.9,border_mode=cv2.BORDER_CONSTANT),
        #         # A.HorizontalFlip(p=0.5),
        #         # A.VerticalFlip(p=0.3), 
        #         # A.ToGray(p=1), 
        #         A.Affine(shear=30,p=0.3),
        #         A.Blur(blur_limit=3,p=0.3),
        #         A.GaussNoise(p=0.3),
        #         A.ISONoise(p=0.3),

        #     ],
        #     bbox_params=A.BboxParams(format="yolo",min_visibility=0.7,label_fields = [])
        # )
        self.status = 0

    def transform(self,img,box):
        result = None
        if self.status == 0:
            result = self.default(image = img,bboxes=box)
        elif self.status ==1:
            img = cv2.resize(img, (640,480))
            result = self.default(image = img,bboxes=box)
        return result

    def changeStatus(self,i=1):
        self.status = i