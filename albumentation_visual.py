import albumentations as A
from utils import  *
import matplotlib.pyplot as plt 
category_ids = [17, 18]
category_id_to_name = {17: 'cat', 18: 'dog'}
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
        A.GaussNoise(p=0.3)
    ],
    bbox_params=A.BboxParams(format="yolo",label_fields = [])
)

file = "/home/eddie/Downloads/yolo_img/images/0.jpg"
image = cv2.imread(file)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
bboxes = [[0.3716263171032291, 0.5 ,0.7432526342064582 ,1.0]]
# bboxes = [[5.66, 138.95, 147.09, 164.88]]
augumentation = transform(image=image,bboxes=bboxes)
print(augumentation)
au_img = augumentation['image']
# import pyplot as plt
plt.imshow(au_img)
plt.show()
# box = augumentation['bboxes'][0]