import matplotlib.pyplot as plt
import numpy as np
import cv2



# def plot_image(image, factor=1.0, clip_range=None, **kwargs):
#     """
#     Utility function for plotting RGB images.
#     """
#     fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
#     if clip_range is not None:
#         ax.imshow(np.clip(image * factor, *clip_range), **kwargs)
#     else:
#         ax.imshow(image * factor, **kwargs)
#     ax.set_xticks([])
#     ax.set_yticks([])

BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White


def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    img_h,img_w,_ = img.shape
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
   
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35, 
        color=TEXT_COLOR, 
        lineType=cv2.LINE_AA,
    )
    return img

def visualize_bbox_yolo_single(img, bbox, class_name, color=BOX_COLOR, thickness=5):
    """Visualizes a single bounding box on the image"""
    print(img.shape)
    img_h,img_w,_ = img.shape
    x_ratio, y_ratio, w_ratio, h_ratio = bbox
    x = img_w*x_ratio
    y = img_h*y_ratio

    w = img_w*w_ratio
    h = img_h*h_ratio



    x_min,  y_min, x_max,y_max = int(x-w/2), int(y-h/2), int(x+w/2), int(y+h/2)
   
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35, 
        color=TEXT_COLOR, 
        lineType=cv2.LINE_AA,
    )
    return img


def visualize(image, bboxes, category_ids, category_id_to_name):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(img)

if __name__ == '__main__':
    file = "/home/eddie/Downloads/yolo_img/images/1.jpg"
    image = cv2.imread(file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    bboxes = [0.3363633362071983 ,0.4305174590559635 ,0.6727266724143967, 0.48243874948711896]
    img = visualize_bbox_yolo_single(image,bboxes,"wayz_logo")
    plt.imshow(img)
    plt.show()
