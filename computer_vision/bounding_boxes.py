import torch
import matplotlib.pyplot as plt


# read image
img_path = "img_bbox/catdog.jpeg"
img = plt.imread(img_path)


""" bounding box coordinates (estimated)
    cat [350, 250, 610, 650]
    dog [590, 100, 1000, 650]
    Type 1: x,y coordinate of upper left corner and lower right corner
    Type 2: (x,y)-axis coordinates of the bbox center and width, height
"""


# converts from Type 1 to Type 2
def box_corner_to_center(boxes):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = torch.stack((cx, cy, w, h), axis=-1)
    return boxes


# converts from Type 2 to Tpye 1
def box_center_to_corner(boxes):
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = torch.stack((x1, y1, x2, y2), axis=-1)
    return boxes



print("Test if two converstion methods are correct")
cat_bbox = [350, 250, 610, 650]
dog_bbox = [590, 100, 1000, 650]
boxes = torch.tensor((cat_bbox, dog_bbox))
print(f"Boxes: {boxes}")
print(box_center_to_corner(box_corner_to_center(boxes)) == boxes)




def bbox_to_rect(bbox, color):
    w = bbox[2]-bbox[0]
    h = bbox[3]-bbox[1]
    return plt.Rectangle(xy=(bbox[0], bbox[1]), width=w, height=h, 
                         fill=False, edgecolor=color, linewidth=2)



fig = plt.imshow(img)
fig.axes.add_patch(bbox_to_rect(cat_bbox, "blue"))
fig.axes.add_patch(bbox_to_rect(dog_bbox, "red"))

plt.show()
