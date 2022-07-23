import torch
import matplotlib.pyplot as plt
from anchor_boxes import multibox_prior
from anchor_boxes import show_bboxes


img = plt.imread("img_bbox/catdog.jpeg")
print(f"Image shape: {img.shape}")
h, w = img.shape[:2]
print(f"Height: {h} -- Width: {w}")




""" generate anchor boxes on the feature map with each unit (pixel) as the anchor
    box center. (x,y)-axis coordinate value in the anchor boxes are divided by the
    width and height of the feature map """
def display_anchors(fmap_w, fmap_h, s):
    # set_figsize()
    # values on the first two dimensions do not affect the output
    fmap = torch.zeros((1, 10, fmap_h, fmap_w))
    anchors = multibox_prior(fmap, scales=s, ratios=[1, 2, 0.5])
    bbox_scale = torch.tensor((w, h, w, h))
    show_bboxes(plt.imshow(img).axes, anchors[0] * bbox_scale)
    plt.show()




display_anchors(fmap_w=4, fmap_h=4, s=[0.15])
display_anchors(fmap_w=2, fmap_h=2, s=[0.4])
display_anchors(fmap_w=1, fmap_h=1, s=[0.8])




