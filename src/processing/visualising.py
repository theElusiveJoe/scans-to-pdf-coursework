from torchvision.transforms.functional import to_pil_image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def draw_img(img, bboxes=None, labels=None, colors=None):
    if colors is None:
        colors = {
            1: 'red',
            2: 'yellow',
            3: 'magenta',
            4: 'blue',
            5: 'green',
        }
    pix = np.array(img.getdata()).reshape(img.size[0], img.size[1], 3)
    plt.subplot()
    plt.imshow(pix)
    if bboxes is not None and labels is not None:
        for bbox, label in zip(bboxes, labels):
            x1,y1,x2,y2 = bbox
            h = (y2-y1)
            w = (x2-x1)
            plt.gca().add_patch(Rectangle((x1,y1),w,h,
                                          edgecolor=colors[label],
                                          facecolor='none',
                                          lw=1))
    plt.show()