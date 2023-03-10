import cv2
import json
import sys
import os
import numpy as np

def color2label(path):
    with open(path,'r') as f:
        data = json.load(f)
    
    c2l_dict = {}
    for item in data:
        for c in item["colors"]:
            c2l_dict[tuple(c)] = item["id"]
    
    return c2l_dict

if __name__ == "__main__":
    _cmap = {
        0: (128, 128, 128),    # sky
        1: (128, 0, 0),        # building
        2: (192, 192, 128),    # column_pole
        3: (128, 64, 128),     # road
        4: (0, 0, 192),        # sidewalk
        5: (128, 128, 0),      # Tree
        6: (192, 128, 128),    # SignSymbol
        7: (64, 64, 128),      # Fence
        8: (64, 0, 128),       # Car
        9: (64, 64, 0),        # Pedestrian
        10: (0, 128, 192),     # Bicyclist
        11: (0, 0, 0)}         # Void

    # c2l_dict = color2label("./label_group.json")
    c2l_dict = dict(zip(_cmap.values(), _cmap.keys()))
    print(c2l_dict,len(c2l_dict))

    label_img_dir = sys.argv[1]
    output_dir = label_img_dir+"-idx-with-ignored"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for img in os.listdir(label_img_dir):
        if img.endswith('png') or img.endswith('jpg'):
            color = cv2.imread(os.path.join(label_img_dir,img))[:,:,::-1]
            label = np.zeros(color.shape[:-1],dtype=int)
            for y in range(color.shape[0]):
                for x in range(color.shape[1]):
                    if tuple(color[y,x]) in c2l_dict.keys():
                        label[y,x] = c2l_dict[tuple(color[y,x])]
                    else:
                        label[y,x] = 255
            
            cv2.imwrite(os.path.join(output_dir,img),label.astype(np.uint8))
            print(label_img_dir, img)
            




