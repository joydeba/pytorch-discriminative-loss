import os
import cv2
import json
import numpy as np
import math  
from xml.dom import minidom


source_folder = os.path.join(os.getcwd(), "ethz_1/images")
# json_path = "maskGen_json.json"                     # Relative to root directory
count = 0                                           # Count of total images saved
file_bbs = {}                                       # Dictionary containing polygon coordinates for mask
MASK_WIDTH = 1024									# Dimensions should match those of ground truth image
MASK_HEIGHT = 1024									

file_bbs = {}



xmlpath = 'ethz_1/rotated_xml'
xmlpath_rec = 'ethz_1/axis_aligned_xml'
for filename in os.listdir(xmlpath):
    if not filename.endswith('.xml'): continue
    fullname = os.path.join(xmlpath, filename)
    fullname_rec = os.path.join(xmlpath_rec, filename)
    fname = os.path.splitext(filename)[0]
    xmldoc = minidom.parse(fullname)
    xmldoc_rec = minidom.parse(fullname_rec)
    itemlist_rec = xmldoc_rec.getElementsByTagName('bndbox')
    itemlist = xmldoc.getElementsByTagName('robndbox')
    print(len(itemlist))
    all_points = []
    for s, s_rec in zip(itemlist, itemlist_rec):
        rec_points =[]
        x0 = float(s.getElementsByTagName('cx')[0].firstChild.nodeValue)
        y0 = float(s.getElementsByTagName('cy')[0].firstChild.nodeValue)
        w = float(s.getElementsByTagName('w')[0].firstChild.nodeValue) 
        h = float(s.getElementsByTagName('h')[0].firstChild.nodeValue)
        theta = float(s.getElementsByTagName('angle')[0].firstChild.nodeValue)

        # x1 = float(s_rec.getElementsByTagName('xmin')[0].firstChild.nodeValue)
        # y1 = float(s_rec.getElementsByTagName('ymin')[0].firstChild.nodeValue)
        # x2 = float(s_rec.getElementsByTagName('xmax')[0].firstChild.nodeValue) 
        # y2 = float(s_rec.getElementsByTagName('ymax')[0].firstChild.nodeValue)

        x1 = x0 -(w/2)
        y1 = y0 - (h/2)
        x2 = x0 + (w/2)
        y2 = y0 + (h/2)

        xmin = x0+(x1-x0)*math.cos(theta)+(y1-y0)*math.sin(theta)
        ymin = y0-(x1-x0)*math.sin(theta)+(y1-y0)*math.cos(theta)

        xmax = x0+(x2-x0)*math.cos(theta)+(y2-y0)*math.sin(theta)
        ymax = y0-(x2-x0)*math.sin(theta)+(y2-y0)*math.cos(theta)



        rec_points.append([xmin,ymin])
        rec_points.append([xmin,ymax])
        rec_points.append([xmax,ymax])
        rec_points.append([xmax,ymin])
        

        
        

        all_points.append(rec_points)
    file_bbs[fname] = all_points

			
print("\nDict size: ", len(file_bbs))


to_save_folder = os.path.join(source_folder)
mask_folder = os.path.join(to_save_folder, "insmasks")

for itr in file_bbs:
    mask = np.zeros((MASK_WIDTH, MASK_HEIGHT))
    for obj in file_bbs[itr]:
        try:
            arr = np.array(obj)
        except:
            print("Not found:", obj)
            continue
        rcolor = list(np.random.random(size=3) * 256)
        # rcolor = (255)    
        cv2.fillPoly(mask, np.int32([arr]), color=rcolor)
    count += 1    
    cv2.imwrite(os.path.join(mask_folder, itr + ".jpg") , mask)
        
print("Images saved:", count)