# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 02:15:58 2021

@author: User
"""
# pip install pycocotools
# pip install plotly
import os

import numpy as np 
import cv2
import matplotlib.pyplot as plt
# from PIL import Image
import PIL.Image
from pycocotools.coco import COCO
from pycocotools import mask
from skimage import measure
import mmcv

train_data_path = os.listdir('/content/dataset/train/')
os.makedirs('dataset_new/train_jpg', exist_ok=True)

# test image .png change to .jpg
# test_data_path = os.listdir('./dataset/test/')
# os.makedirs('dataset_new/test_jpg', exist_ok=True)
# for index in range(len(test_data_path)):
#     path = test_data_path[index]
#     filename = './dataset_new/test_jpg/' + path.replace('.png', '.jpg')   
#     imgs = cv2.imread('./dataset/test/' + path)
#     cv2.imwrite(filename, imgs)
    
coco_output = {
    "images" : [],
    "categories" : [],
    "annotations" : []
    }

categories = [{'id' : 1, 'name' : 'Nuclei'}]
coco_output['categories'] = categories
annotation_id = 0
for index in range(len(train_data_path)):
    # load images and masks
    path = train_data_path[index]
    img = PIL.Image.open('/content/dataset/train/' + path + '/images/' + path + '.png').convert('RGB')
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    # note that we haven't converted the mask to RGB,
    # because each color corresponds to a different instance
    # with 0 being background
    img = np.array(img)
    
    mask_path = os.listdir('/content/dataset/train/' + path + '/masks/')
    height, width = img.shape[0], img.shape[1]
    # filename = './data/train_jpg/' + path + '.jpg'    
    # imgs = cv2.imread('./train/' + path + '/images/' + path + '.png')
    # cv2.imwrite(filename, imgs)
    image_dict = {
        "file_name" : path + '.png', # file_name
        "height" : height,
        "width" : width,
        "id" : index  # image_id
        }    
    coco_output['images'].append(image_dict)
    
    for i in mask_path:
        if i == ".ipynb_checkpoints":
            continue
        masks = PIL.Image.open('/content/dataset/train/' + path + '/masks/' + i)
        masks = np.array(masks)
        binary_mask = np.asfortranarray(masks)
        encoded_mask = mask.encode(binary_mask)
        area_mask = mask.area(encoded_mask)
        bbox_mask = mask.toBbox(encoded_mask)
        contours = measure.find_contours(masks, 0.5)
        
        ann_dict = {
            "id" : annotation_id,
            "image_id" : index,  # image_id
            "category_id" : 1,
            "bbox" : bbox_mask.tolist(),
            "area" : area_mask,
            "iscrowd" : 0,  # suppose all instances are not crowd
            "segmentation" : [],
            }
        for c in contours:
            c = np.flip(c, axis=1)
            segmentation = c.ravel().tolist()
            ann_dict["segmentation"].append(segmentation)
        coco_output["annotations"].append(ann_dict)
        annotation_id += 1
mmcv.dump(coco_output, 'train_all.json')    

        
#%%

# # 檢查mask是否標記正確
# def main():

#     coco_annotation_file_path = "train.json"  # data/annotation/train.json

#     coco_annotation = COCO(annotation_file=coco_annotation_file_path)

#     # Category IDs.
#     cat_ids = coco_annotation.getCatIds(catNms = ['Nuclei'])

#     # All categories.
#     cats = coco_annotation.loadCats(cat_ids)
#     cat_names = [cat["name"] for cat in cats]
#     print("Categories Names:")
#     print(cat_names)

#     # Category ID -> Category Name.
#     query_id = cat_ids[0]
#     query_annotation = coco_annotation.loadCats([query_id])[0]
#     query_name = query_annotation["name"]
#     # query_supercategory = query_annotation["supercategory"]
#     print("Category ID -> Category Name:")
#     print(
#         f"Category ID: {query_id}, Category Name: {query_name}"
#     )

#     # Category Name -> Category ID.
#     query_name = cat_names[0]
#     query_id = coco_annotation.getCatIds(catNms=[query_name])[0]
#     print("Category Name -> ID:")
#     print(f"Category Name: {query_name}, Category ID: {query_id}")

#     # Get the ID of all the images containing the object of the category.
#     img_ids = coco_annotation.getImgIds(catIds=[query_id])
#     print(f"Number of Images Containing {query_name}: {len(img_ids)}")

#     # Pick one image.
#     img_id = img_ids[1]
#     img_info = coco_annotation.loadImgs([img_id])[0]
#     img_file_name = img_info["file_name"]

#     print(
#         f"Image ID: {img_id}, File Name: {img_file_name}"
#     )

#     # Get all the annotations for the specified image.
#     ann_ids = coco_annotation.getAnnIds(imgIds=[img_id], iscrowd=None)
#     anns = coco_annotation.loadAnns(ann_ids)
#     print(f"Annotations for Image ID {img_id}:")
#     print(anns)

#     # Use URL to load image.
#     im = PIL.Image.open('./data/train_jpg/' + img_file_name)

#     # Save image and its labeled version.
#     plt.axis("off")
#     plt.imshow(np.asarray(im))
#     plt.savefig(f"{img_id}.jpg", bbox_inches="tight", pad_inches=0)
#     # Plot segmentation and bounding box.
#     coco_annotation.showAnns(anns, draw_bbox=True)
#     plt.savefig(f"{img_id}_annotated.jpg", bbox_inches="tight", pad_inches=0)

#     return


# if __name__ == "__main__":
#     main()   
        
        
        
# #%%
# # 檢查coco格式否都能正確讀取
# coco_coco = COCO('data/annotation/train.json')
# Catids = coco_coco.getCatIds()[0]
# Imgids = coco_coco.getImgIds(catIds= Catids)
# img = coco_coco.loadImgs(Imgids)[0]
# img_path = 'data/train_jpg/' + img['file_name']
# I = PIL.Image.open(img_path)
# plt.axis('off')
# annIds = coco_coco.getAnnIds(imgIds = img['id'], catIds = Catids, iscrowd = None)
# anns = coco_coco.loadAnns(annIds)
# anns

    
    
    