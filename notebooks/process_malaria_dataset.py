#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 11:32:05 2019

@author: jsearcy
"""
import os
import json
import pdb
from PIL import Image
from shutil import copyfile
from random import random

images_dir ='/home/jsearcy/Desktop/ML Data Sets/malaria/'
use_list='all'
#[1,2,3,4,5,6]
scale_height=1200
scale_width=1600


def write_new_json(input_files,output_filename,output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(os.path.join(output_folder,"train"))
        os.makedirs(os.path.join(output_folder,"develop"))
        os.makedirs(os.path.join(output_folder,"test"))

    train_split=0.85
    develop=0.10
    test=0.05

    cat_dict={}
    cat_dict["ring"]=1
#    cat_dict["red blood cell"]=2
    cat_dict['trophozoite']=2
    cat_dict['schizont']=3
#    cat_dict['difficult']=4
    cat_dict['gametocyte']=4
#    cat_dict['leukocyte']=6
    
    categories_json=[]
    for c in cat_dict:
        categories_json.append({"supercategory": "cell","id": cat_dict[c],"name": c})
   
    all_json=[
             {"info":{},"licenses":{},"images":[],"annotations": [],
              "categories": categories_json              
              }
             for i in range(3)
             ]
    
    id_v=0
    ann_id=0
    for input_file in input_files:
     
        for image in input_file:
            rval=random()
            t_class=0
            if rval < test:
                t_class=2
            elif rval < develop:
                t_class=1
                
                
            
            height=image['image']['shape']['r']
            width=image['image']['shape']['c']
            fname=os.path.abspath(images_dir+image['image']['pathname'])
            
            if t_class==0:
                new_fname=os.path.abspath(os.path.join(output_folder,"train",os.path.basename(fname) ))
            if t_class==1:
                new_fname=os.path.abspath(os.path.join(output_folder,"develop",os.path.basename(fname) ))
            if t_class==2:
                new_fname=os.path.abspath(os.path.join(output_folder,"test",os.path.basename(fname) ))


            height_offset=0
            width_offset=0
            if height > scale_height:height_offset=height-scale_height
            if width > scale_width: width_offset=width-scale_width

            if height < scale_height:continue
            if width < scale_width:continue


            skip=True
            
            for anno in image['objects']:
                if anno['category'] not in cat_dict: continue
            
                bbox=[0,0,0,0]
                bbox[0]=anno['bounding_box']['minimum']['c']-width_offset
                bbox[1]=anno['bounding_box']['minimum']['r']-height_offset
                
                bbox[2]=anno['bounding_box']['maximum']['c']-bbox[0]-width_offset
                bbox[3]=anno['bounding_box']['maximum']['r']-bbox[1]-height_offset
                if any([b < 0 for b in bbox]):
                    continue
                skip=False    
        
        
                all_json[t_class]['annotations'].append({ 
                                                "segmentation": [[bbox]],
                                                "bbox": bbox,
                                                "category_id": cat_dict[anno['category']],
                                                "image_id":id_v,
                                                "id":ann_id,
                                                })
                ann_id+=1
            if skip:continue


            if height_offset !=0 or width_offset !=0:
                pixel_data=Image.open(fname) 
                pixel_data=pixel_data.crop((width_offset,height_offset,width,height))
                pixel_data.save(new_fname)
            else:
                copyfile(fname,new_fname)
            
 
            
        
        
            all_json[t_class]["images"].append({"file_name":new_fname,
                                       "height": image['image']['shape']['r'],
                                       "width": image['image']['shape']['c'],
                                       "id":id_v})
        
        
            id_v+=1
       
  
    open(os.path.join(output_folder,'train',output_filename),"w").write(json.dumps(all_json[0]))
    open(os.path.join(output_folder,'develop',output_filename),"w").write(json.dumps(all_json[1]))
    open(os.path.join(output_folder,'test',output_filename),"w").write(json.dumps(all_json[2]))


train=json.load(open("/home/jsearcy/Desktop/ML Data Sets/malaria/training.json"))
test=json.load(open("/home/jsearcy/Desktop/ML Data Sets/malaria/test.json"))

#write_new_json(train,'output_train.json')
write_new_json([test,train],'output.json','/home/jsearcy/Desktop/new_malaria/')