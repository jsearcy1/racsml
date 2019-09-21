#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 09:21:51 2018

@author: jsearcy
"""

from PIL import Image,ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform
import pdb
import pickle
import os

class BBToyData():

    def __init__(self,multi_object=False):

        if multi_object:
            if not all([os.path.exists(i) for i in ['bb_toy_split_multi.npy','bb_toy_images_multi.npy','bb_toy_labels_multi.pk']]):        
                self.make_data(multi_object=True)
            train,develop,test=np.load('bb_toy_split_multi.npy')
            image_data=np.load('bb_toy_images_multi.npy')
            bb_toy_labels=pickle.load(open('bb_toy_labels_multi.pk','rb'))

        else:
            if not all([os.path.exists(i) for i in ['bb_toy_split.npy','bb_toy_images.npy','bb_toy_labels.pk']]):        
                self.make_data(multi_object=False)

                
            train,develop,test=np.load('bb_toy_split.npy')
            image_data=np.load('bb_toy_images.npy')
            bb_toy_labels=pickle.load(open('bb_toy_labels.pk','rb'))

        self.X_train=np.expand_dims(image_data[train],-1)
        self.X_develop=np.expand_dims(image_data[develop],-1)
        self.X_test=np.expand_dims(image_data[test],-1)

        self.Y_train=[bb_toy_labels[i] for i in train]
        self.Y_develop=[bb_toy_labels[i] for i in develop]
        self.Y_test=[bb_toy_labels[i] for i in test]

            
            
            
    def smallest_dist(self,points):
        return np.min(np.abs(points[0]-points[1]))

    def area(self,points):
        return np.sum((points[0]-points[1])**2)**.5

        
    def make_data(self,multi_object=False):
        

        if multi_object:
            start_size=200
            final_size=100
        else:
            start_size=200
            final_size=50
        arrays=[]
        scores=[]
        train=[]
        test=[]
        develop=[]
        
        labels=[]
        
        for index in range(20000):
            select= np.random.uniform()
            
            if select < 0.8:
                train.append(index)
            elif select < 0.9:                   
                test.append(index)
            else:
                develop.append(index)
 
            im=Image.fromarray(np.zeros((start_size,start_size)))
            if multi_object:
                n_objects=np.random.randint(10)
            else:
                n_objects=np.random.uniform()<0.8 # 1 most of the time 0 otherwise
            labels.append([])
            
            for i in range(n_objects):
                _points=np.random.uniform(.1,.9,size=(2,2))
                if multi_object:
                    while self.area(_points) <0.1 or self.area(_points)>.3 or self.smallest_dist(_points) < .05:
                        _points=np.random.uniform(.1,.9,size=(2,2))
                else:
                    while self.area(_points) <0.5 or self.area(_points)>.9 or self.smallest_dist(_points) < .05:
                        _points=np.random.uniform(.1,.9,size=(2,2))

                
                #Put points in correct order
                x_i=np.min(_points[:,0])
                x_f=np.max(_points[:,0])
                y_i=np.min(_points[:,1])
                y_f=np.max(_points[:,1])

                points=np.array([[x_i,y_i],[x_f,y_f]])

                p1,p2=(points*start_size).astype('int32')
                coords=(tuple(p1),tuple(p2)) #I don't know why this is necessary?



                category=np.random.randint(4)
                draw=ImageDraw.Draw(im)
                bbox=[x_i*final_size,y_i*final_size,(x_f-x_i)*final_size,(y_f-y_i)*final_size]
                if category==0:
                    draw.rectangle(coords,outline=255.,fill=255.)
                    labels[-1].append([category,bbox])
                elif category ==1:
                    draw.ellipse(coords,outline=255.,fill=255.)
                    labels[-1].append([category,bbox])
                elif category ==2:
                    p3=(np.array([x_i,y_f])*start_size).astype('int32')
                    Tcoords=[tuple(p1),tuple(p2),tuple(p3)] #I don't know why this is necessary?                    
                    
                    draw.polygon(Tcoords,fill=255.)
                    labels[-1].append([category,bbox])


            array=transform.resize(np.asarray(im).astype('uint8'),(final_size,final_size),anti_aliasing=True)   
            if np.max(array) !=0:
                array=1./np.max(array)*array

            if np.isnan(array).any():pdb.set_trace()
            arrays.append(np.expand_dims(array,0))

        assert(len(train) > 20000*.7)

        output_data=np.concatenate(arrays)
        if multi_object:
            np.save('bb_toy_split_multi',(train,develop,test))
            np.save('bb_toy_images_multi',output_data)
            pickle.dump(labels,open('bb_toy_labels_multi.pk','wb'))
        else:
            np.save('bb_toy_split',(train,develop,test))
            np.save('bb_toy_images',output_data)
            pickle.dump(labels,open('bb_toy_labels.pk','wb'))


if __name__=="__main__":
    b=BBToyData()
