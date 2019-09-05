#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 16:34:29 2019

@author: jsearcy
"""
import tensorflow as tf
import os
import numpy as np
class CamelyOnPatch():
    
    def normalize(self,image):
        out_image=image/255.    
 #       target_area=np.zeros((96,96,1),dtype='float32')
 #       target_area[32:64,32:64,:]=1
 #       out_image=np.concatenate([out_image,target_area],axis=-1)
        return out_image
        
        
    def __init__(self,datadir):
   
        self.datadir=datadir
        print( os.path.join(datadir,'camelyonpatch_level_2_split_train_x.h5'))
        self.X_train=tf.keras.utils.HDF5Matrix(
              os.path.join(datadir,'camelyonpatch_level_2_split_train_x.h5'),'x',normalizer=self.normalize)
        self.Y_train=np.squeeze(tf.keras.utils.HDF5Matrix(os.path.join(datadir,'camelyonpatch_level_2_split_train_y.h5'),'y'))
                                  
        
    
    
    
    
        self.datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                #rescale=1./255.,
                width_shift_range=0,  # randomly shift images horizontally
                height_shift_range=0,  # randomly shift images vertically 
                horizontal_flip=True,  # randomly flip images
                vertical_flip=True,
                rotation_range=45
            
                
            )  # randomly flip images
        
        
        
        self.X_develop=tf.keras.utils.HDF5Matrix(os.path.join(datadir,'camelyonpatch_level_2_split_valid_x.h5'),'x',normalizer=self.normalize)
        self.Y_develop=np.squeeze(np.array(tf.keras.utils.HDF5Matrix(os.path.join(datadir,'camelyonpatch_level_2_split_valid_y.h5'),'y')))
        self.X_test=tf.keras.utils.HDF5Matrix(os.path.join(datadir,'camelyonpatch_level_2_split_test_x.h5'),'x',normalizer=self.normalize)
        self.Y_test=np.squeeze(np.array(tf.keras.utils.HDF5Matrix(os.path.join(datadir,'camelyonpatch_level_2_split_test_y.h5'),'y')))
    
    def scan_tiff(self,tiff_file='/home/jsearcy/Desktop/ML Data Sets/pcamv1/test_013.tif',model=None):
        import openslide
        import matplotlib.pyplot as plt
        
        t1=[(48770.9,157277),
            (48728.1,157343),
            (48674,157402),
            (48608.7,157442),
            (48539.7,157475),
            (48468.8,157509),
            (48411,157557),
            (48396.1,157632),
            (48440.8,157695),
            (48515.4,157718),
            (48593.8,157716),
            (48657.2,157669),
            (48716.9,157621),
            (48765.4,157555),
            (48815.7,157499),
            (48849.3,157429),
            (48867.9,157352),
            (48856.7,157277)]


        t2=[(49115.9,155007),
            (49124.3,155123),
            (49178.3,155230),
            (49163.5,155345),
            (49183.1,155440),
            (49230.9,155557),
            (49231.5,155667),
            (49170.1,155749),
            (49155.2,155854),
            (49170.1,155954),
            (49202.5,156054),
            (49244.9,156148),
            (49344.6,156173),
            (49394.5,156084),
            (49414.4,155981),
            (49456.7,155882),
            (49389.4,155788),
            (49379.5,155692),
            (49367.1,155592),
            (49382,155488),
            (49354.6,155388),
            (49327.2,155288),
            (49297.3,155189),
            (49270,155093),
            (49178.2,154978)]
        
        
        
        
        
        slide_image=openslide.OpenSlide(tiff_file)
        coords=slide_image.level_dimensions[0]
        sfactor=(coords[0]/800.,coords[1]/400.)
        t1=[(i[0]/sfactor[0],i[1]/sfactor[1]) for i in t1]
        t2=[(i[0]/sfactor[0],i[1]/sfactor[1]) for i in t2]


        res=slide_image.level_dimensions[2]
        
        image=np.asarray(slide_image.read_region( (30000,80000),2,(800*16,1600*16)  ))
        res_x,res_y,chann=image.shape
        new_image=np.zeros((res_x//32,res_y//32,3))
        mask=np.zeros((res_x//32,res_y//32))
        
        for x in range(0,res_x-96,32):
            pred_data=[]
            for y in range(0,res_y-96,32):
                data=image[x:x+96,y:y+96,:]
                pred_data.append(np.expand_dims(data,0)/255.)                
                color=np.mean(np.mean(data[32:64,32:64,:],axis=0),axis=0)
#                print(color)
#                print(x//32,y//32)
                new_image[x//32,y//32,:]=color[0:3]
            pred_data.append(data)
            if model!=None:
                mask[x//32,:]= model.predict(np.concatenate(pred_data,axis=0))
            
                
        return image,new_image,mask,[t1,t2]
#        plt.imshow(new_image)
#        plt.show()
        
if __name__=="__main__":
    import matplotlib.pyplot as plt
    cp=CamelyOnPatch('/home/jsearcy/Desktop/ML Data Sets/pcamv1')
    image,new_image,mask,[t1,t2]=cp.scan_tiff()
    
    f=plt.figure(figsize=(40,120))
    img = plt.imshow(image)    
    plt.scatter(*zip(*t1))
    plt.scatter(*zip(*t2))
    plt.axis('off')
    f.savefig("../assets/full_slide.png", bbox_inches='tight')
    plt.close(f)
