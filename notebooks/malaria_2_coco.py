#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 15:01:45 2019

@author: jsearcy
"""
import h5py
import json
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES']='0'

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, CSVLogger
from keras import backend as K
from keras.models import load_model
from math import ceil
import numpy as np
from matplotlib import pyplot as plt

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast
from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms
from data_generator.data_augmentation_chain_variable_input_size import DataAugmentationVariableInputSize
from data_generator.data_augmentation_chain_constant_input_size import DataAugmentationConstantInputSize
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation



from models.keras_ssd7 import build_model
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast




images_dir ='/home/jsearcy/Desktop/ML Data Sets/malaria/'
use_list='all'
#[1,2,3,4,5,6]

def write_new_json(input_file,output_filename,folder):
    cat_dict={}
    cat_dict["ring"]=1
#    cat_dict["red blood cell"]=2
    cat_dict['trophozoite']=2
    cat_dict['schizont']=3
    cat_dict['difficult']=4
    cat_dict['gametocyte']=5
    cat_dict['leukocyte']=6
    
    categories_json=[]
    for c in cat_dict:
        categories_json.append({"supercategory": "cell","id": cat_dict[c],"name": c})
        
    new_json={"info":{},"licenses":{},"images":[],"annotations": [],
              "categories": categories_json
              
              }
    id_v=0
    ann_id=0
    for image in input_file:
        
        
        
        skip=True
        for anno in image['objects']:
            if anno['category'] =="red blood cell": continue
            skip=False    
        
            bbox=[0,0,0,0]
            bbox[0]=anno['bounding_box']['minimum']['c']
            bbox[1]=anno['bounding_box']['minimum']['r']
            
            bbox[2]=anno['bounding_box']['maximum']['c']-bbox[0]
            bbox[3]=anno['bounding_box']['maximum']['r']-bbox[1]
    
    
            new_json['annotations'].append({ 
                                            "segmentation": [[bbox]],
                                            "bbox": bbox,
                                            "category_id": cat_dict[anno['category']],
                                            "image_id":id_v,
                                            "id":ann_id,
                                            })
            ann_id+=1
        if skip:continue
        new_json["images"].append({"file_name":os.path.join(folder,image['image']['pathname']),
                                   "height": image['image']['shape']['r'],
                                   "width": image['image']['shape']['c'],
                                   "id":id_v})
    
    
        id_v+=1
  
    open(output_filename,"w").write(json.dumps(new_json))

train=json.load(open("/home/jsearcy/Desktop/ML Data Sets/malaria/training.json"))
test=json.load(open("/home/jsearcy/Desktop/ML Data Sets/malaria/test.json"))

write_new_json(train,'/home/jsearcy/UO Data Science Dropbox/Public Datasets/BBBC041o/train.json',folder='train')
write_new_json(test,'/home/jsearcy/UO Data Science Dropbox/Public Datasets/BBBC041o/test.json',folder='test')

oa.aosefoeji

train_dataset = DataGenerator(load_images_into_memory=True, hdf5_dataset_path=None)
train_dataset.parse_json(        images_dirs=images_dir,
                                 annotations_filenames=["output_train.json"],
                                 ground_truth_available=True,
                                 include_classes=use_list
                                 )

val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)

val_dataset.parse_json(        images_dirs=images_dir,
                                 annotations_filenames=["output_test.json"],
                                 ground_truth_available=True,
                                 include_classes=use_list)

val_dataset.create_hdf5_dataset(file_path='val_dataset.h5',
                                  resize=(1200,1600),
                                  variable_image_size=True,
                                  verbose=True)



train_generator = train_dataset.generate(batch_size=1,
                                         shuffle=True,
                                         transformations=[],
                                         label_encoder=None,
                                         returns={'processed_images',
                                                  'processed_labels',
                                                  'filenames'},
                                         keep_images_without_gt=False)

batch_images, batch_labels, batch_filenames = next(train_generator)


# 5: Draw the predicted boxes onto the image

plt.figure(figsize=(20,12))
plt.imshow(batch_images[0])

current_axis = plt.gca()

colors = plt.cm.hsv(np.linspace(0, 1, 7)).tolist() # Set the colors for the bounding boxes

# Draw the ground truth boxes in green (omit the label for more clarity)
for box in batch_labels[0]:
    xmin = box[1]
    ymin = box[2]
    xmax = box[3]
    ymax = box[4]
#    label = '{}'.format(classes[int(box[0])])
    current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color='green', fill=False, linewidth=2))  
    #current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':'green', 'alpha':1.0})
print(batch_labels)





# 1: Build the Keras model

K.clear_session() # Clear previous models from memory.









img_height = 1200 # Height of the input images
img_width = 1600 # Width of the input images
img_channels = 3 # Number of color channels of the input images
intensity_mean = 127.5 # Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
intensity_range = 127.5 # Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
n_classes = 7 # Number of positive classes
scales = [0.08, 0.16, 0.32, 0.64, 0.96] # An explicit list of anchor box scaling factors. If this is passed, it will override `min_scale` and `max_scale`.
aspect_ratios = [0.5, 1.0, 2.0] # The list of aspect ratios for the anchor boxes
two_boxes_for_ar1 = True # Whether or not you want to generate two anchor boxes for aspect ratio 1
steps = None # In case you'd like to set the step sizes for the anchor box grids manually; not recommended
offsets = None # In case you'd like to set the offsets for the anchor box grids manually; not recommended
clip_boxes = False # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
variances = [1.0, 1.0, 1.0, 1.0] # The list of variances by which the encoded target coordinates are scaled
normalize_coords = True # Whether or not the model is supposed to use coordinates relative to the image size



model = build_model(image_size=(img_height, img_width, img_channels),
                    n_classes=n_classes,
                    mode='training',
                    l2_regularization=0.0005,
                    scales=scales,
                    aspect_ratios_global=aspect_ratios,
                    aspect_ratios_per_layer=None,
                    two_boxes_for_ar1=two_boxes_for_ar1,
                    steps=steps,
                    offsets=offsets,
                    clip_boxes=clip_boxes,
                    variances=variances,
                    normalize_coords=normalize_coords,
                    subtract_mean=intensity_mean,
                    divide_by_stddev=intensity_range)







# 5: Instantiate an encoder that can encode ground truth labels into the format needed by the SSD loss function.

# The encoder constructor needs the spatial dimensions of the model's predictor layers to create the anchor boxes.
predictor_sizes = [model.get_layer('classes4').output_shape[1:3],
                   model.get_layer('classes5').output_shape[1:3],
                   model.get_layer('classes6').output_shape[1:3],
                   model.get_layer('classes7').output_shape[1:3]]

ssd_input_encoder = SSDInputEncoder(img_height=img_height,
                                    img_width=img_width,
                                    n_classes=n_classes,
                                    predictor_sizes=predictor_sizes,
                                    scales=scales,
                                    aspect_ratios_global=aspect_ratios,
                                    two_boxes_for_ar1=two_boxes_for_ar1,
                                    steps=steps,
                                    offsets=offsets,
                                    clip_boxes=clip_boxes,
                                    variances=variances,
                                    matching_type='multi',
                                    pos_iou_threshold=0.5,
                                    neg_iou_limit=0.3,
                                    normalize_coords=normalize_coords)

# 6: Create the generator handles that will be passed to Keras' `fit_generator()` function.
batch_size=5
train_generator = train_dataset.generate(batch_size=batch_size,
                                         shuffle=True,
                                         transformations=[],
                                         label_encoder=ssd_input_encoder,
                                         returns={'processed_images',
                                                  'encoded_labels'},
                                         keep_images_without_gt=False)

val_generator = val_dataset.generate(batch_size=batch_size,
                                     shuffle=False,
                                     transformations=[],
                                     label_encoder=ssd_input_encoder,
                                     returns={'processed_images',
                                              'encoded_labels'},
                                     keep_images_without_gt=False)










adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

initial_epoch   =5
final_epoch     = 20
steps_per_epoch = 1000

history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=steps_per_epoch,
                              epochs=final_epoch,
                              workers=1,
                              #callbacks=callbacks,
                              #validation_data=val_generator,
                              #validation_steps=ceil(val_dataset_size/batch_size),
                              initial_epoch=initial_epoch)




# 1: Set the generator for the predictions.

predict_generator = train_dataset.generate(batch_size=1,
                                         shuffle=True,
                                         transformations=[],
                                         label_encoder=None,
                                         returns={'processed_images',
                                                  'processed_labels',
                                                  'filenames'},
                                         keep_images_without_gt=False)


batch_images, batch_labels, batch_filenames = next(predict_generator)

i = 0 # Which batch item to look at

print("Image:", batch_filenames[i])
print()
print("Ground truth boxes:\n")
print(batch_labels[i])

y_pred = model.predict(batch_images)
y_pred_decoded = decode_detections(y_pred,
                                   confidence_thresh=0.5,
                                   iou_threshold=0.1,
                                   top_k=200,
                                   normalize_coords=normalize_coords,
                                   img_height=img_height,
                                   img_width=img_width)

np.set_printoptions(precision=2, suppress=True, linewidth=90)
print("Predicted boxes:\n")
print('   class   conf xmin   ymin   xmax   ymax')
print(y_pred_decoded[i])
#f = h5py.File("/home/jsearcy/Desktop/ML Data Sets/malaria/data.h5", "w")
#f.create_dataset("test",)
plt.figure(figsize=(20,12))
plt.imshow(batch_images[i])

current_axis = plt.gca()


cat_dict={}
cat_dict["ring"]=1
cat_dict["red blood cell"]=2
cat_dict['trophozoite']=3
cat_dict['schizont']=4
cat_dict['difficult']=5
cat_dict['gametocyte']=6
cat_dict['leukocyte']=7

inv_dict={}
for v,ind in cat_dict.items():
    inv_dict[ind]=v

colors = plt.cm.hsv(np.linspace(0, 1, n_classes+1)).tolist() # Set the colors for the bounding boxes
classes = [inv_dict[i+1] for i in range(len(inv_dict))] # Just so we can print class names onto the image instead of IDs

# Draw the ground truth boxes in green (omit the label for more clarity)
for box in batch_labels[i]:
    xmin = box[1]
    ymin = box[2]
    xmax = box[3]
    ymax = box[4]
    color = colors[int(box[0])]

    label = '{}'.format(classes[int(box[0])])
    current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))  
    #current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':'green', 'alpha':1.0})

# Draw the predicted boxes in blue
for box in y_pred_decoded[i]:
    xmin = box[-4]
    ymin = box[-3]
    xmax = box[-2]
    ymax = box[-1]
    #olor = colors[int(box[0])]
    label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
    current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color='blue', fill=False, linewidth=2))  
#    current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})
