import tensorflow as tf
import keras
import os

this_folder=os.path.abspath(os.path.dirname(__file__))

os.system("git clone https://github.com/pierluigiferrari/ssd_keras.git "+os.path.join(this_folder,'ssd_keras'))





print("Tests Successful")