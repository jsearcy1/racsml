from random import random
import numpy as np
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES']=''

class LinearModel():
    def __init__(self):
        self.slope=tf.Variable(0., name='slope')
        self.intercept= tf.Variable(0., name='intercept')
        self.x=tf.placeholder('float32',shape=(None,1), name='X')
        self.y_true=tf.placeholder('float32',shape=(None,1), name='y_true')
        self.y=self.x*self.slope+self.intercept
        self._loss= tf.losses.mean_squared_error(self.y_true,self.y)
        self.session=tf.Session()
        self.session.run(tf.global_variables_initializer())

        
    def theta(self):
        return self.session.run([self.slope,self.intercept])

    def settheta(self,theta):
        self.session.run(self.slope.assign(theta[0]))
        self.session.run(self.intercept.assign(theta[1]))
    
    def loss(self,x,y_true):
        return self.session.run(self._loss,feed_dict={self.x:x,self.y_true:y_true})

    def _loss_exact(self,x,y_true):
        return self.session.run(self._loss,feed_dict={self.x:x,self.y_true:y_true})

    
    def predict(self,x):
        return self.session.run(self.y,feed_dict={self.x:x})

    def optimize(self,x,y_true,learning_rate,steps=20,batch_size=None):
        grad_v=[]
        path_v=[]

        self.opt=tf.train.GradientDescentOptimizer(learning_rate,use_locking=True)
        grads=self.opt.compute_gradients(self._loss,[self.slope,self.intercept])
        apply_g=self.opt.apply_gradients(grads)

        for i in range(steps):
            if batch_size !=None:
                b_index=np.random.choice(range(len(x)),batch_size)
                batch=x[b_index]
                batch_y=y_true[b_index]
            else:
                batch=x
                batch_y=y_true
            _slopev,_interceptv=self.session.run([self.slope,self.intercept])
            
            out,_=self.session.run([grads,apply_g],feed_dict={self.x:batch,self.y_true:batch_y})

            
            grad_v.append((out[0][0],out[1][0]))
#           path_v.append((out[0][1],out[1][1]))
            path_v.append((_slopev,_interceptv))

        return grad_v,path_v
