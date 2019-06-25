import os
import numpy as np

class WineData():
    def __init__(self):
                
        if not os.path.exists('winequality-red.csv'):
            print('Downloading')
            os.system('wget https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv')
            os.system('wget https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv')
            os.system('wget https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality.names')
        self.reds,self.reds_target,self.reds_header=self.read_data('winequality-red.csv')
        self.whites,self.whites_target,self.whites_header=self.read_data('winequality-white.csv')
        assert self.whites_header==self.reds_header, 'Header mis-match check input files, winequality-red/white.csv!'         
        self.header=self.reds_header.split(';')[0:-1] #Remove target variable
        self.header.append('red')
        self.all=np.concatenate([self.reds,self.whites],axis=0)
        self.all_targets=np.concatenate([self.reds_target,self.whites_target],axis=0)
        
        
        train=[]
        develop=[]
        test=[]
        for i in range(len(self.all)):
            if np.random.uniform()>0.9:test.append(i)
            if np.random.uniform()>0.7:develop.append(i)
            else:train.append(i)
        self.x_train=self.all[train]
        self.x_develop=self.all[develop]
        self.x_test=self.all[test]

        self.y_train=self.all_targets[train]
        self.y_develop=self.all_targets[develop]
        self.y_test=self.all_targets[test]
    
    
    
    
    def read_data(self,file):
        data=[]
        target=[]
        header=None
        red='winequality-red' in file
        for l in open(file,'r').readlines():
            if header==None:
                header=l
                continue
            if l=='':continue
            float_data=[float(i.strip('\n')) for i in l.split(';')]
            target_value=float_data[-1]>5
            float_data=float_data[:-1]
            float_data.append(red)
            float_data=np.array(float_data)
            target.append(target_value)                        
            data.append(np.expand_dims(float_data,0))
        data=np.concatenate(data,axis=0)
        return data,np.array(target),header
            
            
            
if __name__=='__main__':
    wd=WineData()