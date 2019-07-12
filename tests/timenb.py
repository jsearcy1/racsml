from time import time
import os
from glob import glob
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import pdb

def scan_nb(nb):
    errors=[]
    for cell in nb['cells']:
        if 'outputs' in cell:
            for output in cell['outputs']:
                if output['output_type'] == 'error':
                    if output['ename'] != 'AssertionError':
                        errors.append(output['evalue'])
    if errors != []:
        print('Notebook Failed with the following Errors')
        print('-----------------------------------------')
        for e in errors:
            print(e)
        print('-----------------------------------------')
        

if __name__=="__main__":
    base_path=os.path.dirname(os.path.abspath(__file__))
    notebook_path=os.path.join(os.path.dirname(base_path),'notebooks')
    
    
    nb_files=glob(os.path.join(notebook_path,'*.ipynb'))

    for nb_path in nb_files:
        print(nb_path)
        itime=time()        
        with open(nb_path) as f:
            nb = nbformat.read(f, as_version=4)
            ep = ExecutePreprocessor(timeout=3600, kernel_name='python3',allow_errors=True)
            try:
                ep.preprocess(nb,{'metadata':{'path':notebook_path}})
            except(TimeoutError):
                print('Notebook Execution Timeout')
                pass
        ftime=time()
        scan_nb(nb)
        
        print(nb_path, "Executes in:", ftime-itime, " S")

        
