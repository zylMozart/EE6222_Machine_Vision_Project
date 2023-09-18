import torch
import numpy as np

def save_outputs(outputs):
    for i in range(len(outputs)):
        res=np.argmax(outputs[i])
        print(i,'\t',res)
    pass

if __name__=='__main__':
    outputs = torch.load('scripts/Analysis/files/outputs.pt')
    save_outputs(outputs)