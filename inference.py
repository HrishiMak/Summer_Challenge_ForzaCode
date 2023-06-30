from datasets import TestDataset
from torch.utils.data import DataLoader
import numpy as np
import torch
import pandas as pd
from network import ClassificationNet
import argparse

def inference(path_to_checkpoint1,path_to_checkpoint2,path_to_test_csv,path_to_test_imgdir):
    y_proba=[]
    ids=[]
    test_dataset=TestDataset(path_to_test_csv,path_to_test_imgdir)
    test_loader = DataLoader(test_dataset, batch_size=64)
    model_checkpoint1=torch.load(path_to_checkpoint1,map_location=torch.device('cpu'))
    model_checkpoint2=torch.load(path_to_checkpoint2,map_location=torch.device('cpu'))
    with torch.no_grad():
        model_checkpoint1.eval()
        model_checkpoint2.eval()
        for i,data in enumerate(test_loader):
            if not type(data) in (tuple, list):
                data = (data,)
            data = tuple(d.to('cpu') for d in data)
            outputs = model_checkpoint1(data[0],data[1])
            distance=(outputs[0]-outputs[1]).pow(2)
            probas=model_checkpoint2(distance)
            y_proba.extend(probas.cpu().numpy())
    test_df=pd.read_csv(path_to_test_csv)
    for i in range(len(test_df.index)):
        ids.append(test_df.img1_name[i]+"_"+test_df.img2_name[i])
    submission_df=pd.DataFrame({"id":ids,"proba":np.array(y_proba).flatten()})
    submission_df.to_csv("subimission.csv",index=False)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_checkpoint1', type=str, required=True, help='path to model checkpoint1') #path to model checkpoint1
    parser.add_argument('--path_to_checkpoint2', type=str, required=True, help='path to model checkpoint2') #path to model checkpoint2
    parser.add_argument('--path_to_test_csv',type=str, required=True, help='path to test file')
    parser.add_argument('--path_to_test_imgdir',type=str,required=True, help='path to test img directory')

    args = parser.parse_args()
    inference(args.path_to_checkpoint1,args.path_to_checkpoint2,args.path_to_test_csv,args.path_to_test_imgdir)