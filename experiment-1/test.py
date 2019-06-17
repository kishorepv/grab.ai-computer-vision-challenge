from fastai.vision import *
from fastai.metrics import error_rate
from fastai import *
import cv2 as cv
import numpy as np
import pandas as pd
import scipy.io as sio
import time

import argparse


#import os
#os.environ['CUDA_VISIBLE_DEVICES']='1'
#torch.cuda.set_device(torch.device("cuda:1"))

def uprint(strng):
    print(f"\r{strng}".ljust(len(strng)+20), end='')
    
def vertical_splitter(src, dest):
    img_names=[x for x in os.listdir(src) if os.path.isfile(src/x)]
    if os.path.exists(dest):
        print(f"WARNING: {dest} directory already exists. Stopping pre-processing")
        return dest
    os.mkdir(dest)
    N=len(img_names)
    for c,im_name in enumerate(img_names,1):
        uprint(f"{c}/{N}")
        im_path=src/im_name
        img=plt.imread(im_path)
        #img=np.asarray(Image.open(im_path))
        r,c,*_=img.shape
        img1,img2=img[:,:c//2],img[:,c//2:]
        plt.imsave(dest/f"{im_path.stem}x1{im_path.suffix}", img1)
        plt.imsave(dest/f"{im_path.stem}x2{im_path.suffix}", img2)
    return dest

def create_ordered_files(suffixes,dir_name, fnames, labels=None):
    """
        Match the file names and the labels
        This is required because, for some datasets, the augmentations are pre-processed and stored in the train/test folders
        Returns df with `name` (file names) and `label` (category labels)
    """
    len_suf=len(suffixes)
    dir_name=pathlib.Path(dir_name)
    labels=labels if labels else []
    N=len(fnames)
    M=len_suf*N
    DUMMY=0
    df=pd.DataFrame(index=range(M), columns=["name"], dtype="str")
    new_fnames=[]
    ys=[]
    for i in range(N):
        fname=pathlib.Path(fnames[i])
        for j,suffix in enumerate(suffixes):
            new_fname=(fname.stem)+suffix+(fname.suffix)
            new_fnames.append(new_fname)
            #df.loc[idx,"label"]=DUMMY # slow, DONT do it
            if labels:
                ys.append(labels[i])          
    df["name"]=new_fnames
    df["label"]=ys if labels else DUMMY
    return df

def get_test_acc(model_path, model_name, test_path, ylabel_path):
    y_df=pd.read_csv(ylabel_path)
    y=torch.from_numpy(y_df["0"].values)
    learn=load_learner(model_path,model_name, test=ImageList.from_folder(test_path))
    preds,yhat = learn.get_preds(ds_type=DatasetType.Test)
    acc=accuracy(preds,y-1)
    return acc

def ensemble_acc2(preds,y, idxs):
    df=pd.DataFrame({"preds":preds.argmax(1), "fname":[idx.split('x')[0] for idx in idxs]}, index=idxs)
    yp=df.groupby("fname").agg(lambda x: pd.Series.mode(x)[0])["preds"].values
    yt=np.array(y)
    return (yt==yp).mean()

def ensemble_preds(preds,idxs):
    df=pd.DataFrame(np.asarray(preds), index=idxs)
    df["fname"]=[idx.split('x')[0] for idx in idxs]
    yp=df.groupby("fname").mean().values
    return torch.from_numpy(yp)

def predict_on_test(model_path, model_name,test_path, suffixes, tta=False):
    int_suffix="_2x"
    fname_col="name"
    
    fnames=[f for f in os.listdir(test_path) if os.path.isfile(os.path.join(test_path,f))]
    print(f"No of test files: {len(fnames)}")
    
    src=pathlib.Path(test_path)
    dest=pathlib.Path(test_path+int_suffix)
    #!rm -r {dest} # if it already exists
    
    # UNCOMMENT THIS LINE WHILE USING THE VERY FIRST TIME
    vertical_splitter(src, dest)
    
    test_df=create_ordered_files(suffixes,dest,fnames,labels=None)
    learner=load_learner(model_path,model_name, test=ImageList.from_df(test_df, path=dest))
    pred_func=learner.TTA if tta else learner.get_preds
    preds,_=pred_func(ds_type=DatasetType.Test)
    preds2=ensemble_preds(preds,test_df[fname_col])
    return preds2,pd.DataFrame({"file_names":fnames})


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Script to evaluate model on test set')
    parser.add_argument('--test_path', action="store", type=str, help="path to folder containing test images")
    parser.add_argument('--model_path', action="store", type=str, help="directory containing saved fastai model")
    parser.add_argument('--model_name', action="store", type=str, help="name of fastai model (.pkl file)")
    parser.add_argument('--no_tta', action="store_false", default=True, help="Switch off Test Time Augmentation (TTA)")
    parser.add_argument('--ylabel_path', action="store", default='', type=str, help="csv file containing labels for the test data")
    parser.add_argument('--result_dir', action="store", help="directory to store the results - confidence scores and the corresponding image names")
    
    args=parser.parse_args()
    
    #model_path='data/cars_train_2x'
    #model_name="MODEL-final-stanford-cars-1x2-tfms-res152.pkl" # inside the `model_path` directory above
    #test_path="data/cars_test"
    #tta=True
    

    model_path=args.model_path
    model_name=args.model_name
    test_path=args.test_path
    tta=args.no_tta
    
    
    suffixes=["x1","x2"]
    preds2, test_names_df=predict_on_test(model_path, model_name,test_path, suffixes, tta)

    out_dir=pathlib.Path(args.result_dir)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    fid=int(time.time())
    np.save(out_dir/f"predictions_{fid}.npy", preds2)
    idxs=preds2.argmax(1)
    confidence=preds2[range(preds2.shape[0]),idxs]
    labels_with_confidence=np.column_stack([idxs+1,confidence]) # +1 because the labels start with 1, instead of 0
    np.save(out_dir/f"predicted_labels_with_confidence{fid}.npy", labels_with_confidence)
    test_names_df.to_csv(out_dir/f"test_file_names_{fid}.csv", index=True)
    print(f"Saved confidence scores and corresponding image names to {args.result_dir}")

    # NOTE: test labels are assumed to be from 1 to 196, as in the original Stanford Cars dataset, NOT [0,195]
    
    ylabel_path=args.ylabel_path
    if ylabel_path != '':
        #ylabel_path="data/test_labels.csv" 
        ylabel_path=args.ylabel_path   # test labels are assumed to be from 1 to 196, as in the original Stanford Cars dataset
        y_df=pd.read_csv(ylabel_path)
        y=torch.from_numpy(y_df["label"].values)-1 # -1 because fastai maps labels to [0, n_classes-1]
        print(f"No of labels: {y.shape[0]}")

        acc=accuracy(preds2, y)
        print(f"Test accuracy: {acc.item():.3f}")
