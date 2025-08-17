import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
import shutil
from tqdm import tqdm
import time
from azure.storage.blob import ContainerClient
import os
import json
from megadetector.detection import run_detector
from megadetector.detection.run_detector_batch import load_and_run_detector_batch, get_image_datetime
from megadetector.utils.ct_utils import truncate_float,truncate_float_array
from .viz_utils import visualize_images
from .img_utils import download_img
from .common_utils import value_counts_both, dataframe_apply_parallel
from .fastai_utils import EffNetClassificationInference



def extract_images(absolute_paths,species,extracted_dir,extracted_folder = 'ExtractedSpecies',max_workers=2):
    # For this function to work, there must be a relative path of each of "absolute path" to "extracted_dir"
    # In another word, "extracted_dir" must be within each of "absolute_path"
    # e.g.:
    #    extracted_dir = 'Z:/Yampi/CT Images/2. Classified and Completed surveys/202305_YSTA_LowlandRefuge_cameras/Naomis Sites'
    #    absolute_paths= [Z:/Yampi/CT Images/2. Classified and Completed surveys/202305_YSTA_LowlandRefuge_cameras/Naomis Sites/202305_YSTA_LowlandRefuge/RCNX0016.JPG]
    #    There will be a folder called "ExtractedSpecies" within the "Naomis Sites" folder, and the images will be copied to the respective species folder within "ExtractedSpecies"
    # This is mostly for G's image extraction task 

    def copy_file(file_path, destination_dir,keep_structure=True):
        file_path = Path(file_path)
        destination_dir = Path(destination_dir)
        if keep_structure:
            common_dir = file_path.relative_to(Path(*destination_dir.parts[:-2]))
            destination_dir = (destination_dir/common_dir).parent
            destination_dir.mkdir(parents=True,exist_ok=True)
        if not (destination_dir/file_path.name).exists():
            shutil.copy(file_path, destination_dir)
    
    assert len(absolute_paths)==len(species)
    
    extracted_dir = Path(extracted_dir)/(extracted_folder.strip())
    extracted_dir.mkdir(exist_ok=True)
    
    for spe in set(species):
        start_time = time.time()  # Start time
        print(f'Copying images for species: {spe}')
        species_dir = extracted_dir / spe.replace('/', '-')
        species_dir.mkdir(exist_ok=True)
        
        copy_to_dest_func = partial(copy_file, destination_dir=species_dir)
    
        start_time = time.time()  # Start time
        paths_filtered = [p for p,s in zip(absolute_paths,species) if s==spe]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(tqdm(executor.map(copy_to_dest_func, paths_filtered), total=len(paths_filtered)))
        
        end_time = time.time()  # End time
        print(f'===> Done copying {len(paths_filtered)} images. Time taken: {end_time - start_time:.2f} seconds.\n')

    print('Done copying all images.')


def mdv5_json_to_df(json_file):
    # Example of Megadetector JSON file, version 1.4
    # https://github.com/agentmorris/MegaDetector/blob/main/megadetector/api/batch_processing/README.md
    # Did not include the classification part yet
    results=[]
    for img in json_file['images']:
        img_file = img['file']
        img_datetime = img.get('datetime', None)
        if 'detections' not in img or img['detections'] is None or len(img['detections'])==0:
            result = [Path(img_file).as_posix(),img_datetime,None,None,None,None,None]
            if 'failure' in img:
                result[-1] = img['failure']
            results.append(result)

        else:
            for i,_d in enumerate(img['detections']):
                results.append([Path(img_file).as_posix(),img_datetime,_d['category'],tuple(_d['bbox']),_d['conf'],i,None])

    df = pd.DataFrame(results,columns=['file','datetime','detection_category','detection_bbox','detection_conf','bbox_rank','failure'])
    return df


def _create_detections(df,class_threshold=0):
    if isinstance(df, pd.Series): df = df.to_frame().T
    
    pred_n = len([c for c in df.columns if 'pred' in c])
    detections=[]
    for f,dt,cat,conf,bbox,fa,*predprobs in df[['file','datetime','detection_category','detection_conf','detection_bbox','failure']+\
                                           [f'pred_{i+1}' for i in range(pred_n)]+\
                                           [f'prob_{i+1}' for i in range(pred_n)]].values:
        if fa is not None and fa is not np.nan:
            return {"file":f, "failure": str(fa)}
        if cat is None or cat is np.nan:
            return {"file":f, "detections": [],"datetime":dt}
        _inner={}
        _inner["category"]=str(int(cat))
        _inner["conf"]=truncate_float(conf,precision=3)
        _inner["bbox"]=truncate_float_array(bbox,precision=4)
        if len(predprobs):
            class_results=[]
            for i in range(pred_n):
                _pred,_prob=predprobs[i],predprobs[i+pred_n]
                if _prob>=class_threshold:
                    class_results.append([str(int(_pred)),truncate_float(_prob,precision=3)])
            if len(class_results):
                _inner["classifications"]=class_results
        detections.append(_inner)
    return {"file":f, "detections":detections,"datetime":dt}


def df_to_mdv5_classification_json(df,class_threshold=0.3,n_workers=None):
    df = df.dropna(subset='file')
    if n_workers==1:
        return df.groupby('file').apply(_create_detections,class_threshold=class_threshold)
    return dataframe_apply_parallel(df.groupby('file'), partial(_create_detections,class_threshold=class_threshold),n_workers=n_workers)

def get_bbox_count_and_conf_rank(df,filter_cat=[]):
    # get bbox count and ranking based on detection confidence
    df = df[~df.detection_conf.isna()].copy().reset_index(drop=True)
    if len(filter_cat)>0 and 'detection_category' in df.columns:
        df = df[df.detection_category.isin(filter_cat)].copy().reset_index(drop=True)
    _tmp = df.groupby('file').detection_conf.count()
    _tmp = _tmp.reset_index()
    _tmp.columns=['file','bbox_count']

    df = pd.merge(df,_tmp)
    df['bbox_conf_rank'] = df.groupby('file').detection_conf.rank(method='first',ascending=False).astype(int)
    return df


def viz_by_detection_threshold(df,lower,upper=1.01,label=None,num_imgs=36,figsize=(12,12),fontsize=6,ascending=False):
    _tmp = df[(df.detection_conf>=lower) & (df.detection_conf<upper)]
    if label:
        _tmp = _tmp[_tmp.label.str.contains(label)]
    print(_tmp.shape[0])
    if 'label' in _tmp.columns:
        print(value_counts_both(_tmp.label).head(10))
    if _tmp.shape[0]>num_imgs: _tmp = _tmp.sample(num_imgs)
    _tmp = _tmp.sort_values('detection_conf',ascending=ascending)
    to_show = _tmp.detection_conf.round(3).astype(str)
    if 'label' in _tmp.columns:
        to_show+=','+_tmp.label.str.split('|',expand=True)[1].str.strip()
    visualize_images(_tmp.abs_path,to_show,_tmp.detection_bbox,figsize=figsize,fontsize=fontsize)


class MegaDetectorInference:
    def __init__(self,
                 md_path=None, # absolute path to megadetector weight
                ):
        self.md_path = str(md_path)
    
    def write_checkpoint(self,results,checkpoint_path):
        # github.com/agentmorris/MegaDetector/blob/d706f9c31ea0a1f2fef5b4a3846737d3d4cf9d64/megadetector/detection/run_detector_batch.py#L793
        # Back up any previous checkpoints, to protect against crashes while we're writing
        # the checkpoint file.
        checkpoint_tmp_path = None
        if os.path.isfile(checkpoint_path):
            checkpoint_tmp_path = checkpoint_path.parent/(checkpoint_path.stem+'_tmp.json')
            shutil.copyfile(checkpoint_path,checkpoint_tmp_path)
            
        # Write the new checkpoint
        with open(checkpoint_path, 'w') as f:
            json.dump({'images': results}, f, indent=1, default=str)
            
        # Remove the backup checkpoint if it exists
        if checkpoint_tmp_path is not None:
            os.remove(checkpoint_tmp_path)
        
    def predict(self,
                img_paths, # list of absolute image paths if not using SAS key, relative paths otherwise, or list of URLs
                run_batch=True, # whether to run the detector in batch mode or one image at a time
                input_container_sas=None, # to get images from a Azure blob container
                md_threshold=0.1, # megadetector bbox confidence threshold, for NMS
                checkpoint_path=None, # absolute path to save check point
                checkpoint_frequency=0, # write results to JSON checkpoint file every n images, set 0 to disable
                convert_to_df=False, # either to convert the results (list of dictionary) to dataframe format
                n_cores=1, # number of cores to use for parallel processing, ignored if we're running on a GPU
               ):
        if checkpoint_path is not None and checkpoint_frequency<=0:
            raise Exception('Invalid checkpoint frequency, please input a positive value')

        if len(img_paths)==0: return []
            
        input_container_client = None
        if input_container_sas is not None:
            input_container_client = ContainerClient.from_container_url(input_container_sas)
            run_batch = False
        
        if not run_batch:
            md_detector = run_detector.load_detector(self.md_path)
            results = []
            img_count=0
            for img_path in tqdm(img_paths):
                img_count+=1
                try:
                    img = download_img(img_path,input_container_client,ignore_exif_rotation=True,load_img=False)
                except Exception as e:
                    print(f'File {img_path}, Download image exception: {e}')
                    result= {'file': img_path,
                            'failure': "Failure image access",
                            }
                else:
                    result = md_detector.generate_detections_one_image(img, 
                                                                       img_path, 
                                                                       detection_threshold=md_threshold)
                    
                    result['datetime'] = get_image_datetime(img)
                            
                results.append(result)
                
                if checkpoint_frequency!=0 and ((img_count%checkpoint_frequency)==0 or img_count==len(img_paths)):
                    print(f'Write checkpoint for {img_count} images')
                    self.write_checkpoint(results,Path(checkpoint_path))
        else:
            results = load_and_run_detector_batch(self.md_path, 
                                                  img_paths,
                                                  confidence_threshold=md_threshold,
                                                  quiet=True,
                                                  include_image_timestamp=True,
                                                  checkpoint_path=checkpoint_path,
                                                  checkpoint_frequency=checkpoint_frequency,
                                                  n_cores=n_cores
                                                  )

        if convert_to_df:
            results = mdv5_json_to_df({'images':results})
        
        return results
    

class DetectAndClassify:
    def __init__(self, 
                 md_path, # absolute path to megadetector weight
                 finetuned_model=None, # absolute path to efficient model that has been finetuned
                 label_info=None, # list of output labels, or number of labels
                 item_tfms=None, # list of item transformations
                 efficient_model='efficientnet-b3', # name of pretrained efficient model
                 aug_tfms=None, # augmentation transformations, needed if TTA is used
                 parent_info=None, # list of parent labels, or number of parent labels, needed for hierarchical classification (hitax)
                 child2parent=None, # dictionary of child to parent mapping (hitax)
                 hitax_output=None, # None for merged output, 'parent' for parent only, 'child' for child only (hitax)
                 parent2child=None, # dictionary of parent to child mapping, needed for rollup classification
                 hitax_threshold=0.75 # threshold for for hitax or rollup classification, default is 0.75
                ):
        self.md_inference = MegaDetectorInference(md_path)
        self.class_inference = None
        self.hitax_output = hitax_output
        if hitax_output is not None:
            if hitax_output not in ['parent','child']:
                raise Exception('hitax_output must be either None, "parent" or "child"')
            hitax_threshold = None
        if finetuned_model is not None and label_info is not None:
            self.label_info = label_info
            self.class_inference = EffNetClassificationInference(label_info=label_info,
                                                                 efficient_model=efficient_model,
                                                                 finetuned_model=finetuned_model,
                                                                 item_tfms=item_tfms,
                                                                 aug_tfms=aug_tfms,
                                                                 parent_info=parent_info,
                                                                 child2parent=child2parent,
                                                                 parent2child=parent2child,
                                                                 hitax_threshold=hitax_threshold
                                                                 )

    def hitax_cleanup(self,df):
        if self.class_inference.is_rollup or (self.class_inference.is_hitax and self.class_inference.hitax_threshold is not None):
            # file  detection_bbox  pred_1  prob_1  level
            # shift index of children (level 2) by len(self.class_inference.parent_info)
            df.loc[(~df['level'].isna()) & (df['level']==2),'pred_1'] = df.loc[(~df['level'].isna()) & (df['level']==2),'pred_1'] + len(self.class_inference.parent_info)
            df = df.drop(columns=['level'])
            return df
            
        if self.class_inference.is_hitax and self.hitax_output is not None:
            if self.hitax_output == 'child':
                parent_cols = [c for c in df.columns if 'parent_' in c]
                df = df.drop(columns=parent_cols)
                # rename child_pred_1 to pred_1, child_prob_1 to prob_1, etc.
                n_preds = [c for c in df.columns if 'child_pred_' in c]
                df = df.rename(columns={f'child_pred_{i+1}': f'pred_{i+1}' for i in range(len(n_preds))})
                df = df.rename(columns={f'child_prob_{i+1}': f'prob_{i+1}' for i in range(len(n_preds))})
            elif self.hitax_output == 'parent':
                child_cols = [c for c in df.columns if 'child_' in c]
                df = df.drop(columns=child_cols)
                # rename parent_pred_1 to pred_1, parent_prob_1 to prob_1, etc.
                n_preds = [c for c in df.columns if 'parent_pred_' in c]
                df = df.rename(columns={f'parent_pred_{i+1}': f'pred_{i+1}' for i in range(len(n_preds))})
                df = df.rename(columns={f'parent_prob_{i+1}': f'prob_{i+1}' for i in range(len(n_preds))})
        return df
            
    def predict(self,
                img_paths, # list of absolute image paths if not using SAS key, relative paths otherwise, or list of URLs
                input_container_sas=None, # to get images from a Azure blob container
                md_threshold=0.1, # megadetector bbox confidence threshold, for NMS
                checkpoint_path=None, # absolute path to save check point
                checkpoint_frequency=0, # write results to JSON checkpoint file every n images, set 0 to disable
                classify_batch_size=16, # batch size for classification model
                tta_n=0, # whether to perform test time augmentation, and how many
                pred_topn=1, # to return top n predictions
                prob_round=3, # number of decimal points to round the probability
                class_threshold=0.3, # the probability threshold to keep in the JSON file output,
                n_workers=None, # number of workers to use for parallel processing
                pin_memory=False, # If True, the data loader (classification only) will copy Tensors into CUDA pinned memory before returning them
                convert_to_json=True # either to convert the predictions (dataframe) to JSON format
               ):
        md_result = self.md_inference.predict(img_paths,
                                              run_batch=True,
                                              input_container_sas=input_container_sas,
                                              md_threshold=md_threshold,
                                              checkpoint_path=checkpoint_path,
                                              checkpoint_frequency=checkpoint_frequency,
                                              convert_to_df=self.class_inference is not None,
                                              n_cores=n_workers
                                              )
        # md_result is a dataframe with columns:
        # file	datetime    detection_category	detection_bbox	detection_conf	bbox_rank	failure
        
        if self.class_inference is None:
            return md_result
        
        md_result_valid = md_result[(~md_result['detection_bbox'].isna()) & (~md_result['detection_category'].isna())]

        # filter animal images (cat_id is 1) only
        md_result_valid = md_result_valid[md_result_valid['detection_category'].astype(int).isin([1])].copy()
        c_result = self.class_inference.predict(md_result_valid,
                                                input_container_sas=input_container_sas,
                                                batch_size=classify_batch_size,
                                                tta_n=tta_n,
                                                pred_topn=pred_topn,
                                                name_output=False,
                                                prob_round=prob_round,
                                                n_workers=n_workers,
                                                pin_memory=pin_memory
                                                )
        # default
        # file	detection_bbox	pred_1	pred_2	pred_3	prob_1	prob_2	prob_3

        # for hitax (with no parent-child merging)
        # file  detection_bbox  parent_pred_1  parent_prob_1  parent_pred_2  parent_prob_2
        #                       child_pred_1   child_prob_1  child_pred_2   child_prob_2

        # for rollup, or hitax with parent-child merging
        # file  detection_bbox  pred_1  prob_1  level

        c_result = self.hitax_cleanup(c_result)
        c_result = c_result.set_index(md_result_valid.index.values)
        c_result = pd.concat([md_result,c_result.iloc[:,2:]],axis=1) # concat the preds and probs to md_result
        # file	datetime    detection_category	detection_bbox	detection_conf	bbox_rank   failure pred_1	prob_1	pred_2	prob_2	pred_3	prob_3
        if convert_to_json:
            return df_to_mdv5_classification_json(c_result,class_threshold=class_threshold,n_workers=n_workers)

        return c_result