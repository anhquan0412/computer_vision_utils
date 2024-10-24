import ast
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
import shutil
from tqdm import tqdm
import time

from .viz_utils import visualize_images
from .common_utils import value_counts_both



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

    results=[]
    for img in json_file['images']:
        img_file = img['file']
        if 'detections' not in img or img['detections'] is None or len(img['detections'])==0:
            result = [Path(img_file).as_posix(),None,None,None,None,None]
            if 'failure' in img:
                result[-1] = img['failure']
            results.append(result)

        else:
            for i,_d in enumerate(img['detections']):
                results.append([Path(img_file).as_posix(),_d['category'],tuple(_d['bbox']),_d['conf'],i,None])

    df = pd.DataFrame(results,columns=['file','detection_category','detection_bbox','detection_conf','bbox_rank','failure'])
    return df

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
    to_show = _tmp.detection_conf.round(2).astype(str)
    if 'label' in _tmp.columns:
        to_show+=','+_tmp.label.str.split('|',expand=True)[1].str.strip()
    visualize_images(_tmp.abs_path,to_show,_tmp.detection_bbox,figsize=figsize,fontsize=fontsize)