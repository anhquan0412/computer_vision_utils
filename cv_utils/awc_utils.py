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
    # Example of mdv5 json file:
    # {
    #     "info": {
    #         "classifier": "classifier_v5b",
    #         "classification_completion_time": "2024-06-15T09:14:54.294909Z",
    #         "format_version": "1.1"
    #     },
    #     "detection_categories": {
    #         "1": "animal",
    #         "2": "person",
    #         "3": "vehicle"
    #     },
    #     "images": [
    #         {
    #             "file": "CPW_Files_/june_24/Charnley River - Artesian Range/Petrogale burbidgei - Monjon/Pre-Grant Extraction_Cha_P burbidgei/2019-12-17 19-13-17 M 1_5.JPG",
    #             "max_detection_conf": 0.919,
    #             "detections": [
    #                 {
    #                     "category": "1",
    #                     "conf": 0.919,
    #                     "bbox": [
    #                         0,
    #                         0.3984,
    #                         0.1455,
    #                         0.1848
    #                     ]
    #                 }
    #             ]
    #         },
    #     ]
    # }
    results=[]
    for img in json_file['images']:
        img_file = img['file']
        if len(img['detections']) == 0:
            results.append([img_file,None,None,None])
        else:
            for _d in img['detections']:
                results.append([img_file,_d['category'],_d['conf'],_d['bbox']])

    df =  pd.DataFrame(results,columns=['file','detection_category','detection_conf','detection_bbox'])
    df['file'] = df['file'].apply(lambda x: Path(x).as_posix())

    # drop duplicates
    # it's okay to convert to string, since the float values are already rounded by mdv5
    df['detection_bbox'] = df['detection_bbox'].astype(str)
    df = df.drop_duplicates().reset_index(drop=True)
    df['detection_bbox'] = df['detection_bbox'].apply(lambda x: ast.literal_eval(x))

    # get bbox count and bbox rank
    df = df[~df.detection_conf.isna()].reset_index(drop=True)
    _tmp = df.groupby('file').detection_category.count()
    _tmp = _tmp.reset_index()
    _tmp.columns=['file','bbox_count']

    df = pd.merge(df,_tmp)
    df['bbox_rank'] = df.groupby('file').detection_conf.rank(method='dense',ascending=False)
    return df



def viz_by_detection_threshold(df,lower,upper=1.01,label=None,num_imgs=36,figsize=(12,12),fontsize=6):
    _tmp = df[(df.detection_conf>=lower) & (df.detection_conf<upper)]
    if label:
        _tmp = _tmp[_tmp.label.str.contains(label)]
    print(_tmp.shape[0])
    if 'label' in _tmp.columns:
        print(value_counts_both(_tmp.label).head(10))
    if _tmp.shape[0]>num_imgs: _tmp = _tmp.sample(num_imgs)
    to_show = _tmp.detection_conf.round(2).astype(str)
    if 'label' in _tmp.columns:
        to_show+=','+_tmp.label.str.split('|',expand=True)[1].str.strip()
    visualize_images(_tmp.abs_path,to_show,_tmp.detection_bbox,figsize=figsize,fontsize=fontsize)