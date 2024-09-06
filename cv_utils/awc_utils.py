import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
import shutil
from tqdm import tqdm
import time

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
    results=[]
    for img in json_file['images']:
        img_file = img['file']
        if len(img['detections']) == 0:
            results.append([img_file,None,None,None])
        else:
            for _d in img['detections']:
                results.append([img_file,_d['category'],_d['conf'],_d['bbox']])

    return pd.DataFrame(results,columns=['file','detection_category','detection_conf','detection_bbox'])

