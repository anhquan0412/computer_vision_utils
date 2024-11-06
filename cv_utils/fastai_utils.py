
from sklearn.metrics import precision_recall_fscore_support
from fastai.vision.all import *
from fastai.callback.wandb import *
from collections.abc import Iterable
from pathlib import Path
import ast
import torch
import pandas as pd
import numpy as np
import wandb
from azure.storage.blob import ContainerClient
import os
import warnings; warnings.simplefilter('ignore')
from .img_utils import crop_image
from efficientnet_pytorch import EfficientNet
from multiprocessing import cpu_count

def bold_print(txt):
    print(f"{'='*20} {txt.upper()} {'='*20}")  

def seed_notorch(seed=42):
    random.seed(seed)
    np.random.seed(seed)

def seed_everything(seed=42):
    seed_notorch(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def precision_recall_f1_func(label_name,label_idx,mtype="f1"):
    def _metric(y_true,y_pred):
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
        if mtype=='precision':
            return round(precision[label_idx],4)
        elif mtype=='recall':
            return round(recall[label_idx],4)
        else:
            return round(f1[label_idx],4)
    _metric.__name__ = f'{mtype}_{label_name}'
    return _metric

def get_precision_recall_f1_metrics(label_names,mtype="f1"):
    metrics = []
    for i,v in enumerate(label_names):
        metrics.append(AccumMetric(precision_recall_f1_func(v,i,mtype),dim_argmax=-1,to_np=True,invert_arg=True))
    return metrics

def fastai_predict_val(learner,label_names,df_val=None,save_path=None):
    val_probs,val_true,val_pred = learner.get_preds(with_decoded=True,with_preds=True,with_input=False)
    val_pred_str = list(map(lambda x: label_names[x],val_pred))
    val_true_str = list(map(lambda x: label_names[x],val_true))
    df_pred = pd.DataFrame()
    df_pred['y_pred'] = val_pred_str
    df_pred['y_true'] = val_true_str
    df_pred[label_names] = torch.round(val_probs,decimals=5)
    print(df_pred.head())
    print(df_val.head())
    if df_val is not None:
        assert len(df_val)==len(df_pred)
        df_pred = pd.concat([df_val.reset_index(drop=True),df_pred],axis=1)
    
    if save_path:
        df_pred.to_csv(save_path,index=False)
        print(f'Predictions saved to {save_path}')
        return 
    return df_pred

def _download_img_tiny(input_container_client,inp):
    if input_container_client is not None:
        downloader = input_container_client.download_blob(inp)
        inp = io.BytesIO()
        blob_props = downloader.download_to_stream(inp)
    return inp


def PILImageFactory(container_client=None):
    class PILMDImage(PILBase):
        # Blob client variable
        input_container_client=container_client
        
        @classmethod
        def create(cls, inps, **kwargs):
            if not isinstance(inps,str):
                inps = list(inps)
                inps[0] = _download_img_tiny(PILMDImage.input_container_client,inps[0])
                img = PILImage.create(inps[0])
                norm_bbox = inps[1]
                img = crop_image(img,norm_bbox,square_crop=True)
                return PILImage.create(img)

            inps = _download_img_tiny(PILMDImage.input_container_client,inps)
            return PILImage.create(inps)
 
    return PILMDImage

def fastai_cv_train_efficientnet(config,df,aug_tfms=None,label_names=None,save_valid_pred=False):
    # The first column of df should be the file path, or a tuple of file path and bbox coord
    # The second column is the label (string)
    # There is a column called 'is_val', for train val split (boolean)

    class ColMDReader(DisplayedTransform):
        "Read `cols` in `row` with potential `pref` and `suff`"
        def __init__(self, cols, pref='', suff='', label_delim=None):
            store_attr()
            self.pref = str(pref) + os.path.sep if isinstance(pref, Path) else pref
            self.cols = L(cols)

        def _do_one(self, r, c):
            o = r[c] if isinstance(c, int) or not c in getattr(r, '_fields', []) else getattr(r, c)
            # o is a tuple of (relative_path, bbox_coords)
            if len(self.pref)==0 and len(self.suff)==0 and self.label_delim is None: return o

            return f'{self.pref}{o[0]}{self.suff}' # get the first element of the tuple, which is the path. 
            

        def __call__(self, o, **kwargs):
            if len(self.cols) == 1: return self._do_one(o, self.cols[0])
            return L(self._do_one(o, c) for c in self.cols)
        
    def ImageDataLoaders_from_df(df, path='.', valid_pct=0.2, seed=None, fn_col=0, folder=None, suff='', label_col=1, label_delim=None,
                y_block=None, valid_col=None, item_tfms=None, batch_tfms=None, img_cls=PILImage, **kwargs):
        "Create from `df` in `path` using `fn_col` and `label_col`"
        pref = f'{Path(path) if folder is None else Path(path)/folder}{os.path.sep}'
        
        if y_block is None:
            is_multi = (is_listy(label_col) and len(label_col) > 1) or label_delim is not None
            y_block = MultiCategoryBlock if is_multi else CategoryBlock
        splitter = RandomSplitter(valid_pct, seed=seed) if valid_col is None else ColSplitter(valid_col)

        # check whether bbox coord is the input
        if isinstance(df.iloc[0,0],(list,tuple)) and len(df.iloc[0,0])==2 and len(df.iloc[0,0][1])==4:
            PILImageClass = PILImageFactory()
            col_reader = ColMDReader(fn_col, pref=pref, suff=suff)
        else:
            PILImageClass = PILImage
            col_reader = ColReader(fn_col, pref=pref, suff=suff)

        dblock = DataBlock(blocks=(ImageBlock(PILImageClass), y_block),
                           get_x=col_reader,
                           get_y=ColReader(label_col, label_delim=label_delim),
                           splitter=splitter,
                           item_tfms=item_tfms,
                           batch_tfms=batch_tfms)
        return ImageDataLoaders.from_dblock(dblock, df, path=path, **kwargs)
    
    use_wandb = 'WANDB_PROJECT' in config

    if 'SEED' in config:
        seed = config['SEED']
        seed_everything(seed)
    else:
        seed=None
    

    dls = ImageDataLoaders_from_df(df, 
                                   path=config['IMAGE_DIRECTORY'],
                                   seed=seed,
                                   fn_col=0,
                                   label_col=1,
                                   valid_col='is_val',
                                   item_tfms= Resize(config['ITEM_RESIZE']) if 'ITEM_RESIZE' in config else Resize(750),
                                   bs=config['BATCH_SIZE'],
                                   shuffle=True,
                                   batch_tfms=aug_tfms
                                  )
        
    if not label_names:
        label_names = dls.vocab.items.items

    model = EfficientNet.from_pretrained(config['EFFICIENT_MODEL'],num_classes=len(label_names))
    
    monitor_metric = config['MONITOR_METRIC'] if 'MONITOR_METRIC' in config else 'f1_score' #'valid_loss'
    save_every_epoch = config['SAVE_EVERY_EPOCH'] if 'SAVE_EVERY_EPOCH' in config else False
    save_directory = Path(config['SAVE_DIRECTORY']) if 'SAVE_DIRECTORY' in config else Path('.')/'model'
    save_directory.mkdir(exist_ok=True,parents=True)
    save_name = config['SAVE_NAME'] if 'SAVE_NAME' in config else 'model'
    log_metrics = config['LOG_LABEL_METRICS'] if 'LOG_LABEL_METRICS' in config else []
    assert set(log_metrics) - set(['precision','recall','f1'])==set()
    cbs=[
            SaveModelCallback(monitor=monitor_metric,
                              every_epoch=save_every_epoch,
                              fname=(save_directory/save_name),
                              comp=np.greater if 'loss' not in monitor_metric else np.less,
                              ),
            CSVLogger(fname=(save_directory/f"{save_name}_training_log.csv"), append=True)
        ]
    
    mixup_alpha = config['MIXUP_ALPHA'] if ('MIXUP_ALPHA' in config and config['MIXUP_ALPHA']>0) else None
    if mixup_alpha is not None:
        cbs.append(MixUp(mixup_alpha))
    
    if use_wandb:
        os.environ["WANDB_SILENT"] = "true"
        wandb.init(project=config['WANDB_PROJECT'],name=save_name,config=config);
        cbs.append(WandbCallback(log_dataset=False, log_model=False, n_preds=49))
    
    metric_lists = []
    for lm in log_metrics:
        metric_lists += get_precision_recall_f1_metrics(label_names,mtype=lm)

    learn = Learner(dls, model, 
                    metrics=[F1Score(average='macro')]+metric_lists,
                    cbs=cbs,
                   ).to_fp16()
    
    bold_print('training model')
    epoch = config['EPOCH']
    freeze_epoch = config['FREEZE_EPOCH'] if 'FREEZE_EPOCH' in config else 1
    if len(metric_lists)<8:
        learn.fine_tune(epoch,freeze_epochs=freeze_epoch,base_lr=config['LR'])
    else:
        with learn.no_bar(), learn.no_logging():
            learn.fine_tune(epoch,freeze_epochs=freeze_epoch,base_lr=config['LR'])

    _ax = learn.recorder.plot_loss(show_epochs=True)
    plt.savefig((save_directory/f'{save_name}_learning_curve.png'), bbox_inches='tight')

    if save_valid_pred:
        bold_print('predicting validation set')
        df_val = df[df['is_val']==True].copy()
        save_path = (save_directory/f'{save_name}_val_pred.csv')
        _ = fastai_predict_val(learn,label_names,df_val=df_val,save_path=save_path)
    
    if use_wandb:
        wandb.finish();
    return learn


# example usage
# aug_tfms = aug_transforms(size=224,
#                          max_rotate=45, # 10
#                          max_zoom=1.3, # 1.1
#                          min_scale=1., # 1.
#                          do_flip=True,
#                          max_lighting=0.5, # 0.2
#                          p_lighting=0.75, # 0.75
#                         )

# config = {
#     'SEED':42,
#     'IMAGE_DIRECTORY':'path/to/image/directory',
#     'ITEM_RESIZE':750,
#     'BATCH_SIZE':64,
#     'EFFICIENT_MODEL':'efficientnet-b3',
#     'EPOCH':4,
#     'FREEZE_EPOCH':1,
#     'LR':0.0021,
#     'SAVE_NAME':'name_for_this_run',
#     'SAVE_DIRECTORY':'path/to/save/directory',
#     'SAVE_EVERY_EPOCH':True,
#     'WANDB_PROJECT':'wandb_project_name',
#     'LOG_LABEL_METRICS':['precision','f1']
# }

# # bold_print('reading config')
# # with open('config.json', 'r') as json_file:
# #     config = json.load(json_file)


# bold_print('load mastersheet')
# df = pd.read_pickle('path/to/mastersheet.pkl')
# df['is_val'] = df.split.map({'train':False,'val':True,'test':True})
# df = df[['cropped_file','label','is_val','detection_conf','split']].copy()
# df = df.sample(9000,random_state=42,ignore_index=True)
# _ = fastai_cv_train_efficientnet(config,df,aug_tfms=aug_tfms,save_valid_pred=True)


def _verify_images(inps,input_container_sas=None):    
    try:
        input_container_client=None
        if input_container_sas is not None:
            input_container_client = ContainerClient.from_container_url(input_container_sas)
        if not isinstance(inps,str):
            inps = list(inps)
            inps[0] = _download_img_tiny(input_container_client,inps[0])
            img = PILImage.create(inps[0])
            norm_bbox = inps[1]
            img = crop_image(img,norm_bbox,square_crop=True)
            img = PILImage.create(img)
        else:
            inps = _download_img_tiny(input_container_client,inps)
            img = PILImage.create(inps)
    
    except Exception as e:
        return False
    else:
        return True


class EffNetClassificationInference:
    def __init__(self,
                 label_info, # list of output labels, or the number of labels
                 finetuned_model, # absolute path to efficient model that has been finetuned
                 efficient_model='efficientnet-b3', # name of pretrained efficient model
                 item_tfms=Resize(750), # list of item transformations
                 aug_tfms=None, # augmentation transformations, needed if TTA is used
                ):
        self.label_info = label_info
        finetuned_model = Path(finetuned_model)
        finetuned_model = finetuned_model.parent/finetuned_model.stem
        self.finetuned_model = finetuned_model
        self.item_tfms = item_tfms
        self.aug_tfms = aug_tfms
        self.model = EfficientNet.from_pretrained(efficient_model,
                                                  weights_path=str(finetuned_model)+'.pth',
                                                  num_classes=label_info if isinstance(label_info,int) else len(label_info))

    def validate_df(self,df):
        if 'file' in df.columns.tolist():
            if len(df)==0: return []
            if 'detection_bbox' in df.columns.tolist():
                if not isinstance(df['detection_bbox'].values[0],(list,tuple)):
                    df['detection_bbox'] = df['detection_bbox'].apply(lambda x: None if (x is None or x is np.NaN) else list(ast.literal_eval(x)))
                return df[['file','detection_bbox']].values
            else:
                return df['file'].values
        else:
            raise Exception("For dataframe input, the dataframe must have a column named 'file' \
            containing the absolute path of images, and optionally a column named 'detection_bbox' for their bounding boxes")

    def create_output_df(self,inputs,preds,pred_idxs,valid_idxs,name_output):
        if isinstance(inputs[0],str):
            df= pd.DataFrame(inputs,columns=['file'])
        else:
            df= pd.DataFrame(inputs,columns=['file','detection_bbox'])

        top_n = preds.shape[1]
        df_pred = pd.DataFrame(pred_idxs,columns=[f'pred_{i+1}' for i in range(top_n)])
        if name_output and isinstance(self.label_info,(list,tuple,np.ndarray)):
            df_pred = df_pred.map(lambda x: self.label_info[x])            
        df_prob = pd.DataFrame(preds,columns=[f'prob_{i+1}' for i in range(top_n)])
        
        df_pred.index=valid_idxs
        df_prob.index=valid_idxs
        
        df = pd.concat([df,df_pred,df_prob],axis=1)
        return df
        
    def predict(self,
                inputs, # can be list of img paths, list of tuples, or dataframe
                input_container_sas=None, # SAS for accessing images in Blob Container
                batch_size=16,
                tta_n=0, # whether to perform test time augmentation, and how many
                pred_topn=1, # to return top n predictions
                name_output=True, # whether to return the label names instead of label indices
                prob_round=3, # number of decimal points to round the probability
                do_image_check=False, # to check whether input images can be opened. Note: without this, invalid images will interrupt the prediction process
                n_workers=1, # number of workers for parallel processing (image verification and dataloaders). None for all, up to 16
                pin_memory=False # If True, the data loader will copy Tensors into CUDA pinned memory before returning them
               ):
        if not isinstance(inputs, Iterable) or isinstance(inputs,str):
            inputs = np.array([inputs])
        if isinstance(inputs, pd.DataFrame):
            inputs = inputs.copy()
            inputs = self.validate_df(inputs)
        if len(inputs)==0: return pd.DataFrame()
        

        if isinstance(inputs[0],str) or (len(inputs[0])==2 and isinstance(inputs[0][0],str) and len(inputs[0][1])==4):
            inputs = np.array(inputs)
            
            input_container_client=None
            if input_container_sas is not None:
                input_container_client = ContainerClient.from_container_url(input_container_sas)

            PILImageClass = PILImageFactory(container_client=input_container_client)
            # check for imgs that can be opened only
            valid_idxs = list(range(len(inputs)))
            if do_image_check:
                print('Perform image validations...')
                if input_container_sas is not None:
                    print('Warning: verifying images on Blob Container can be time-consuming')
                if n_workers==1:
                    valid_idxs = [i for i,o in enumerate(inputs) if _verify_images(o,input_container_sas)]
                else:
                    if n_workers is None: n_workers=min(16,cpu_count())
                    valid_idxs = [i for i,o in enumerate(parallel(partial(_verify_images,input_container_sas=input_container_sas), inputs, n_workers=n_workers)) if o]
                if len(valid_idxs)<len(inputs):
                    print(f'There is/are {len(inputs)-len(valid_idxs)} invalid input(s), out of {len(inputs)} inputs')
                else:
                    print(f'All {len(inputs)} inputs are valid')
        
            blocks = ImageBlock(PILImageClass)
        else:
            raise Exception('Unknown input type')

        datablock = DataBlock(blocks = blocks,
                              item_tfms = self.item_tfms,
                              batch_tfms = self.aug_tfms,
                              splitter = lambda x: (L(0),L(list(torch.arange(len(valid_idxs)).numpy())))
                             )
        dls = DataLoaders.from_dblock(datablock,
                                      inputs[valid_idxs],
                                      bs=batch_size,
                                      num_workers=n_workers,
                                      pin_memory=pin_memory,
                                      shuffle=False)
        
        learner = Learner(dls,self.model,loss_func = CrossEntropyLossFlat())
        _ = learner.load(self.finetuned_model)

        if tta_n>0:
            preds = learner.tta(dl = dls.valid,n=tta_n)[0]
        else:
            preds = learner.get_preds(dl = dls.valid)[0]

        preds = torch.round(preds,decimals = prob_round)
        preds,pred_idxs = preds.sort(dim=1,descending=True)
        preds = preds[:,:pred_topn]
        pred_idxs = pred_idxs[:,:pred_topn]
        
        return self.create_output_df(inputs,preds,pred_idxs,valid_idxs,name_output)