
from sklearn.metrics import precision_recall_fscore_support
from fastai.vision.all import *
from fastai.callback.wandb import *
from collections.abc import Iterable
import ast
import wandb
from azure.storage.blob import ContainerClient
import warnings; warnings.simplefilter('ignore')
from .img_utils import crop_image, download_img
from .common_utils import check_and_fix_http_path
from .hierarchical_model import load_hier_model, HierarchicalClassificationLoss
from .hierarchical_rollup import precompute_rollup_maps_dynamic, rollup_predictions_dynamic
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import efficientnet, efficientnet_params
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
    if df_val is not None:
        assert len(df_val)==len(df_pred)
        df_pred = pd.concat([df_val.reset_index(drop=True),df_pred],axis=1)
    
    if save_path:
        df_pred.to_csv(save_path,index=False)
        print(f'Predictions saved to {save_path}')
        return 
    return df_pred


class PILMDImage(PILBase):
    # Blob client variable
    input_container_client = None

    @classmethod
    def create(cls, inps, **kwargs):
        if isinstance(inps, Iterable) and not isinstance(inps, str):
            # containing bbox. Either (file_path, bbox) or [[file_path, bbox]]
            if len(inps) == 1:
                inps = inps[0]
            inps = list(inps)
            inps[0] = download_img(check_and_fix_http_path(inps[0]),
                                   PILMDImage.input_container_client)
            img = PILImage.create(inps[0])
            norm_bbox = inps[1]
            img = crop_image(img, norm_bbox, square_crop=True)
            return PILImage.create(img)

        inps = download_img(check_and_fix_http_path(inps),
                            PILMDImage.input_container_client)
        return PILImage.create(inps)

# Update PILImageFactory to set the container client
def PILImageFactory(container_client=None):
    PILMDImage.input_container_client = container_client
    return PILMDImage

def fastai_cv_train_efficientnet(config,df,aug_tfms=None,label_names=None,save_valid_pred=False,n_workers=None):
    # The first column of df should be the file path, or a tuple of file path and bbox coord
    # The second column is the label (string)
    # There is a column called 'is_val', for train val split (boolean)

    class ColMDReader(DisplayedTransform):
        "Read `cols` in `row` with potential `pref` and `suff`"
        # https://github.com/fastai/fastai/blob/master/fastai/data/transforms.py#L206
        def __init__(self, cols, pref='', suff='', label_delim=None):
            store_attr()
            self.pref = str(pref) + os.path.sep if isinstance(pref, Path) else pref
            self.cols = L(cols)

        def _do_one(self, r, c):
            o = r[c] if isinstance(c, int) or not c in getattr(r, '_fields', []) else getattr(r, c)
            # o can be a string (relative_path) or a tuple of (relative_path, bbox_coords)
            bbox=None
            if isinstance(o,(list,tuple)) and len(o)==2 and len(o[1])==4:
                bbox = o[1]
                o = o[0]
            if len(self.pref)==0 and len(self.suff)==0 and self.label_delim is None: 
                return o if not bbox else [[o,bbox]]

            return f'{self.pref}{o}{self.suff}' if not bbox else [[f'{self.pref}{o}{self.suff}',bbox]]
                        

        def __call__(self, o, **kwargs):
            if len(self.cols) == 1: return self._do_one(o, self.cols[0])
            return L(self._do_one(o, c) for c in self.cols)
        
    def ImageDataLoaders_from_df(df, path='.', valid_pct=0.2, seed=None, fn_col=0, folder=None, suff='', label_col=1, label_delim=None,
                y_block=None, valid_col=None, item_tfms=None, batch_tfms=None, **kwargs):
        "Create from `df` in `path` using `fn_col` and `label_col`"
        pref = f'{Path(path) if folder is None else Path(path)/folder}{os.path.sep}'
        
        if y_block is None:
            is_multi = (is_listy(label_col) and len(label_col) > 1) or label_delim is not None
            y_block = MultiCategoryBlock if is_multi else CategoryBlock
        splitter = RandomSplitter(valid_pct, seed=seed) if valid_col is None else ColSplitter(valid_col)

        PILImageClass = PILImageFactory()
        col_reader = ColMDReader(fn_col, pref=pref, suff=suff)

        # check, if df[fn_col] also contains bbox, then each bbox must be tuple of float instead of str
        if not isinstance(fn_col,int):
            raise Exception('fn_col must be an integer, which is the index of the filename column. Note that this column can contain a tuple of (name,bbox)')
        if isinstance(df.iloc[0,fn_col],(tuple,list)) and len(df.iloc[0,fn_col])==2 and not isinstance(df.iloc[0,fn_col][1],(tuple,list)):
            print('Convert bbox to tuple format')
            df.iloc[:,fn_col] = df.iloc[:,fn_col].apply(lambda x: (x[0],tuple(ast.literal_eval(x[1]))))

        dblock = DataBlock(blocks=(ImageBlock(PILImageClass), y_block),
                           get_x=col_reader,
                           get_y=ColReader(label_col, label_delim=label_delim),
                           splitter=splitter,
                           item_tfms=item_tfms,
                           batch_tfms=batch_tfms)
        return ImageDataLoaders.from_dblock(dblock, df, path=path, num_workers=n_workers, **kwargs)
    
    use_wandb = 'WANDB_PROJECT' in config

    if 'SEED' in config:
        seed = config['SEED']
        seed_everything(seed)
    else:
        seed=None
    
    _item_tfms = Resize(750)
    if 'ITEM_RESIZE' in config:
        if isinstance(config['ITEM_RESIZE'],int):
            _item_tfms = Resize(config['ITEM_RESIZE'])
        elif isinstance(config['ITEM_RESIZE'],str) and not config['ITEM_RESIZE'].lower().strip() in ['none','']:
            try:
                _item_tfms = Resize(int(config['ITEM_RESIZE']))
            except:
                _item_tfms = Resize(750)            
        else:
            _item_tfms = config['ITEM_RESIZE']

    dls = ImageDataLoaders_from_df(df, 
                                   path=config['IMAGE_DIRECTORY'],
                                   seed=seed,
                                   fn_col=0,
                                   label_col=1,
                                   valid_col='is_val',
                                   item_tfms= _item_tfms,
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
    pct_start = config['PCT_START'] if 'PCT_START' in config else 0.25
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
        if 'FREEZE_EPOCH' in config and config['FREEZE_EPOCH']>0:
            learn.fine_tune(epoch,freeze_epochs=freeze_epoch,base_lr=config['LR'])
        else:
            learn.unfreeze()
            learn.fit_one_cycle(epoch,config['LR'],pct_start=pct_start)
        
    else:
        with learn.no_bar(), learn.no_logging():
            if 'FREEZE_EPOCH' in config and config['FREEZE_EPOCH']>0:
                learn.fine_tune(epoch,freeze_epochs=freeze_epoch,base_lr=config['LR'])
            else:
                learn.unfreeze()
                learn.fit_one_cycle(epoch,config['LR'],pct_start=pct_start)

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
#     'SAVE_DIRECTORY':'absolute/path/to/save/directory',
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
            # inps[0] = _download_img_tiny(input_container_client,inps[0])
            inps[0] = download_img(inps[0],input_container_client)
            img = PILImage.create(inps[0])
            norm_bbox = inps[1]
            img = crop_image(img,norm_bbox,square_crop=True)
            img = PILImage.create(img)
        else:
            # inps = _download_img_tiny(input_container_client,inps)
            inps = download_img(inps,input_container_client)
            img = PILImage.create(inps)
    
    except Exception as e:
        return False
    else:
        return True

def prepare_inference_dataloader(inputs, # list of image paths or tuples of (image_path,bbox)
                                 input_container_sas=None,
                                 do_image_check=False,
                                 item_tfms=None,
                                 aug_tfms=None,
                                 batch_size=16,
                                 pin_memory=False,
                                 n_workers=1):

    if isinstance(inputs[0],str) or (len(inputs[0])==2 and isinstance(inputs[0][0],str) and len(inputs[0][1])==4):
        inputs = np.array(inputs,dtype='object')
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
                          item_tfms = item_tfms,
                          batch_tfms = aug_tfms,
                          splitter = lambda x: (L(0),L(list(torch.arange(len(valid_idxs)).numpy())))
                          )
    dls = DataLoaders.from_dblock(datablock,
                                  inputs[valid_idxs],
                                  bs=batch_size,
                                  num_workers=n_workers,
                                  pin_memory=pin_memory,
                                  shuffle=False)
    return dls,valid_idxs

def load_efficientnet_model(finetuned_model, 
                            efficient_model='efficientnet-b3', 
                            label_info=None # list of output labels, or the number of labels
                            ):
    w, d, s, p = efficientnet_params(efficient_model)
    blocks_args, global_params = efficientnet(include_top=True,
                                            width_coefficient=w, 
                                            depth_coefficient=d, 
                                            dropout_rate=p, # 0.3
                                            image_size=s,
                                            num_classes=label_info if isinstance(label_info,int) else len(label_info),
                                            )
    model = EfficientNet(blocks_args, global_params)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(finetuned_model, map_location=device)
    ret = model.load_state_dict(state_dict, strict=False)
    if len(ret.missing_keys):
        print(f'Missing weights: {ret.missing_keys}')
    return model

class EffNetClassificationInference:
    def __init__(self,
                 label_info, # list of output labels, or the number of labels
                 finetuned_model, # absolute path to efficient model that has been finetuned
                 efficient_model='efficientnet-b3', # name of pretrained efficient model
                 item_tfms=Resize(750), # list of item transformations
                 aug_tfms=None, # augmentation transformations, needed if TTA is used
                 parent_info=None, # list of parent labels, or nuber of parent labels, needed for hierarchical classification/rollup classification
                 child2parent=None, # dictionary of child to parent mapping, needed for hierarchical classification
                 child_threshold=None, # threshold, any child label with probability less than this will be replaced with parent label, needed for hierarchical classification
                #  l1_morethan=None, # threshold, any parent label with probability more than this will be chosen, needed for hierarchical classification
                 parent2child=None, # dictionary of parent to child mapping, needed for rollup classification
                 rollup_threshold=0.75 # threshold for rollup classification, default is 0.75
                ):
        
        # check whether finetuned_model string ends with .pth or .pt
        finetuned_model = str(finetuned_model)
        if not (finetuned_model.endswith('.pth') or finetuned_model.endswith('.pt')):
            finetuned_model = finetuned_model + '.pth'
        finetuned_model = Path(finetuned_model)
        self.item_tfms = item_tfms
        self.aug_tfms = aug_tfms
        self.is_hitax = False
        self.is_rollup = False
        self.parent_info = parent_info
        self.label_info = label_info
        self.child2parent = child2parent
        self.child_threshold = child_threshold # if defined, 0.75 is the default
        self.parent2child = parent2child
        self.rollup_threshold = rollup_threshold
        self.model = None
        if parent_info is not None:
            if child2parent is not None:
                self.is_hitax = True
                if isinstance(parent_info,int) or isinstance(label_info,int):
                    raise Exception('For HiTax model, parent_info and label_info must each be a list of string labels, not number of labels')
                label_count = len(label_info)
                parent_count = len(parent_info)
                assert label_count == len(child2parent), \
                    f"Label (children) names and child2parent mapping lengths do not match: {label_count} != {len(child2parent)}"

                self.child2parent_idx = torch.tensor([parent_info.index(child2parent[ch]) for ch in label_info], dtype=torch.int32)
                self.model = load_hier_model(parent_count = parent_count,
                                            children_count = label_count,
                                            lin_dropout_rate=0.3,
                                            last_hidden=256,
                                            use_simple_head=True,
                                            base_model=efficient_model,
                                            trained_weight_path=finetuned_model,
                                            )
            elif parent2child is not None:
                self.is_rollup = True
                if isinstance(parent_info,int) or isinstance(label_info,int):
                    raise Exception('For rollup model, parent_info and label_info must each be a list of string labels, not number of labels')
                self.agg_maps_2L = precompute_rollup_maps_dynamic([parent_info,label_info], [parent2child])

        if self.model is None:
            self.model = load_efficientnet_model(finetuned_model=finetuned_model,
                                                 efficient_model=efficient_model,
                                                 label_info=label_info)

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

    def create_output_df(self,inputs,probs,pred_idxs,valid_idxs,name_output):
        if isinstance(inputs[0],str):
            df= pd.DataFrame(inputs,columns=['file'])
        else:
            df= pd.DataFrame(inputs,columns=['file','detection_bbox'])

        top_n = probs.shape[1]
        df_pred = pd.DataFrame(pred_idxs,columns=[f'pred_{i+1}' for i in range(top_n)])
        if name_output and isinstance(self.label_info,(list,tuple,np.ndarray)):
            df_pred = df_pred.map(lambda x: self.label_info[x])            
        df_prob = pd.DataFrame(probs,columns=[f'prob_{i+1}' for i in range(top_n)])

        df_pred.index=valid_idxs
        df_prob.index=valid_idxs
        
        df = pd.concat([df,df_pred,df_prob],axis=1)
        # file	detection_bbox	pred_1	pred_2	pred_3	prob_1	prob_2	prob_3
        return df

    def create_output_df_hitax_merge(self,inputs,probs,preds,level,valid_idxs,name_output,is_rollup=False):
        if isinstance(inputs[0],str):
            df= pd.DataFrame(inputs,columns=['file'])
        else:
            df= pd.DataFrame(inputs,columns=['file','detection_bbox'])

        df_pred = pd.DataFrame(preds,columns=['pred_1'])
        df_prob = pd.DataFrame(probs,columns=['prob_1'])
        df_level = pd.DataFrame(level,columns=['level'])
        df_pred.index=valid_idxs
        df_prob.index=valid_idxs
        df_level.index=valid_idxs
        df = pd.concat([df,df_pred,df_prob,df_level],axis=1)
        if isinstance(self.label_info,(list,tuple,np.ndarray)) and isinstance(self.parent_info,(list,tuple,np.ndarray)):
            if is_rollup and not name_output:
                # since rollup output (pred_1) is string labels, for not name_output, we need to convert them to indices
                parent2idx = {v:i for i,v in enumerate(self.parent_info)}
                label2idx = {v:i for i,v in enumerate(self.label_info)}
                df.loc[(~df['level'].isna() & df['level']==1),'pred_1'] = df.loc[(~df['level'].isna() & df['level']==1)].map(lambda x: parent2idx[x])
                df.loc[(~df['level'].isna() & df['level']==2),'pred_1'] = df.loc[(~df['level'].isna() & df['level']==2)].map(lambda x: label2idx[x])
            elif not is_rollup and name_output:
                # this is hitax with only 1 prediction pred_1 each row (which is an index)
                df.loc[(~df['level'].isna() & df['level']==1),'pred_1'] = df.loc[(~df['level'].isna() & df['level']==1)].map(lambda x: self.parent_info[x])
                df.loc[(~df['level'].isna() & df['level']==2),'pred_1'] = df.loc[(~df['level'].isna() & df['level']==2)].map(lambda x: self.label_info[x])
        # file  detection_bbox  pred_1  prob_1  level  
        return df

    def create_output_df_hitax(self,inputs,prob_l1,pred_l1_idxs,prob_l2,pred_l2_idxs,valid_idxs,name_output):
        if isinstance(inputs[0],str):
            df= pd.DataFrame(inputs,columns=['file'])
        else:
            df= pd.DataFrame(inputs,columns=['file','detection_bbox'])
        top_n = prob_l2.shape[1]
        df_l1_pred = pd.DataFrame(pred_l1_idxs,columns=[f'parent_pred_{i+1}' for i in range(top_n)])
        df_l2_pred = pd.DataFrame(pred_l2_idxs,columns=[f'child_pred_{i+1}' for i in range(top_n)])
        if name_output and isinstance(self.parent_info,(list,tuple,np.ndarray)):
            df_l1_pred = df_l1_pred.map(lambda x: self.parent_info[x])
            df_l2_pred = df_l2_pred.map(lambda x: self.label_info[x])
        df_l1_prob = pd.DataFrame(prob_l1,columns=[f'parent_prob_{i+1}' for i in range(top_n)])
        df_l2_prob = pd.DataFrame(prob_l2,columns=[f'child_prob_{i+1}' for i in range(top_n)])
        df_l1_pred.index=valid_idxs
        df_l2_pred.index=valid_idxs
        df_l1_prob.index=valid_idxs
        df_l2_prob.index=valid_idxs
        df = pd.concat([df,df_l1_pred,df_l1_prob,df_l2_pred,df_l2_prob],axis=1)
        # file  detection_bbox  parent_pred_1  parent_prob_1  
        #                       child_pred_1   child_prob_1 
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
        if (not isinstance(inputs, Iterable)) or isinstance(inputs,str):
            inputs = np.array([inputs])
        if isinstance(inputs, pd.DataFrame):
            inputs = inputs.copy()
            inputs = self.validate_df(inputs)
        if len(inputs)==0: return pd.DataFrame()
        
        dls,valid_idxs = prepare_inference_dataloader(inputs,
                                                      input_container_sas=input_container_sas,
                                                      do_image_check=do_image_check,
                                                      item_tfms=self.item_tfms,
                                                      aug_tfms=self.aug_tfms,
                                                      batch_size=batch_size,
                                                      pin_memory=pin_memory,
                                                      n_workers=n_workers)
        
        if not self.is_hitax: # including normal classification and rollup classification
            loss_func = CrossEntropyLossFlat()
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            loss_func = HierarchicalClassificationLoss(self.parent_info if isinstance(self.parent_info,int) else len(self.parent_info),
                                                       l1_weight=1,
                                                       l2_weight=2,
                                                       consistency_weight=0,
                                                       focal_gamma=1.0,
                                                       child_to_parent_mapping= self.child2parent_idx.to(device))
        learner = Learner(dls,self.model,
                          loss_func = loss_func)

        if tta_n>0 and not self.is_hitax:
            preds = learner.tta(dl = dls.valid,n=tta_n)[0]
        else:
            preds = learner.get_preds(dl = dls.valid)[0]

        if not self.is_hitax:
            if not self.is_rollup:
                # default
                preds = torch.round(preds,decimals = prob_round)
                preds,pred_idxs = preds.sort(dim=1,descending=True)
                preds = preds[:,:pred_topn]
                pred_idxs = pred_idxs[:,:pred_topn]
                return self.create_output_df(inputs,preds,pred_idxs,valid_idxs,name_output)
            else:
                # rollup
                preds = torch.round(preds,decimals = 4)
                df_preds_rollup = rollup_predictions_dynamic(softmax_most_specific=preds,
                                                             all_level_labels=[self.parent_info,self.label_info],
                                                             aggregation_maps=self.agg_maps_2L,
                                                             threshold=self.rollup_threshold)
                # 'prediction' 'probability' 'level' 'passes_threshold'
                # 'level' is 1 for parent, 2 for child
                pred_str = df_preds_rollup['prediction'].values
                level = df_preds_rollup['level'].values
                probs = df_preds_rollup['probability'].values
                return self.create_output_df_hitax_merge(inputs,probs,pred_str,level,valid_idxs,name_output,
                                                          is_rollup=True)
        
        # hitax
        parent_length = self.parent_info if isinstance(self.parent_info,int) else len(self.parent_info)
        pred_l1_prob = torch.round(preds[:,:len(parent_length)].softmax(axis=1),decimals=prob_round) # parent probabilities (level 1)
        pred_l2_prob = torch.round(preds[:,len(parent_length):].softmax(axis=1),decimals=prob_round) # child probabilities (level 2)
        pred_l1_prob,pred_l1_idxs = pred_l1_prob.sort(dim=1,descending=True)
        pred_l2_prob,pred_l2_idxs = pred_l2_prob.sort(dim=1,descending=True)
        if self.child_threshold is not None:
            pred_l1_prob = pred_l1_prob[:,0]
            pred_l2_prob = pred_l2_prob[:,0]
            pred_l1_idxs = pred_l1_idxs[:,0]
            pred_l2_idxs = pred_l2_idxs[:,0]
            # if child probability is less than threshold, replace with parent probability
            _mask = pred_l2_prob < self.child_threshold
            pred_l2_prob[_mask] = pred_l1_prob[_mask]
            pred_l2_idxs[_mask] = pred_l1_idxs[_mask]
            # 'level' is 1 for parent (that was replaced), 2 for child
            level= torch.where(_mask,1,2)
            return self.create_output_df_hitax_merge(inputs,pred_l2_prob,pred_l2_idxs,level,valid_idxs,name_output,is_rollup=False)
        
        pred_l1_prob = pred_l1_prob[:,:pred_topn]
        pred_l2_prob = pred_l2_prob[:,:pred_topn]
        pred_l1_idxs = pred_l1_idxs[:,:pred_topn]
        pred_l2_idxs = pred_l2_idxs[:,:pred_topn]
        return self.create_output_df_hitax(inputs,pred_l1_prob,pred_l1_idxs,pred_l2_prob,pred_l2_idxs,valid_idxs,name_output)
        
