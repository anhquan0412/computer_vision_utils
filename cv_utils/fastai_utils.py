from sklearn.metrics import precision_recall_fscore_support
from fastai.vision.all import *
from fastai.callback.wandb import *
from fastai.callback.training import GradientClip
from fastai.losses import LabelSmoothingCrossEntropy
from collections.abc import Iterable
import ast
import wandb
from azure.storage.blob import ContainerClient
import warnings; warnings.simplefilter('ignore')
from .img_utils import crop_image, download_img
from .common_utils import check_and_fix_http_path
from .hierarchical_model import HierarchicalClassificationLoss,get_precision_recall_f1_metrics_group,get_precision_recall_f1_metrics_dhc
from .hierarchical_rollup import precompute_rollup_maps_dynamic, rollup_predictions_dynamic
from .viz_utils import clas_report_compact,plot_classification_report
import timm
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

def fastai_predict_val(learner,label_names,path_prefix,df_val=None,tta_n=2):
    def _get_label_for_plot(x_prob):
        x_sort = x_prob.iloc[x_prob.argsort()[::-1][:2]] # get the top 2 probabilities and predictions. Note: hardcode
        return np.array([x_sort,x_sort.index]).flatten()
    path_prefix = str(path_prefix)
    if tta_n>0:
        val_probs,val_true = learner.tta(n=tta_n)
        val_pred = val_probs.max(axis=1)[1]
    else:
        val_probs,val_true,val_pred = learner.get_preds(with_decoded=True,with_preds=True,with_input=False)
    val_pred_str = list(map(lambda x: label_names[x],val_pred))
    val_true_str = list(map(lambda x: label_names[x],val_true))
    df_pred = pd.DataFrame()
    df_pred['y_pred'] = val_pred_str
    df_pred['y_true'] = val_true_str
    df_pred[label_names] = torch.round(val_probs,decimals=5)
    # hardcode true and pred columns in df_pred
    report_df_compact,report_df = clas_report_compact(df_pred['y_true'].tolist(),df_pred['y_pred'].tolist(),
                                      label_names=label_names)
    
    df_show=None
    if df_val is not None:
        if len(df_val)==len(df_pred):
            df_show = pd.concat([df_val.reset_index(drop=True),df_pred],axis=1)
            df_show['abs_file'] = df_show['file_and_bbox'].apply(lambda x: ast.literal_eval(x)[0] if isinstance(x,str) else x[0])
            df_show['bbox'] = df_show['file_and_bbox'].apply(lambda x: ast.literal_eval(x)[1] if isinstance(x,str) else x[1])
        else:
            print(f'Mismatch length between validation data ({len(df_val)}) and validation predictions ({len(df_pred)})')

    if df_show is None:
        df_show = df_pred

    df_prob = pd.DataFrame(df_show[label_names].apply(_get_label_for_plot,axis=1).tolist(),
                           columns=['y_prob1','y_prob2','y_pred1','y_pred2'])
    df_prob = pd.concat([df_show[['y_true']].copy(),df_prob],axis=1)

    metadata_cols = list(set(['abs_file','bbox','identifier','identifier_2','is_color','is_prev']) & set(df_show.columns))
    if len(metadata_cols):
        df_show = pd.concat([df_show[metadata_cols].copy(),df_prob],axis=1) #note: hardcode columns
    else:
        df_show = df_prob

    # add a column label_show, which is (prob1, prob2) if pred2==true, else (prob1, pred2: prob2)
    def _format_label_show(x):
        _tmp = x['y_pred1'].split('|')
        pred1_label = _tmp[1].strip() if len(_tmp) > 1 else _tmp[0].strip()
        _tmp = x['y_pred2'].split('|')
        pred2_label = _tmp[1].strip() if len(_tmp) > 1 else _tmp[0].strip()
        return f"{pred1_label}: {round(x['y_prob1'],2)}\n{pred2_label}: {round(x['y_prob2'],2)}"
    df_show['label_show'] = df_show[['y_true','y_prob1','y_prob2','y_pred1','y_pred2']].apply(lambda x: _format_label_show(x),
                                                                                                axis=1)

    report_df_compact.to_csv(path_prefix + '_short_report.csv')
    report_df.to_csv(path_prefix + '_full_report.csv')
    plot_classification_report(report_df,figsize=(30,16),fontsize=10,fname=path_prefix + '_full_report.png')
    plot_classification_report(report_df[report_df['f1-score']<=0.9],figsize=(16,10),fontsize=10,fname=path_prefix + '_low_f1_report.png')

    df_pred.to_csv(path_prefix + '_val_pred.csv',index=False)
    df_show.to_csv(path_prefix + '_val_pred_for_show.csv',index=False)
    print(f'Predictions saved with path prefix {path_prefix}')


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

def fastai_cv_train(config,df,aug_tfms=None,label_names=None,save_valid_pred=False,tta_n=0):
    # The first column of df should be the file path, or a tuple of file path and bbox coord
    # The second column is the label (string)
    # There is a column called 'is_val', for train val split (boolean)
    from .fastai_train_utils import ImageDataLoaders_from_df
    
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

    n_workers = config['N_WORKERS'] if 'N_WORKERS' in config else 4
    dls = ImageDataLoaders_from_df(df, 
                                   path=config['IMAGE_DIRECTORY'],
                                   seed=seed,
                                   fn_col=0,
                                   label_col=1,
                                   valid_col='is_val',
                                   item_tfms= _item_tfms,
                                   bs=config['BATCH_SIZE'],
                                   shuffle=True,
                                   batch_tfms=aug_tfms,
                                   n_workers=n_workers
                                  )
        
    if not label_names:
        label_names = dls.vocab.items.items

    timm_model_name = config['CLASSIFICATION_MODEL'].replace('-', '_')
    model = timm.create_model(timm_model_name, pretrained=True, num_classes=len(label_names))
    
    monitor_metric = config['MONITOR_METRIC'] if 'MONITOR_METRIC' in config else 'f1_score' #'valid_loss'
    save_every_epoch = config['SAVE_EVERY_EPOCH'] if 'SAVE_EVERY_EPOCH' in config else False
    save_directory = Path(config['SAVE_DIRECTORY']) if 'SAVE_DIRECTORY' in config else Path('.')/'model'
    save_directory.mkdir(exist_ok=True,parents=True)
    save_name = config['SAVE_NAME'] if 'SAVE_NAME' in config else 'model'
    pct_start = config['PCT_START'] if 'PCT_START' in config else 0.25
    log_metrics = config['LOG_LABEL_METRICS'] if 'LOG_LABEL_METRICS' in config else []
    assert set(log_metrics) - set(['precision','recall','f1'])==set()
    
    # Enhanced training parameters for Vision Transformers
    label_smoothing = config['LABEL_SMOOTHING'] if 'LABEL_SMOOTHING' in config else None
    gradient_clip = config['GRADIENT_CLIP'] if 'GRADIENT_CLIP' in config else None
    weight_decay = config['WEIGHT_DECAY'] if 'WEIGHT_DECAY' in config else None
    
    cbs=[
            SaveModelCallback(monitor=monitor_metric,
                              every_epoch=save_every_epoch,
                              fname=(save_directory/save_name),
                              comp=np.greater if 'loss' not in monitor_metric else np.less,
                              ),
            CSVLogger(fname=(save_directory/f"{save_name}_training_log.csv"), append=True)
        ]
    
    # Add gradient clipping callback if specified
    if gradient_clip is not None and gradient_clip > 0:
        cbs.append(GradientClip(max_norm=gradient_clip))
        print(f"Using gradient clipping with max_norm={gradient_clip}")
    
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

    # Set up loss function with optional label smoothing
    loss_func = None
    if label_smoothing is not None and label_smoothing > 0:
        loss_func = LabelSmoothingCrossEntropy(eps=label_smoothing)
        print(f"Using label smoothing with eps={label_smoothing}")

    learn = Learner(dls, model, 
                    metrics=[F1Score(average='macro')]+metric_lists,
                    cbs=cbs,
                    loss_func=loss_func,
                   ).to_fp16()
    
    # Print weight decay information
    if weight_decay is not None and weight_decay > 0:
        print(f"Using weight decay: {weight_decay}")

    bold_print('training model')
    epoch = config['EPOCH']
    if len(metric_lists)<8:
        if 'FREEZE_EPOCH' in config and config['FREEZE_EPOCH']>0:
            learn.fine_tune(epoch,freeze_epochs=config['FREEZE_EPOCH'],base_lr=config['LR'],wd=weight_decay)
        else:
            learn.unfreeze()
            learn.fit_one_cycle(epoch,config['LR'],pct_start=pct_start,wd=weight_decay)
        
    else:
        with learn.no_bar(), learn.no_logging():
            if 'FREEZE_EPOCH' in config and config['FREEZE_EPOCH']>0:
                learn.fine_tune(epoch,freeze_epochs=config['FREEZE_EPOCH'],base_lr=config['LR'],wd=weight_decay)
            else:
                learn.unfreeze()
                learn.fit_one_cycle(epoch,config['LR'],pct_start=pct_start,wd=weight_decay)

    _ax = learn.recorder.plot_loss(show_epochs=True)
    plt.savefig((save_directory/f'{save_name}_learning_curve.png'), bbox_inches='tight')

    if save_valid_pred:
        bold_print('predicting validation set')
        df_val = df[df['is_val']==True].copy()
        path_prefix = str(save_directory/save_name)
        fastai_predict_val(learn,label_names,df_val=df_val,path_prefix=path_prefix,tta_n=tta_n)

    if use_wandb:
        wandb.finish();
    return learn


def fastai_cv_train_hierarchical_model(config,df,aug_tfms=None,parent_label=None,children_label=None,concat_label=None):
    # df should have these columns
    # - file_and_bbox: a tuple/list of (file_path, bbox), or a list of file_path. file_path is relative path
    # - parent_label: the parent label (string)
    # - children_label: the child label (string)
    # - concat_label: the concatenated label (string) of parent and children labels, separated by symbol $
    # - is_val: boolean, True if the row is for validation set, False otherwise

    from .fastai_train_utils import ImageDataLoaders_from_df

    parent_labels = np.sort(df[parent_label].unique()).tolist()
    children_labels = np.sort(df[children_label].unique()).tolist()
    child2parent = list(df[[children_label,parent_label]].drop_duplicates().set_index(children_label).to_dict().values())[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    child2parent_idx = torch.tensor([parent_labels.index(child2parent[ch]) for ch in children_labels],dtype=torch.int32).to(device)

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
    
    n_workers = config['N_WORKERS'] if 'N_WORKERS' in config else 4
    dls = ImageDataLoaders_from_df(df, 
                                   path=config['IMAGE_DIRECTORY'],
                                   seed=seed,
                                   fn_col=0,
                                   label_col=concat_label,
                                   label_delim='$',
                                   valid_col='is_val',
                                   item_tfms= _item_tfms,
                                   bs=config['BATCH_SIZE'],
                                   shuffle=True,
                                   batch_tfms=aug_tfms,
                                   n_workers=n_workers
                                  )
    # Import and use the new timm-compatible hierarchical model loader
    from .hierarchical_model import load_hier_model_timm
    hier_model = load_hier_model_timm(parent_count = len(parent_labels),
                                      child_count = len(children_labels),
                                      base_model = config['CLASSIFICATION_MODEL'].replace('-', '_'),
                                      lin_dropout_rate=config.get('HITAX_DROPOUT', 0.3),
                                      last_hidden=config.get('HITAX_LAST_HIDDEN', 256),
                                      use_simple_head=config.get('HITAX_USE_SIMPLE_HEAD', True)
                                    )

    save_directory = Path(config['SAVE_DIRECTORY']) if 'SAVE_DIRECTORY' in config else Path('.')/'hier_model'
    save_directory.mkdir(exist_ok=True,parents=True)
    save_name = config['SAVE_NAME'] if 'SAVE_NAME' in config else 'hier_model'
    pct_start = config['PCT_START'] if 'PCT_START' in config else 0.25
    log_metrics = config['LOG_LABEL_METRICS'] if 'LOG_LABEL_METRICS' in config else []
    assert set(log_metrics) - set(['precision','recall','f1'])==set()

    # Enhanced training parameters for Vision Transformers
    gradient_clip = config['GRADIENT_CLIP'] if 'GRADIENT_CLIP' in config else None
    weight_decay = config['WEIGHT_DECAY'] if 'WEIGHT_DECAY' in config else None

    cbs=[
        SaveModelCallback(every_epoch=True,
                          fname=(save_directory/save_name)
                          ),
        CSVLogger(fname=(save_directory/f"{save_name}_training_log.csv"), append=True)
    ]
    # Add gradient clipping callback if specified
    if gradient_clip is not None and gradient_clip > 0:
        cbs.append(GradientClip(max_norm=gradient_clip))
        print(f"Using gradient clipping with max_norm={gradient_clip}")

    mixup_alpha = config['MIXUP_ALPHA'] if ('MIXUP_ALPHA' in config and config['MIXUP_ALPHA']>0) else None
    if mixup_alpha is not None:
        cbs.append(MixUp(mixup_alpha))

    use_wandb = 'WANDB_PROJECT' in config
    if use_wandb:
        os.environ["WANDB_SILENT"] = "true"
        wandb.init(project=config['WANDB_PROJECT'],name=save_name,config=config);
        cbs.append(WandbCallback(log_dataset=False, log_model=False, n_preds=49))


    metric_lists = get_precision_recall_f1_metrics_group(parent_labels)
    for lm in log_metrics:
        metric_lists += get_precision_recall_f1_metrics_dhc(parent_labels,children_labels,mtype=lm)

    l1_weight = config['L1_WEIGHT'] if 'L1_WEIGHT' in config else 1.0
    l2_weight = config['L2_WEIGHT'] if 'L2_WEIGHT' in config else 2.0
    focal_gamma = config['FOCAL_GAMMA'] if 'FOCAL_GAMMA' in config else 1.0

    learn = Learner(dls, hier_model, 
                metrics=metric_lists,
                loss_func = HierarchicalClassificationLoss(len(parent_labels),l1_weight,l2_weight,0.,
                                                           focal_gamma=focal_gamma,
                                                          child_to_parent_mapping=child2parent_idx),
                cbs=cbs
               ).to_fp16()

    # Print weight decay information
    if weight_decay is not None and weight_decay > 0:
        print(f"Using weight decay: {weight_decay}")

    bold_print('training model')
    epoch = config['EPOCH']

    if len(metric_lists)<8:
        if 'FREEZE_EPOCH' in config and config['FREEZE_EPOCH']>0:
            learn.fine_tune(epoch,freeze_epochs=config['FREEZE_EPOCH'],base_lr=config['LR'],wd=weight_decay)
        else:
            learn.unfreeze()
            learn.fit_one_cycle(epoch,config['LR'],pct_start=pct_start,wd=weight_decay)
        
    else:
        with learn.no_bar(), learn.no_logging():
            if 'FREEZE_EPOCH' in config and config['FREEZE_EPOCH']>0:
                learn.fine_tune(epoch,freeze_epochs=config['FREEZE_EPOCH'],base_lr=config['LR'],wd=weight_decay)
            else:
                learn.unfreeze()
                learn.fit_one_cycle(epoch,config['LR'],pct_start=pct_start,wd=weight_decay)



    _ax = learn.recorder.plot_loss(show_epochs=True)
    plt.savefig((save_directory/f'{save_name}_learning_curve.png'), bbox_inches='tight')
    

    
    if use_wandb:
        wandb.finish();
    return learn

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

def load_classification_model(finetuned_model, 
                            classification_model='tf_efficientnet_b5.ns_jft_in1k', 
                            label_info=None, # list of output labels, or the number of labels
                            image_size=None
                            ):
    # Convert model name format for timm
    timm_model_name = classification_model.replace('-', '_')
    num_classes = label_info if isinstance(label_info, int) else len(label_info)
    
    # Create model with timm (without pretrained weights)
    model = timm.create_model(timm_model_name, pretrained=False, num_classes=num_classes)
    
    # Load fine-tuned weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(finetuned_model, map_location=device)
    ret = model.load_state_dict(state_dict, strict=False)
    if len(ret.missing_keys):
        print(f'Missing weights: {ret.missing_keys}')
    if len(ret.unexpected_keys):
        print(f'Unexpected weights: {ret.unexpected_keys}')
    
    print(f'Loaded timm classification model: {timm_model_name} with {num_classes} classes')
    return model

class ClassificationInference:
    def __init__(self,
                 label_info, # list of output labels, or the number of labels
                 finetuned_model, # absolute path to classification model that has been finetuned
                 classification_model='tf_efficientnet_b5.ns_jft_in1k', # name of pretrained classification model
                 item_tfms=Resize(750), # list of item transformations
                 aug_tfms=None, # augmentation transformations, needed if TTA is used
                 parent_info=None, # list of parent labels, or nuber of parent labels, needed for hierarchical classification/rollup classification
                 child2parent=None, # dictionary of child to parent mapping, needed for hierarchical classification
                 parent2child=None, # dictionary of parent to child mapping, needed for rollup classification
                 hitax_threshold=0.75, # threshold for for hitax or rollup classification, default is 0.75
                 #  l1_morethan=None, # threshold, any parent label with probability more than this will be chosen, needed for hierarchical classification
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
        self.hitax_threshold = hitax_threshold
        self.parent2child = parent2child
        self.model = None

        image_size = None
        if hasattr(self.item_tfms, 'size'):
            # Handles Resize, Squish, Pad, Crop
            size = self.item_tfms.size
            if isinstance(size, int):
                image_size = size
            elif isinstance(size, (list, tuple)) and len(size) == 2:
                image_size = size[0] # Assume square images for simplicity, take height

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
                # Import the new timm-compatible hierarchical model loader
                from .hierarchical_model import load_hier_model_timm
                self.model = load_hier_model_timm(parent_count = parent_count,
                                                children_count = label_count,
                                                lin_dropout_rate=0.3,
                                                last_hidden=256,
                                                use_simple_head=True,
                                                base_model=classification_model,
                                                trained_weight_path=finetuned_model,
                                                image_size=image_size
                                                )
            elif parent2child is not None:
                self.is_rollup = True
                if isinstance(parent_info,int) or isinstance(label_info,int):
                    raise Exception('For rollup model, parent_info and label_info must each be a list of string labels, not number of labels')
                assert len(parent_info) == len(parent2child), \
                    f"Parent names and parent2child mapping lengths do not match: {len(parent_info)} != {len(parent2child)}"
                self.agg_maps_2L = precompute_rollup_maps_dynamic([parent_info,label_info], [parent2child])

        if self.model is None:
            self.model = load_classification_model(finetuned_model=finetuned_model,
                                                 classification_model=classification_model,
                                                 label_info=label_info,
                                                 image_size=image_size)
        self.model.eval()

    def validate_df(self,df):
        if 'file' in df.columns.tolist():
            if len(df)==0: return []
            if 'detection_bbox' in df.columns.tolist():
                if not isinstance(df['detection_bbox'].values[0],(list,tuple)):
                    df['detection_bbox'] = df['detection_bbox'].apply(lambda x: None if (x is None or x is np.nan) else list(ast.literal_eval(x)))
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
            df_pred = df_pred.map(lambda x: self.label_info[int(x)])            
        df_prob = pd.DataFrame(probs,columns=[f'prob_{i+1}' for i in range(top_n)])

        df_pred.index=valid_idxs
        df_prob.index=valid_idxs
        
        df = pd.concat([df,df_pred,df_prob],axis=1)
        # file	detection_bbox	pred_1	pred_2	pred_3	prob_1	prob_2	prob_3
        return df

    def create_output_df_hitax_onepred(self,inputs,probs,preds,level,valid_idxs,name_output,is_rollup=False):
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
                # since rollup output (pred_1) is string labels, not name_output, we need to convert them to indices
                parent2idx = {v:i for i,v in enumerate(self.parent_info)}
                label2idx = {v:i for i,v in enumerate(self.label_info)}
                df.loc[(~df['level'].isna()) & (df['level']==1),'pred_1'] = df.loc[(~df['level'].isna()) & (df['level']==1),'pred_1'].map(lambda x: parent2idx[x])
                df.loc[(~df['level'].isna()) & (df['level']==2),'pred_1'] = df.loc[(~df['level'].isna()) & (df['level']==2),'pred_1'].map(lambda x: label2idx[x])
            elif not is_rollup and name_output:
                # this is hitax with only 1 prediction pred_1 each row (which is an index)
                df.loc[(~df['level'].isna()) & (df['level']==1),'pred_1'] = df.loc[(~df['level'].isna()) & (df['level']==1),'pred_1'].map(lambda x: self.parent_info[int(x)])
                df.loc[(~df['level'].isna()) & (df['level']==2),'pred_1'] = df.loc[(~df['level'].isna()) & (df['level']==2),'pred_1'].map(lambda x: self.label_info[int(x)])
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
            df_l1_pred = df_l1_pred.map(lambda x: self.parent_info[int(x)])
            df_l2_pred = df_l2_pred.map(lambda x: self.label_info[int(x)])
        df_l1_prob = pd.DataFrame(prob_l1,columns=[f'parent_prob_{i+1}' for i in range(top_n)])
        df_l2_prob = pd.DataFrame(prob_l2,columns=[f'child_prob_{i+1}' for i in range(top_n)])
        df_l1_pred.index=valid_idxs
        df_l2_pred.index=valid_idxs
        df_l1_prob.index=valid_idxs
        df_l2_prob.index=valid_idxs
        df = pd.concat([df,df_l1_pred,df_l1_prob,df_l2_pred,df_l2_prob],axis=1)
        # file  detection_bbox  parent_pred_1  parent_prob_1  parent_pred_2  parent_prob_2
        #                       child_pred_1   child_prob_1  child_pred_2   child_prob_2

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
                pin_memory=False, # If True, the data loader will copy Tensors into CUDA pinned memory before returning them
                use_fp16=True # whether to use fp16 for inference
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
        if use_fp16:
            learner = learner.to_fp16()

        if tta_n>0 and not self.is_hitax:
            preds = learner.tta(dl = dls.valid,n=tta_n)[0]
        else:
            preds = learner.get_preds(dl = dls.valid)[0]

        # convert to fp32
        if use_fp16:
            preds = preds.float()
        
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
                                                             threshold=self.hitax_threshold)
                # 'prediction' 'probability' 'level' 'passes_threshold'
                # 'level' is 1 for parent, 2 for child
                pred_str = df_preds_rollup['prediction'].values
                level = df_preds_rollup['level'].values
                probs = df_preds_rollup['probability'].values
                return self.create_output_df_hitax_onepred(inputs,probs,pred_str,level,valid_idxs,name_output,
                                                          is_rollup=True)
        
        # hitax
        parent_length = self.parent_info if isinstance(self.parent_info,int) else len(self.parent_info)
        pred_l1_prob = torch.round(preds[:,:parent_length].softmax(axis=1),decimals=prob_round) # parent probabilities (level 1)
        pred_l2_prob = torch.round(preds[:,parent_length:].softmax(axis=1),decimals=prob_round) # child probabilities (level 2)
        pred_l1_prob,pred_l1_idxs = pred_l1_prob.sort(dim=1,descending=True)
        pred_l2_prob,pred_l2_idxs = pred_l2_prob.sort(dim=1,descending=True)
        if self.hitax_threshold is not None:
            pred_l1_prob = pred_l1_prob[:,0]
            pred_l2_prob = pred_l2_prob[:,0]
            pred_l1_idxs = pred_l1_idxs[:,0]
            pred_l2_idxs = pred_l2_idxs[:,0]
            # if child probability is less than threshold, replace with parent probability
            _mask = pred_l2_prob < self.hitax_threshold
            pred_l2_prob[_mask] = pred_l1_prob[_mask]
            pred_l2_idxs[_mask] = pred_l1_idxs[_mask]
            # 'level' is 1 for parent (that was replaced), 2 for child
            level= torch.where(_mask,1,2)
            return self.create_output_df_hitax_onepred(inputs,pred_l2_prob,pred_l2_idxs,level,valid_idxs,name_output,is_rollup=False)
        
        pred_l1_prob = pred_l1_prob[:,:pred_topn]
        pred_l2_prob = pred_l2_prob[:,:pred_topn]
        pred_l1_idxs = pred_l1_idxs[:,:pred_topn]
        pred_l2_idxs = pred_l2_idxs[:,:pred_topn]
        return self.create_output_df_hitax(inputs,pred_l1_prob,pred_l1_idxs,pred_l2_prob,pred_l2_idxs,valid_idxs,name_output)
        
