from sklearn.metrics import precision_recall_fscore_support
from fastai.vision.all import *
from fastai.callback.wandb import *
import wandb
import os
import warnings; warnings.simplefilter('ignore')
from efficientnet_pytorch import EfficientNet 

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

def fastai_cv_train_efficientnet(config,df,aug_tfms=None,label_names=None,save_valid_pred=False):
    # The first column of df should be the file path
    # The second column is the label (string)
    # The third column is the 'is_val' split (boolean)

    use_wandb = 'WANDB_PROJECT' in config

    if 'SEED' in config:
        seed = config['SEED']
        seed_everything(seed)
    else:
        seed=None
    
    dls = ImageDataLoaders.from_df(df, 
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
#     'SAVE_NAME':'sample_73',
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

