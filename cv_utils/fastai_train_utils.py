from fastai.vision.all import *
import ast
from .fastai_utils import PILImageFactory

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
            y_block=None, valid_col=None, item_tfms=None, batch_tfms=None, n_workers=None, **kwargs):
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