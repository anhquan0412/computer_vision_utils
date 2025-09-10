from fastai.vision.all import *
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import round_filters, MemoryEfficientSwish
from sklearn.metrics import precision_recall_fscore_support


class HierarchicalClassificationLoss(Module):
    def __init__(self,
                 parent_count,
                 l1_weight=1,
                 l2_weight=1,
                 consistency_weight=1,
                 focal_gamma=2.0,  # Focal loss parameter
                 child_to_parent_mapping = None
                ):
        super().__init__()
        self.parent_count = parent_count
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.consistency_weight= consistency_weight
        self.focal_gamma = focal_gamma
        self.child_to_parent_mapping = child_to_parent_mapping
        
        
    def focal_cross_entropy(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.focal_gamma * ce_loss).mean()
        return focal_loss
        
    def forward(self, inputs, targets):
        l1_logits = inputs[:,:self.parent_count]
        l2_logits = inputs[:,self.parent_count:]
        
        labels_l1 = targets[:,:self.parent_count].argmax(dim=-1)
        labels_l2 = targets[:,self.parent_count:].argmax(dim=-1)
        
        # # Use focal loss for better handling of class imbalance
        l1_loss = self.focal_cross_entropy(l1_logits, labels_l1)
        l2_loss = self.focal_cross_entropy(l2_logits, labels_l2)
        
        # Add consistency loss
        parent_probs = F.softmax(l1_logits, dim=1) # (bs, parent_count)
        child_probs = F.softmax(l2_logits, dim=1) # (bs, children_count)
        
        # child_to_parent_mapping: tensor of shape [num_children] containing parent class indices
        grouped_child_probs = torch.zeros_like(parent_probs) # (bs, parent_count)
        for parent_idx in range(self.parent_count):
            child_indices = (self.child_to_parent_mapping == parent_idx)
            # sum child prob, for each parent
            grouped_child_probs[:, parent_idx] = child_probs[:, child_indices].sum(dim=1)
        
        consistency_loss = F.kl_div(parent_probs.log(), grouped_child_probs, reduction='batchmean')
        
        total_loss = self.l1_weight * l1_loss + self.l2_weight * l2_loss + self.consistency_weight*consistency_loss
        return total_loss

def precision_recall_f1_func_group(parent_count,is_parent,mtype="f1"):
    def _metric(y_true,y_pred):
        # y_pred: (bs,l1+l2), raw logits
        # y_true: (bs,l1+l2), 0s and 2 1s
        # print(y_true)
        # print(y_pred)
        if is_parent==True:
            prob = np.argmax(y_pred[:, :parent_count],axis=-1)
            label = np.argmax(y_true[:,:parent_count],axis=-1)
        else:
            prob = np.argmax(y_pred[:, parent_count:],axis=-1)
            label = np.argmax(y_true[:,parent_count:],axis=-1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(label,prob,average='macro')
        if mtype=='precision':
            return round(precision,4)
        elif mtype=='recall':
            return round(recall,4)
        else:
            return round(f1,4)
    _metric.__name__ = f"{mtype}_{'parent' if is_parent==True else 'children'}"
    return _metric

def get_precision_recall_f1_metrics_group(parent_labels):
    metrics = []
    metrics.append(AccumMetric(precision_recall_f1_func_group(len(parent_labels),True,'f1'),
                                   dim_argmax=None,
                                   activation='no',
                                   flatten=False,
                                   to_np=True,invert_arg=True))
    metrics.append(AccumMetric(precision_recall_f1_func_group(len(parent_labels),False,'f1'),
                                   dim_argmax=None,
                                   activation='no',
                                   flatten=False,
                                   to_np=True,invert_arg=True))
    return metrics

def precision_recall_f1_func_dhc(label_name,label_idx,parent_count,is_parent,mtype="f1"):
    def _metric(y_true,y_pred):
        # y_pred: (bs,l1+l2), raw logits
        # y_true: (bs,2)
        if is_parent==True:
            prob = np.argmax(y_pred[:, :parent_count],axis=-1)
            label = np.argmax(y_true[:,:parent_count],axis=-1)
        else:
            prob = np.argmax(y_pred[:, parent_count:],axis=-1)
            label = np.argmax(y_true[:,parent_count:],axis=-1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(label,prob,average=None)
        if mtype=='precision':
            return round(precision[label_idx],4)
        elif mtype=='recall':
            return round(recall[label_idx],4)
        else:
            return round(f1[label_idx],4)
    _metric.__name__ = f'{mtype}_{label_name}'
    return _metric

def get_precision_recall_f1_metrics_dhc(parent_labels,children_labels,mtype="f1"):
    metrics = []
    for i,v in enumerate(parent_labels):
        metrics.append(AccumMetric(precision_recall_f1_func_dhc(v,i,len(parent_labels),True,mtype),
                                   dim_argmax=None,
                                   activation='no',
                                   flatten=False,
                                   to_np=True,invert_arg=True))
    for i,v in enumerate(children_labels):
        metrics.append(AccumMetric(precision_recall_f1_func_dhc(v,i,len(parent_labels),False,mtype),
                                   dim_argmax=None,
                                   activation='no',
                                   flatten=False,
                                   to_np=True,invert_arg=True))
    return metrics



class HierarchicalSimpleLinearLayer(nn.Module):
    def __init__(self, hidden_size, parent_count, children_count,dropout_rate=0.3, last_hidden=256):
        super().__init__()
        # Parent classification branch
        self.parent_fc = nn.Linear(hidden_size, parent_count)
        nn.init.constant_(self.parent_fc.bias,0)
        
        # Child classification branch - simplified but effective
        self.l2_block = nn.Sequential(
            nn.Linear(hidden_size + parent_count, last_hidden),
            nn.BatchNorm1d(last_hidden),
            MemoryEfficientSwish(),  # Maintain consistency with EfficientNet
            nn.Dropout(dropout_rate),  # Slightly higher dropout for regularization
            nn.Linear(last_hidden, children_count)
        )
        
        # Proper initialization
        nn.init.kaiming_normal_(self.parent_fc.weight)
        for layer in self.l2_block:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
    
    def forward(self, x):
        # Parent predictions
        parent_logits = self.parent_fc(x)
        
        with torch.no_grad():
            parent_probs = F.softmax(parent_logits, dim=1)
            
        # Child predictions with detached parent information
        # Detach to prevent child errors from destabilizing parent training
        l2_input = torch.cat([x, parent_probs.detach()], dim=1)
        child_logits = self.l2_block(l2_input)
        
        return torch.cat([parent_logits, child_logits], dim=1)
    
class HierarchicalLinearLayer(nn.Module):
    def __init__(self, hidden_size, parent_count, children_count,dropout_rate=0.3, last_hidden=256):
        super().__init__()
        # Parent classification branch
        self.parent_fc = nn.Linear(hidden_size, parent_count)
        nn.init.constant_(self.parent_fc.bias,0)
        
        # Child classification branch - simplified but effective
        self.l2_block = nn.Sequential(
            nn.Linear(hidden_size, last_hidden),  # Process features independently first
            nn.BatchNorm1d(last_hidden),
            MemoryEfficientSwish(),
            nn.Dropout(dropout_rate),
            nn.Linear(last_hidden + parent_count, last_hidden),  # Then combine with parent info
            nn.BatchNorm1d(last_hidden),
            MemoryEfficientSwish(),
            nn.Dropout(dropout_rate),
            nn.Linear(last_hidden, children_count)
        )
        
        # Proper initialization
        nn.init.kaiming_normal_(self.parent_fc.weight)
        for layer in self.l2_block:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
    
    def forward(self, x):
        # Parent predictions
        parent_logits = self.parent_fc(x)
        
        with torch.no_grad():
            parent_probs = F.softmax(parent_logits, dim=1)
            

        # Child predictions with better feature processing
        l2_features = self.l2_block[0:4](x)  # Process features first
        l2_input = torch.cat([l2_features, parent_probs.detach()], dim=1)  # Use probabilities instead of logits
        child_logits = self.l2_block[4:](l2_input)
        
        return torch.cat([parent_logits, child_logits], dim=1)

class HierarchicalEfficientNet(EfficientNet):
    def __init__(self, parent_count,children_count,lin_dropout_rate=0.3, last_hidden=256, use_simple_head=True,blocks_args=None, global_params=None):
        super().__init__(blocks_args, global_params)

        print('Image size for HierarchicalEfficientNet: ',self._global_params.image_size)
        # Remove the existing final linear layer
        if self._global_params.include_top:
            del self._fc
        out_channels = round_filters(1280, self._global_params)
        if use_simple_head:
            self._hierarchical_fc = HierarchicalSimpleLinearLayer(out_channels, parent_count,children_count,lin_dropout_rate,last_hidden)
        else:
            self._hierarchical_fc = HierarchicalLinearLayer(out_channels, parent_count,children_count,lin_dropout_rate,last_hidden)

    def forward(self, inputs):
        """EfficientNet's forward function with custom hierarchical final layer.
           Calls extract_features to extract features, applies custom hierarchical layer, and returns logits.

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Output of this model after processing.
        """
        # Convolution layers
        x = self.extract_features(inputs)
        # Pooling and custom hierarchical layer
        x = self._avg_pooling(x)
        if self._global_params.include_top:
            x = x.flatten(start_dim=1)
            x = self._dropout(x)
            x = self._hierarchical_fc(x)
        return x
    
def hierarchical_param_splitter(m,parent_last=True):
    # rough draft of differential learning rate for EfficientNet
    # not really affective for wildlife dataset
    len_hierar_fc = len(list(m._hierarchical_fc.parameters()))
    group1 = [p for p in list(m.parameters())[:-len_hierar_fc] if p.requires_grad]
    parent_group = [p for p in m._hierarchical_fc.parent_fc.parameters() if p.requires_grad]
    children_group = [p for p in m._hierarchical_fc.l2_block.parameters() if p.requires_grad]
    if parent_last:
        return [group1,children_group,parent_group]
    return [group1,parent_group,children_group]


def load_hier_model(parent_count,children_count,lin_dropout_rate=0.3, last_hidden=256, use_simple_head=True,
                    base_model='efficientnet-b3',trained_weight_path=None,image_size=None):
                    
    from efficientnet_pytorch.utils import efficientnet, efficientnet_params,load_pretrained_weights
    w, d, s, p = efficientnet_params(base_model)
    s = image_size if image_size is not None else s
    
    blocks_args, global_params = efficientnet(include_top=True,
                                              width_coefficient=w, 
                                              depth_coefficient=d, 
                                              dropout_rate=p, # 0.3
                                              image_size=s,
                                              num_classes=parent_count+children_count,
                                             )


    hier_model = HierarchicalEfficientNet(parent_count,children_count,
                                                lin_dropout_rate, 
                                                last_hidden, 
                                                use_simple_head,
                                                blocks_args,global_params)
    
    if trained_weight_path is None:
        try:
            load_pretrained_weights(hier_model, model_name = base_model, weights_path=None,
                                        load_fc=False, advprop=False)
        except:
            pass
    else:
        state_dict = torch.load(trained_weight_path)
        ret = hier_model.load_state_dict(state_dict, strict=False)
        if len(ret.missing_keys):
            print(f'Missing keys: {ret.missing_keys}')
    
    return hier_model
    