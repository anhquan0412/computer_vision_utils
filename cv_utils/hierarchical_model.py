from fastai.vision.all import *
import timm
from sklearn.metrics import precision_recall_fscore_support


class MemoryEfficientSwish(torch.autograd.Function):
    """
    Memory efficient implementation of Swish activation function.
    Swish(x) = x * sigmoid(x)
    This is equivalent to SiLU in modern PyTorch.
    """
    @staticmethod
    def forward(ctx, x):
        result = x * torch.sigmoid(x)
        ctx.save_for_backward(x)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        sigmoid_x = torch.sigmoid(x)
        return grad_output * (sigmoid_x * (1 + x * (1 - sigmoid_x)))


class MemoryEfficientSwishModule(nn.Module):
    """Module wrapper for MemoryEfficientSwish"""
    def forward(self, x):
        return MemoryEfficientSwish.apply(x)


def round_filters(filters, global_params=None):
    """
    Calculate and round number of filters based on width multiplier.
    
    Args:
        filters (int): Base number of filters
        global_params: Width multiplier (float) or parameter object
    
    Returns:
        int: Rounded number of filters
    """
    # Handle different input types for backward compatibility
    if global_params is None:
        return filters
    
    # If global_params is actually a float (width_multiplier)
    if isinstance(global_params, (int, float)):
        width_multiplier = global_params
        depth_divisor = 8
        min_depth = None
    else:
        # If it's a global_params object (backward compatibility)
        width_multiplier = getattr(global_params, 'width_coefficient', 1.0)
        depth_divisor = getattr(global_params, 'depth_divisor', 8)
        min_depth = getattr(global_params, 'min_depth', None)
    
    if not width_multiplier:
        return filters
    
    filters *= width_multiplier
    min_depth = min_depth or depth_divisor
    
    # Round to nearest multiple of depth_divisor
    new_filters = max(min_depth, int(filters + depth_divisor / 2) // depth_divisor * depth_divisor)
    
    # Prevent rounding by more than 10%
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
        
    return int(new_filters)


def get_activation_layer(name='swish'):
    """
    Get activation layer by name. Returns modern PyTorch implementations.
    
    Args:
        name (str): Activation name ('swish', 'silu', 'relu', 'gelu', etc.)
    
    Returns:
        nn.Module: Activation layer
    """
    name = name.lower()
    
    if name in ['swish', 'silu']:
        # Use modern PyTorch SiLU (equivalent to Swish)
        return nn.SiLU()
    elif name == 'memory_efficient_swish':
        # Use our custom memory efficient implementation
        return MemoryEfficientSwishModule()
    elif name == 'relu':
        return nn.ReLU()
    elif name == 'gelu':
        return nn.GELU()
    elif name == 'mish':
        return nn.Mish()
    else:
        raise ValueError(f"Unknown activation: {name}")

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
        # y_true: (bs,l1+l2), 0s and 2 1s
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
            nn.SiLU(),  # Modern PyTorch Swish equivalent (more efficient than custom implementation)
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
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(last_hidden + parent_count, last_hidden),  # Then combine with parent info
            nn.BatchNorm1d(last_hidden),
            nn.SiLU(),
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


class HierarchicalTimmEfficientNet(nn.Module):
    """
    Hierarchical model using timm EfficientNet as backbone
    """
    def __init__(self, parent_count, children_count, lin_dropout_rate=0.3, 
                 last_hidden=256, use_simple_head=True, base_model='tf_efficientnet_b5.ns_jft_in1k'):
        super().__init__()
        
        # Create timm model as feature extractor (without classifier head)
        self.backbone = timm.create_model(base_model, pretrained=True, num_classes=0)  # num_classes=0 removes classifier
        
        # Get feature dimension from the model (much cleaner than forward pass)
        # This gives the number of features after global average pooling
        out_channels = self.backbone.num_features
        
        # Add hierarchical classifier head
        if use_simple_head:
            self._hierarchical_fc = HierarchicalSimpleLinearLayer(
                out_channels, parent_count, children_count, lin_dropout_rate, last_hidden)
        else:
            self._hierarchical_fc = HierarchicalLinearLayer(
                out_channels, parent_count, children_count, lin_dropout_rate, last_hidden)
        
        print(f'Created HierarchicalTimmEfficientNet with {base_model} backbone')
        print(f'Feature dimension: {out_channels}, Parent classes: {parent_count}, Child classes: {children_count}')
    
    def forward(self, x):
        # Extract features using timm backbone
        features = self.backbone(x)  # Output: [batch_size, feature_dim]
        
        # Apply hierarchical classifier
        logits = self._hierarchical_fc(features)
        return logits


def load_hier_model_timm(parent_count, children_count, lin_dropout_rate=0.3, 
                        last_hidden=256, use_simple_head=True, base_model='tf_efficientnet_b5.ns_jft_in1k',
                        trained_weight_path=None):
    """
    Load hierarchical model using timm backend
    """
    # Convert model name format for timm
    timm_model_name = base_model.replace('-', '_')
    
    # Create hierarchical model
    hier_model = HierarchicalTimmEfficientNet(
        parent_count=parent_count,
        children_count=children_count,
        lin_dropout_rate=lin_dropout_rate,
        last_hidden=last_hidden,
        use_simple_head=use_simple_head,
        base_model=timm_model_name
    )
    
    # Load trained weights if provided
    if trained_weight_path is not None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(trained_weight_path, map_location=device)
        ret = hier_model.load_state_dict(state_dict, strict=False)
        if len(ret.missing_keys):
            print(f'Missing keys: {ret.missing_keys}')
        if len(ret.unexpected_keys):
            print(f'Unexpected keys: {ret.unexpected_keys}')
    
    return hier_model
    