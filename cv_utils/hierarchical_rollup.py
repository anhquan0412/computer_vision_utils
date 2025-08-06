import torch
from typing import List, Dict, Any

# --- 1. Dynamic Pre-computation Helper ---

def precompute_rollup_maps_dynamic(
    all_level_labels: List[List[str]],
    hierarchy_maps: List[Dict[str, List[str]]],
) -> List[Dict[int, List[int]]]:
    """
    Pre-computes aggregation maps for a dynamic N-level hierarchy.
    """
    num_levels = len(all_level_labels)
    most_specific_level_idx = num_levels - 1
    most_specific_labels = all_level_labels[most_specific_level_idx]

    # Create name-to-index lookups for all levels for convenience
    all_name_to_idx_maps = [
        {name: i for i, name in enumerate(level_labels)}
        for level_labels in all_level_labels
    ]

    # Create bottom-up child-to-parent index maps
    child_to_parent_maps = []
    # This loop defines the relationship between level i (parent) and i+1 (child)
    for i in range(num_levels - 1):
        parent_level_idx = i
        child_level_idx = i + 1

        child_to_parent_map = {}
        parent_name_map = hierarchy_maps[parent_level_idx]
        
        # Corrected lines: removed the erroneous "+ 1"
        parent_name_to_idx = all_name_to_idx_maps[parent_level_idx]
        child_name_to_idx = all_name_to_idx_maps[child_level_idx]

        for parent_name, child_names in parent_name_map.items():
            parent_idx = parent_name_to_idx[parent_name]
            for child_name in child_names:
                child_idx = child_name_to_idx[child_name]
                child_to_parent_map[child_idx] = parent_idx
        child_to_parent_maps.append(child_to_parent_map)

    # Create final aggregation maps (from most specific up to each level)
    aggregation_maps = [{} for _ in range(num_levels)]
    
    aggregation_maps[most_specific_level_idx] = {
        i: [i] for i in range(len(most_specific_labels))
    }

    for level_idx in range(most_specific_level_idx - 1, -1, -1):
        agg_map = {}
        lower_level_agg_map = aggregation_maps[level_idx + 1]
        parent_map_for_level = child_to_parent_maps[level_idx]

        for lower_level_idx, specific_indices in lower_level_agg_map.items():
            parent_idx = parent_map_for_level.get(lower_level_idx)
            if parent_idx is not None:
                if parent_idx not in agg_map:
                    agg_map[parent_idx] = []
                agg_map[parent_idx].extend(specific_indices)
        aggregation_maps[level_idx] = agg_map
        
    return aggregation_maps


# --- 2. Dynamic Main Rollup Function ---

def rollup_predictions_dynamic(
    softmax_most_specific: torch.Tensor,
    all_level_labels: List[List[str]],
    aggregation_maps: List[Dict[int, List[int]]],
    threshold: float,
) -> List[Dict[str, Any]]:
    """
    Performs hierarchical label rollup for a dynamic N-level hierarchy.
    """
    num_images = softmax_most_specific.shape[0]
    num_levels = len(all_level_labels)
    device = softmax_most_specific.device

    # Calculate probabilities and best predictions for ALL levels
    all_level_probs = []
    best_probs_per_level = []
    best_indices_per_level = []

    for i in range(num_levels):
        level_labels = all_level_labels[i]
        agg_map = aggregation_maps[i]
        
        level_probs = torch.zeros((num_images, len(level_labels)), device=device)
        for label_idx, specific_indices in agg_map.items():
            if specific_indices:
                level_probs[:, label_idx] = torch.sum(softmax_most_specific[:, specific_indices], dim=1)
        
        best_probs, best_indices = torch.max(level_probs, dim=1)
        all_level_probs.append(level_probs)
        best_probs_per_level.append(best_probs)
        best_indices_per_level.append(best_indices)

    # Sequentially check thresholds from specific to general
    output = [{} for _ in range(num_images)]
    decided_mask = torch.zeros(num_images, dtype=torch.bool, device=device)

    for i in range(num_levels - 1, -1, -1): # From N-1 down to 0
        pass_mask = (best_probs_per_level[i] >= threshold) & (~decided_mask)
        decided_mask |= pass_mask
        
        for idx in torch.where(pass_mask)[0]:
            best_label_idx = best_indices_per_level[i][idx]
            output[idx] = {
                'prediction': all_level_labels[i][best_label_idx],
                'probability': best_probs_per_level[i][idx].item(),
                'level': i + 1,
                'passes_threshold': True
            }

    # Handle images that never passed the threshold
    undecided_indices = torch.where(~decided_mask)[0]
    if len(undecided_indices) > 0:
        all_best_probs_stacked = torch.stack(best_probs_per_level, dim=1)
        abs_best_probs, abs_best_level_indices = torch.max(all_best_probs_stacked[undecided_indices], dim=1)

        for i, original_idx in enumerate(undecided_indices):
            best_level = abs_best_level_indices[i].item()
            best_label_idx = best_indices_per_level[best_level][original_idx]
            output[original_idx] = {
                'prediction': all_level_labels[best_level][best_label_idx],
                'probability': abs_best_probs[i].item(),
                'level': best_level + 1,
                'passes_threshold': False
            }
            
    return output