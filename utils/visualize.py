import numpy as np
import torch
import yaml

def get_color_map(dataset_config, return_torch_tenor=False, return_numpy_ndarray=False):
    with open(dataset_config, 'r') as f:
        config = yaml.safe_load(f)
    color_map = dict()
    for i in sorted(list(config['learning_map'].keys()))[::-1]:
        color_map[config['learning_map'][i]] = config['color_map'][i][::-1] # bgr -> rgb

    if return_torch_tenor:
        color_map_tensor = torch.zeros(len(color_map), 3)
        for i in sorted(list(color_map.keys())):
            color_map_tensor[i] = torch.tensor(color_map[i])
        return color_map_tensor
    
    if return_numpy_ndarray:
        color_map_np = np.zeros((len(color_map), 3))
        for i in sorted(list(color_map.keys())):
            color_map_np[i] = np.array(color_map[i])
        return color_map_np

    return color_map


def get_color(labels, dataset_config):
    COLOR_MAP = get_color_map(dataset_config, return_numpy_ndarray=True)
    if labels.ndim == 2:
        labels = labels.squeeze(1)
    colors = np.take(COLOR_MAP, labels, axis=0).astype(np.uint8)
    return colors

def get_color_cluster(labels):
    if np.unique(labels).min() == 0:
        cluster_num = len(np.unique(labels))
        COLOR_MAP = np.random.randint(0, 256, size=(cluster_num, 3), dtype=int) 
        colors = np.take(COLOR_MAP, labels, axis=0).astype(np.uint8)
    else:
        cluster_num = np.unique(labels).max() + 1
        COLOR_MAP = np.random.randint(0, 256, size=(cluster_num, 3), dtype=int) 
        colors = np.take(COLOR_MAP, labels, axis=0).astype(np.uint8)
            
    return colors