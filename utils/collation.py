import MinkowskiEngine as ME
import torch
import numpy as np

def collation_fn_dataset(data_labels):
    coords, feats, labels, selected_idx, pcd_indexes = list(zip(*data_labels))

    # Create batched coordinates for the SparseTensor input
    bcoords = ME.utils.batched_coordinates(coords)

    # Concatenate all lists
    feats = torch.from_numpy(np.concatenate(feats, 0)).float()
    labels = torch.from_numpy(np.concatenate(labels, 0)).int()
    selected_idx = torch.from_numpy(np.concatenate(selected_idx, 0)).long()
    pcd_indexes = torch.tensor(pcd_indexes, dtype=torch.int16)

    return bcoords, feats, labels, selected_idx, pcd_indexes

def collation_fn_restricted_unlab_dataset(data_labels):
    coords, feats, selected_idx, inverse_maps, pcd_indexes = list(zip(*data_labels))
    
    bcoords = ME.utils.batched_coordinates(coords)
    feats = torch.from_numpy(np.concatenate(feats, 0)).float()
    selected_idx = torch.from_numpy(np.concatenate(selected_idx, 0)).long()
    pcd_indexes = torch.tensor(pcd_indexes, dtype=torch.int16)
    
    return bcoords, feats, selected_idx, inverse_maps, pcd_indexes

def collation_fn_restricted_dataset(data_labels):
    coords, feats, labels, selected_idx, mapped_labels, inverse_maps, pcd_indexes = list(zip(*data_labels))

    # Create batched coordinates for the SparseTensor input
    bcoords = ME.utils.batched_coordinates(coords)

    # Concatenate all lists
    feats = torch.from_numpy(np.concatenate(feats, 0)).float()
    labels = torch.from_numpy(np.concatenate(labels, 0)).int()
    selected_idx = torch.from_numpy(np.concatenate(selected_idx, 0)).long()
    mapped_labels = torch.from_numpy(np.concatenate(mapped_labels, 0)).int()
    pcd_indexes = torch.tensor(pcd_indexes, dtype=torch.int16)

    return bcoords, feats, labels, selected_idx, mapped_labels, inverse_maps, pcd_indexes

def collation_fn_restricted_dataset_two_samples(data_labels):
    coords, feats, labels, selected_idx, mapped_labels, inverse_maps, coords1, feats1, labels1, selected_idx1, mapped_labels1, inverse_maps1, pcd_indexes = list(zip(*data_labels))
    # Create batched coordinates for the SparseTensor input
    bcoords = ME.utils.batched_coordinates(coords)
    bcoords1 = ME.utils.batched_coordinates(coords1)

    # Concatenate all lists
    feats = torch.from_numpy(np.concatenate(feats, 0)).float()
    labels = torch.from_numpy(np.concatenate(labels, 0)).int()
    selected_idx = torch.from_numpy(np.concatenate(selected_idx, 0)).long()
    mapped_labels = torch.from_numpy(np.concatenate(mapped_labels, 0)).int()
    feats1 = torch.from_numpy(np.concatenate(feats1, 0)).float()
    labels1 = torch.from_numpy(np.concatenate(labels1, 0)).int()
    selected_idx1 = torch.from_numpy(np.concatenate(selected_idx1, 0)).long()
    mapped_labels1 = torch.from_numpy(np.concatenate(mapped_labels1, 0)).int()
    pcd_indexes = torch.tensor(pcd_indexes, dtype=torch.int16)

    return bcoords, feats, labels, selected_idx, mapped_labels, inverse_maps, bcoords1, feats1, labels1, selected_idx1, mapped_labels1, inverse_maps1, pcd_indexes

def collation_fn_restricted_dataset_two_samples_ssl(data_labels):
    coords, feats, labels, selected_idx, mapped_labels, inverse_maps, \
    coords1, feats1, labels1, selected_idx1, mapped_labels1, inverse_maps1, pcd_indexes, \
    uncoords, unfeats, unlabels, unselected_idx, unmapped_labels, uninverse_maps,\
    uncoords1, unfeats1, unlabels1, unselected_idx1, unmapped_labels1, uninverse_maps1, unpcd_indexes= list(zip(*data_labels))

    bcoords = ME.utils.batched_coordinates(coords)
    bcoords1 = ME.utils.batched_coordinates(coords1)
    unbcoords = ME.utils.batched_coordinates(uncoords)
    unbcoords1 = ME.utils.batched_coordinates(uncoords1)

    feats = torch.from_numpy(np.concatenate(feats, 0)).float()
    labels = torch.from_numpy(np.concatenate(labels, 0)).int()
    selected_idx = torch.from_numpy(np.concatenate(selected_idx, 0)).long()
    mapped_labels = torch.from_numpy(np.concatenate(mapped_labels, 0)).int()
    feats1 = torch.from_numpy(np.concatenate(feats1, 0)).float()
    labels1 = torch.from_numpy(np.concatenate(labels1, 0)).int()
    selected_idx1 = torch.from_numpy(np.concatenate(selected_idx1, 0)).long()
    mapped_labels1 = torch.from_numpy(np.concatenate(mapped_labels1, 0)).int()
    pcd_indexes = torch.tensor(pcd_indexes, dtype=torch.int16)

    unfeats = torch.from_numpy(np.concatenate(unfeats, 0)).float()
    unlabels = torch.from_numpy(np.concatenate(unlabels, 0)).int()
    unselected_idx = torch.from_numpy(np.concatenate(unselected_idx, 0)).long()
    unmapped_labels = torch.from_numpy(np.concatenate(unmapped_labels, 0)).int()
    unfeats1 = torch.from_numpy(np.concatenate(unfeats1, 0)).float()
    unlabels1 = torch.from_numpy(np.concatenate(unlabels1, 0)).int()
    unselected_idx1 = torch.from_numpy(np.concatenate(unselected_idx1, 0)).long()
    unmapped_labels1 = torch.from_numpy(np.concatenate(unmapped_labels1, 0)).int()
    unpcd_indexes = torch.tensor(unpcd_indexes, dtype=torch.int16)

    return {'sup': (bcoords, feats, labels, selected_idx, mapped_labels, inverse_maps, bcoords1, feats1, labels1, selected_idx1, mapped_labels1, inverse_maps1, pcd_indexes), 
            'unsup': (unbcoords, unfeats, unlabels, unselected_idx, unmapped_labels, uninverse_maps, unbcoords1, unfeats1, unlabels1, unselected_idx1, unmapped_labels1, uninverse_maps1, unpcd_indexes)}

def collation_fn_restricted_dataset_two_samples_dual_ssl(data_labels):
    coords, feats, labels, selected_idx, mapped_labels, superpoint_labels, inverse_maps, \
    coords1, feats1, labels1, selected_idx1, mapped_labels1, superpoint_labels1, inverse_maps1, pcd_indexes, \
    uncoords, unfeats, unlabels, unselected_idx, unmapped_labels, uninverse_maps,\
    uncoords1, unfeats1, unlabels1, unselected_idx1, unmapped_labels1, uninverse_maps1, unpcd_indexes= list(zip(*data_labels))

    bcoords = ME.utils.batched_coordinates(coords)
    bcoords1 = ME.utils.batched_coordinates(coords1)
    unbcoords = ME.utils.batched_coordinates(uncoords)
    unbcoords1 = ME.utils.batched_coordinates(uncoords1)

    feats = torch.from_numpy(np.concatenate(feats, 0)).float()
    labels = torch.from_numpy(np.concatenate(labels, 0)).int()
    selected_idx = torch.from_numpy(np.concatenate(selected_idx, 0)).long()
    mapped_labels = torch.from_numpy(np.concatenate(mapped_labels, 0)).int()
    superpoint_labels = torch.from_numpy(np.concatenate(superpoint_labels, 0)).int()
    feats1 = torch.from_numpy(np.concatenate(feats1, 0)).float()
    labels1 = torch.from_numpy(np.concatenate(labels1, 0)).int()
    selected_idx1 = torch.from_numpy(np.concatenate(selected_idx1, 0)).long()
    mapped_labels1 = torch.from_numpy(np.concatenate(mapped_labels1, 0)).int()
    superpoint_labels1 = torch.from_numpy(np.concatenate(superpoint_labels1, 0)).int()
    pcd_indexes = torch.tensor(pcd_indexes, dtype=torch.int16)

    unfeats = torch.from_numpy(np.concatenate(unfeats, 0)).float()
    unlabels = torch.from_numpy(np.concatenate(unlabels, 0)).int()
    unselected_idx = torch.from_numpy(np.concatenate(unselected_idx, 0)).long()
    unmapped_labels = torch.from_numpy(np.concatenate(unmapped_labels, 0)).int()
    unfeats1 = torch.from_numpy(np.concatenate(unfeats1, 0)).float()
    unlabels1 = torch.from_numpy(np.concatenate(unlabels1, 0)).int()
    unselected_idx1 = torch.from_numpy(np.concatenate(unselected_idx1, 0)).long()
    unmapped_labels1 = torch.from_numpy(np.concatenate(unmapped_labels1, 0)).int()
    unpcd_indexes = torch.tensor(unpcd_indexes, dtype=torch.int16)

    return {'sup': (bcoords, feats, labels, selected_idx, mapped_labels, superpoint_labels, inverse_maps, bcoords1, feats1, labels1, selected_idx1, mapped_labels1, superpoint_labels1,  inverse_maps1, pcd_indexes), 
            'unsup': (unbcoords, unfeats, unlabels, unselected_idx, unmapped_labels, uninverse_maps, unbcoords1, unfeats1, unlabels1, unselected_idx1, unmapped_labels1, uninverse_maps1, unpcd_indexes)}


def collation_fn_ssl_dataset(data_labels):
    sup_indices = list(range(0, int(len(data_labels)/2)))
    unsup_indices = list(range(int(len(data_labels)/2), len(data_labels)))
    
    sup_data = [data_labels[i] for i in sup_indices]
    unsup_data = [data_labels[i] for i in unsup_indices]
    
    batch_data = {
        'sup': {},
        'unsup': {},
    }
    
    coords_list ,coords_list1 = [], []
    feats_list, feats_list1 = [], []
    labels_list, labels_list1= [], []
    selected_idx_list, selected_idx_list1 = [], []
    inverse_maps_list, inverse_maps_list1 = [], []
    mapped_labels_list, mapped_labels_list1 = [], []
    t_list, t_list1 = [], []

    for sup_datum in sup_data:
        coords, feats, labels, selected_idx, mapped_labels, inverse_maps, pcd_indexes = sup_datum
        # Append to lists for the batched coordinates for the SparseTensor input
        # batch_data['sup']['coords'].append(coords)
        # batch_data['sup']['feats'].append(feats)
        # batch_data['sup']['labels'].append(labels)
        # batch_data['sup']['selected_idx'].append(selected_idx)
        # batch_data['sup']['mapped_labels'].append(mapped_labels)
        # batch_data['sup']['inverse_maps'].append(inverse_maps)
        # batch_data['sup']['pcd_indexes'].append(torch.tensor(pcd_indexes, dtype=torch.int16))
        coords_list.append(coords)
        feats_list.append(feats)
        labels_list.append(labels)
        selected_idx_list.append(selected_idx)
        mapped_labels_list.append(mapped_labels)
        inverse_maps_list.append(inverse_maps)
        t_list.append(pcd_indexes)
        

    for unsup_datum in unsup_data:
        coords, feats, labels, selected_idx, mapped_labels, inverse_maps, pcd_indexes = unsup_datum
        # batch_data['unsup']['coords'].append(coords)
        # batch_data['unsup']['feats'].append(feats)
        # batch_data['unsup']['selected_idx'].append(selected_idx)
        # batch_data['unsup']['inverse_maps'].append(inverse_maps)
        # batch_data['unsup']['pcd_indexes'].append(torch.tensor(pcd_indexes, dtype=torch.int16))
        coords_list1.append(coords)
        feats_list1.append(feats)
        labels_list1.append(labels)
        mapped_labels_list1.append(mapped_labels)
        selected_idx_list1.append(selected_idx)
        inverse_maps_list1.append(inverse_maps)
        t_list1.append(pcd_indexes)
    # Convert lists to tensors or concatenate them as required before returning
    # For example, if coords should be a single tensor, you can concatenate them here
    
    coords_list = ME.utils.batched_coordinates(coords_list)
    coords_list1 = ME.utils.batched_coordinates(coords_list1)
    
    feats_list = torch.from_numpy(np.concatenate(feats_list, 0)).float()
    feats_list1 = torch.from_numpy(np.concatenate(feats_list1, 0)).float()
    
    labels_list = torch.from_numpy(np.concatenate(labels_list, 0)).int()
    labels_list1 = torch.from_numpy(np.concatenate(labels_list1, 0)).int()
    
    mapped_labels_list = torch.from_numpy(np.concatenate(mapped_labels_list, 0)).int()
    mapped_labels_list1 = torch.from_numpy(np.concatenate(mapped_labels_list1, 0)).int()
    
    selected_idx_list = [torch.from_numpy(idx).to(torch.int32) for idx in selected_idx_list]
    selected_idx_list1 = [torch.from_numpy(idx).to(torch.int32) for idx in selected_idx_list1]
    
    t_list = [torch.tensor(t, dtype=torch.int16) for t in t_list]
    t_list1 = [torch.tensor(t, dtype=torch.int16) for t in t_list1]
    
    batch_data['sup']['coords'] = coords_list
    batch_data['unsup']['coords'] = coords_list1
    
    batch_data['sup']['feats'] = feats_list
    batch_data['unsup']['feats'] = feats_list1
    
    batch_data['sup']['labels'] = labels_list
    batch_data['unsup']['labels'] = labels_list1
    
    batch_data['sup']['mapped_label'] = mapped_labels_list
    batch_data['unsup']['mapped_label'] = mapped_labels_list1

    batch_data['sup']['inverse_map'] = inverse_maps_list
    batch_data['unsup']['inverse_map'] = inverse_maps_list1
    
    batch_data['sup']['selected_idx'] = selected_idx_list
    batch_data['unsup']['selected_idx'] = selected_idx_list1
    
    batch_data['sup']['pcd_idx'] = t_list
    batch_data['unsup']['pcd_idx'] = t_list1
     
    return batch_data
                        
def collation_fn_full_sup_dataset(data_labels):
    sup_indices = list(range(0, int(len(data_labels))))
    sup_data = [data_labels[i] for i in sup_indices]
    
    batch_data = {
        'sup': {},
    }
    
    coords_list = []
    feats_list= []
    labels_list= []
    selected_idx_list= []
    mapped_labels_list = []
    t_list= []

    for sup_datum in sup_data:
        coords, feats, labels, selected_idx, mapped_labels, inverse_maps, pcd_indexes = sup_datum
        
        # Append to lists for the batched coordinates for the SparseTensor input
        # batch_data['sup']['coords'].append(coords)
        # batch_data['sup']['feats'].append(feats)
        # batch_data['sup']['labels'].append(labels)
        # batch_data['sup']['selected_idx'].append(selected_idx)
        # batch_data['sup']['mapped_labels'].append(mapped_labels)
        # batch_data['sup']['inverse_maps'].append(inverse_maps)
        # batch_data['sup']['pcd_indexes'].append(torch.tensor(pcd_indexes, dtype=torch.int16))
        coords_list.append(coords)
        feats_list.append(feats)
        labels_list.append(labels)
        selected_idx_list.append(selected_idx)
        mapped_labels_list.append(mapped_labels)
        # inverse_map_list.append(inverse_maps)
        t_list.append(pcd_indexes)
        
        
    coords_list = ME.utils.batched_coordinates(coords_list)
    feats_list = torch.from_numpy(np.concatenate(feats_list, 0)).float()
    labels_list = torch.from_numpy(np.concatenate(labels_list, 0)).int()
    mapped_labels_list = torch.from_numpy(np.concatenate(mapped_labels_list, 0)).int()
    selected_idx_list = [torch.from_numpy(idx).to(torch.int32) for idx in selected_idx_list]
    t_list = [torch.tensor(t, dtype=torch.int16) for t in t_list]
   
    
    batch_data['sup']['coords'] = coords_list
    batch_data['sup']['feats'] = feats_list
    batch_data['sup']['mapped_label'] = mapped_labels_list
    batch_data['sup']['labels'] = labels_list
    batch_data['sup']['selected_idx'] = selected_idx_list
    batch_data['sup']['pcd_idx'] = t_list
     
    return batch_data   

def collation_fn_lasermix(data_labels):
    sup_indices = list(range(0, int(len(data_labels)/2)))
    unsup_indices = list(range(int(len(data_labels)/2), len(data_labels)))
    
    sup_data = [data_labels[i] for i in sup_indices]
    unsup_data = [data_labels[i] for i in unsup_indices]
    
    batch_data = {
        'sup': {},
        'unsup': {},
    }
    
    coords_list ,coords_list1 = [], []
    feats_list, feats_list1 = [], []
    labels_list= []
    selected_idx_list, selected_idx_list1 = [], []
    mapped_labels_list = []
    t_list, t_list1 = [], []

    for sup_datum in sup_data:
        coords, feats, labels, selected_idx, mapped_labels, inverse_maps, pcd_indexes = sup_datum
        # Append to lists for the batched coordinates for the SparseTensor input
        # batch_data['sup']['coords'].append(coords)
        # batch_data['sup']['feats'].append(feats)
        # batch_data['sup']['labels'].append(labels)
        # batch_data['sup']['selected_idx'].append(selected_idx)
        # batch_data['sup']['mapped_labels'].append(mapped_labels)
        # batch_data['sup']['inverse_maps'].append(inverse_maps)
        # batch_data['sup']['pcd_indexes'].append(torch.tensor(pcd_indexes, dtype=torch.int16))
        coords_list.append(coords)
        feats_list.append(feats)
        labels_list.append(labels)
        selected_idx_list.append(selected_idx)
        mapped_labels_list.append(mapped_labels)
        # inverse_map_list.append(inverse_maps)
        t_list.append(pcd_indexes)
        

    for unsup_datum in unsup_data:
        coords, feats, selected_idx, inverse_maps, pcd_indexes = unsup_datum
        # batch_data['unsup']['coords'].append(coords)
        # batch_data['unsup']['feats'].append(feats)
        # batch_data['unsup']['selected_idx'].append(selected_idx)
        # batch_data['unsup']['inverse_maps'].append(inverse_maps)
        # batch_data['unsup']['pcd_indexes'].append(torch.tensor(pcd_indexes, dtype=torch.int16))
        coords_list1.append(coords)
        feats_list1.append(feats)
        selected_idx_list1.append(selected_idx)
        # inverse_map_list1.append(inverse_maps)
        t_list1.append(pcd_indexes)
    # Convert lists to tensors or concatenate them as required before returning
    # For example, if coords should be a single tensor, you can concatenate them here

    # coords_list = ME.utils.batched_coordinates(coords_list)
    # coords_list1 = ME.utils.batched_coordinates(coords_list1)
    bcoords, bcoords1 = [], []
    for batch_idx, coords in enumerate(coords_list):
        batch_indices = np.full((coords.shape[0],1), batch_idx)
        new_batch = np.hstack((batch_indices, coords))
        bcoords.append(new_batch)
        
    bcoords = np.concatenate(bcoords, 0)
    
    for batch_idx, coords in enumerate(coords_list1):
        batch_indices = np.full((coords.shape[0],1), batch_idx)
        new_batch = np.hstack((batch_indices, coords))
        bcoords1.append(new_batch)
    
    bcoords1 = np.concatenate(bcoords1, 0)
    
    bcoords = torch.Tensor(bcoords)
    bcoords1 = torch.Tensor(bcoords1)
    
    feats_list = torch.from_numpy(np.concatenate(feats_list, 0)).float()
    feats_list1 = torch.from_numpy(np.concatenate(feats_list1, 0)).float()
    
    labels_list = torch.from_numpy(np.concatenate(labels_list, 0)).int()
    mapped_labels_list = torch.from_numpy(np.concatenate(mapped_labels_list, 0)).int()
    
    selected_idx_list = [torch.from_numpy(idx).to(torch.int32) for idx in selected_idx_list]
    selected_idx_list1 = [torch.from_numpy(idx).to(torch.int32) for idx in selected_idx_list1]
    
    t_list = [torch.tensor(t, dtype=torch.int16) for t in t_list]
    t_list1 = [torch.tensor(t, dtype=torch.int16) for t in t_list1]
    
    batch_data['sup']['coords'] = bcoords
    batch_data['unsup']['coords'] = bcoords1
    
    batch_data['sup']['feats'] = feats_list
    batch_data['unsup']['feats'] = feats_list1
    
    batch_data['sup']['labels'] = labels_list
    batch_data['sup']['mapped_label'] = mapped_labels_list
    
    batch_data['sup']['selected_idx'] = selected_idx_list
    batch_data['unsup']['selected_idx'] = selected_idx_list1
    
    batch_data['sup']['pcd_idx'] = t_list
    batch_data['unsup']['pcd_idx'] = t_list1
     
    return batch_data

def collation_fn_polarmix_dataset(data_labels):
    coords, feats, labels, selected_idx, mapped_labels, inverse_maps, coords1, feats1, labels1, selected_idx1, mapped_labels1, inverse_maps1, pcd_indexes = list(zip(*data_labels))
    
    # polarmix
    bcoords = ME.utils.batched_coordinates(coords)
    feats = torch.from_numpy(np.concatenate(feats, 0)).float()
    labels = torch.from_numpy(np.concatenate(labels, 0)).int()
    selected_idx = torch.from_numpy(np.concatenate(selected_idx, 0)).long()
    mapped_labels = torch.from_numpy(np.concatenate(mapped_labels, 0)).int()

    # original
    bcoords1 = ME.utils.batched_coordinates(coords1)
    feats1 = torch.from_numpy(np.concatenate(feats1, 0)).float()
    labels1 = torch.from_numpy(np.concatenate(labels1, 0)).int()
    selected_idx1 = torch.from_numpy(np.concatenate(selected_idx1, 0)).long()
    mapped_labels1 = torch.from_numpy(np.concatenate(mapped_labels1, 0)).int()
    pcd_indexes = torch.tensor(pcd_indexes, dtype=torch.int16)

    batch_data = {
        'polarmix': {},
        'origin': {},
    }

    batch_data['polarmix']['coords'] = bcoords
    batch_data['origin']['coords'] = bcoords1
    
    batch_data['polarmix']['feats'] = feats
    batch_data['origin']['feats'] = feats1
    
    batch_data['polarmix']['labels'] = labels
    batch_data['origin']['labels'] = labels1

    batch_data['polarmix']['mapped_labels'] = mapped_labels
    batch_data['origin']['mapped_labels'] = mapped_labels1

    batch_data['polarmix']['selected_idx'] = selected_idx
    batch_data['origin']['selected_idx'] = selected_idx1

    batch_data['polarmix']['inverse_maps'] = inverse_maps
    batch_data['origin']['inverse_maps'] = inverse_maps1

    batch_data['origin']['pcd_indexes'] = pcd_indexes

    return batch_data

def collation_fn_lasermix_dataset(data_labels):
    point_coords, point_feats, point_labels, point_selected_idx, point_mapped_labels,\
          voxel_coords, voxel_feats, voxel_labels, voxel_selected_idx, voxel_mapped_labels, voxel_inverse_maps, pcd_indexes = list(zip(*data_labels))
    
    batch_data = {
        'points': {},
        'voxel': {},
    }
    # points data
    point_coords = ME.utils.batched_coordinates(point_coords, dtype=torch.float32)
    point_feats = torch.from_numpy(np.concatenate(point_feats, 0)).float()
    point_labels = torch.from_numpy(np.concatenate(point_labels, 0)).int()
    point_selected_idx = torch.from_numpy(np.concatenate(point_selected_idx, 0)).long()
    point_mapped_labels = torch.from_numpy(np.concatenate(point_mapped_labels, 0)).int()

    # voxel data
    voxel_coords = ME.utils.batched_coordinates(voxel_coords)
    voxel_feats = torch.from_numpy(np.concatenate(voxel_feats, 0)).float()
    voxel_labels = torch.from_numpy(np.concatenate(voxel_labels, 0)).int()
    voxel_selected_idx = torch.from_numpy(np.concatenate(voxel_selected_idx, 0)).long()
    voxel_mapped_labels = torch.from_numpy(np.concatenate(voxel_mapped_labels, 0)).int()
    pcd_indexes = torch.tensor(pcd_indexes, dtype=torch.int16)

    batch_data['points']['coords'] = point_coords
    batch_data['points']['feats'] = point_feats
    batch_data['points']['labels'] = point_labels
    batch_data['points']['selected_idx'] = point_selected_idx
    batch_data['points']['mapped_labels'] = point_mapped_labels

    batch_data['voxel']['coords'] = voxel_coords
    batch_data['voxel']['feats'] = voxel_feats
    batch_data['voxel']['labels'] = voxel_labels
    batch_data['voxel']['selected_idx'] = voxel_selected_idx
    batch_data['voxel']['mapped_labels'] = voxel_mapped_labels
    batch_data['voxel']['pcd_indexes'] = pcd_indexes
    batch_data['voxel']['inverse_maps'] = voxel_inverse_maps

    return batch_data