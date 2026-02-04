import torch
import numpy as np
import spconv.pytorch as spconv
import MinkowskiEngine as ME

def collation_fn_ssl_dataset(batch):
    coords_list = []
    # feats_list = []
    labels_list = []
    mapped_labels_list = []
    selected_idx_list = []
    t_list = []
    if isinstance(batch[-1], tuple):
        for data in batch:
            if len(data) == 6: # label dataset
                coords, feats, labels, selected_idx, mapped_labels, t = data
                labels_list.append(labels)
                mapped_labels_list.append(mapped_labels)

            elif len(data) == 4: #unlabel dataset
                coords, feats, selected_idx, t = data
                labels_list.append(torch.tensor([], dtype=torch.int))
                mapped_labels_list.append(torch.tensor([], dtype=torch.int))

            selected_idx_list.append(selected_idx)
            coords_list.append(coords)
            # feats_list.append(feats)
            t_list.append(t)
        bcoords = np.concatenate(coords_list, axis=0)
        bcoords = torch.tensor(bcoords, dtype=torch.float32)
        # feats = torch.from_numpy(np.concatenate(feats_list, 0)).float()
        labels = torch.from_numpy(np.concatenate(labels_list, 0)).int()
        selected_idx = torch.from_numpy(np.concatenate(selected_idx, 0)).long()
        mapped_labels = torch.from_numpy(np.concatenate(mapped_labels_list, 0)).int()
        t = torch.tensor(t_list, dtype=torch.int16)
        return bcoords, feats, labels, selected_idx, mapped_labels, t

    elif isinstance(batch[-1], dict):
        for data in batch:
            if 'labeled' in data.keys(): #label
                datum_type = 'labeled'
                data = data['labeled']
                coords = data['points']
                # features = data['features']
                mapped_labels = data['pts_semantic_mask']
                labels = data['labels']
                selected_idx = data['selected_idx']
                t = data['idx']
                labels_list.append(labels)
                mapped_labels_list.append(mapped_labels)

            elif 'unlabeled' in data.keys(): #unlabel
                datum_type = 'unlabeled'
                data = data['unlabeled']
                coords = data['points']
                # features = data['features']
                selected_idx = data['selected_idx']
                t = data['idx']
                labels_list.append(torch.tensor([], dtype=torch.int))
                mapped_labels_list.append(torch.tensor([], dtype=torch.int))

            selected_idx_list.append(selected_idx)
            coords_list.append(coords)
            # feats_list.append(features)
            t_list.append(t)

        coords_list = [torch.from_numpy(points).to(torch.float32) for points in coords_list]
        # feats_list = [torch.from_numpy(feats).to(torch.float32) for feats in feats_list]
        labels_list = [torch.from_numpy(label).to(torch.int16) for label in labels_list]
        selected_idx_list = [torch.from_numpy(idx).to(torch.int32) for idx in selected_idx_list]
        t_list = [torch.tensor(t, dtype=torch.int16) for t in t_list]
        mapped_labels_list = [torch.from_numpy(mapped_label).to(torch.int16) for mapped_label in mapped_labels_list]

        return {
            'datum_type' : datum_type,
            'points': coords_list,
            # 'features': feats_list,
            'labels': labels_list,
            'selected_idx' : selected_idx_list,
            'pts_semantic_mask': mapped_labels_list,
            'idx': t_list
                }


def collation_fn_ssl_dataset_two_samples(batch):
    coords_list = []
    coords_list1 = []
    # feats_list = []
    # feats_list1 = []
    labels_list = []
    labels_list1 = []
    selected_idx_list, selected_idx_list1 = [], []
    mapped_labels_list = []
    mapped_labels_list1 = []
    t_list = []
    t_list1 = []
    if isinstance(batch[-1][-1], tuple):
        for data in batch:
            if len(data) == 9: #label dataset
                datum_type = 'labeled'
                coords, feats, labels, mapped_labels, coords1, feats1, labels1, mapped_labels1, t = data
                labels_list.append(labels)
                labels_list1.append(labels1)
                mapped_labels_list.append(mapped_labels)
                mapped_labels_list1.append(mapped_labels1)
            elif len(data) == 5: #unlabel dataset
                datum_type = 'unlabeled'
                coords, feats, coords1, feats1, t = data
                labels_list.append(torch.tensor([], dtype=torch.int))
                labels_list1.append(torch.tensor([], dtype=torch.int))
                mapped_labels_list.append(torch.tensor([], dtype=torch.int))
                mapped_labels_list1.append(torch.tensor([], dtype=torch.int))
            coords_list.append(coords)
            coords_list1.append(coords1)
            feats_list.append(feats)
            feats_list1.append(feats1)
            t_list.append(t)

        coords_list = [torch.from_numpy(points).to(torch.float32) for points in coords_list]
        coords_list1 = [torch.from_numpy(points).to(torch.float32) for points in coords_list1]

        feats_list = [torch.from_numpy(feats).to(torch.float32) for feats in feats_list]
        feats_list1 = [torch.from_numpy(feats).to(torch.float32) for feats in feats_list1]

        labels_list = [torch.from_numpy(label).to(torch.int16) for label in labels_list]
        labels_list1 = [torch.from_numpy(label).to(torch.int16) for label in labels_list1]

        t_list = [torch.tensor(t, dtype=torch.int16) for t in t_list]
        t_list1 = [torch.tensor(t, dtype=torch.int16) for t in t_list1]


        mapped_labels_list = [torch.from_numpy(mapped_label).to(torch.int16) for mapped_label in mapped_labels_list]
        mapped_labels_list1 = [torch.from_numpy(mapped_label).to(torch.int16) for mapped_label in mapped_labels_list1]


        return datum_type, coords_list, feats_list, labels_list, mapped_labels_list, t_list,  coords_list1, feats_list1, labels_list1, mapped_labels_list1, t_list1

    elif isinstance(batch[-1][-1], dict):
        for data in batch:
            datum1, datum2 = data
            if 'labeled' in datum1.keys(): #label
                datum_type = 'labeled'
                datum1 = datum1['labeled']
                coords = datum1['points']
                # features = datum1['features']
                mapped_labels = datum1['pts_semantic_mask']
                selected_idx = datum1['selected_idx']
                labels = datum1['labels']
                t = datum1['idx']
                labels_list.append(labels)
                mapped_labels_list.append(mapped_labels)
            elif 'unlabeled' in datum1.keys(): #unlabel
                datum_type = 'unlabeled'
                datum1 = datum1['unlabeled']
                coords = datum1['points']
                selected_idx = datum1['selected_idx']
                # features = datum1['features']
                t = datum1['idx']
                labels_list.append(torch.tensor([], dtype=torch.int))
                mapped_labels_list.append(torch.tensor([], dtype=torch.int))

            selected_idx_list.append(selected_idx)
            coords_list.append(coords)
            # feats_list.append(features)
            t_list.append(t)

            if 'labeled' in datum2.keys(): #label
                datum2 = datum2['labeled']
                coords1 = datum2['points']
                # features1 = datum2['features']
                mapped_labels1 = datum2['pts_semantic_mask']
                labels1 = datum2['labels']
                selected_idx1 = datum2['selected_idx']
                t1 = datum2['idx']
                labels_list1.append(labels1)
                mapped_labels_list1.append(mapped_labels1)

            elif 'unlabeled' in datum2.keys(): #unlabel
                datum2 = datum2['unlabeled']
                coords1 = datum2['points']
                # features1 = datum2['features']
                selected_idx1 = datum2['selected_idx']
                t1 = datum2['idx']
                labels_list1.append(torch.tensor([], dtype=torch.int))
                mapped_labels_list1.append(torch.tensor([], dtype=torch.int))

            selected_idx_list1.append(selected_idx1)
            coords_list1.append(coords1)
            # feats_list1.append(features1)
            t_list1.append(t1)

        # coords_list의 원소 np.ndarray -> torch.Tensor로 바꿔야함
        coords_list = [torch.from_numpy(points).to(torch.float32) for points in coords_list]
        coords_list1 = [torch.from_numpy(points).to(torch.float32) for points in coords_list1]

        selected_idx_list = [torch.from_numpy(idx).to(torch.int32) for idx in selected_idx_list]
        selected_idx_list1 = [torch.from_numpy(idx).to(torch.int32) for idx in selected_idx_list1]

        t_list = [torch.tensor(t, dtype=torch.int16) for t in t_list]
        t_list1 = [torch.tensor(t, dtype=torch.int16) for t in t_list1]

        if datum_type == 'labeled':
            labels_list = [torch.from_numpy(label).to(torch.int16) for label in labels_list]
            labels_list1 = [torch.from_numpy(label).to(torch.int16) for label in labels_list1]
            mapped_labels_list = [torch.from_numpy(mapped_label).to(torch.int16) for mapped_label in mapped_labels_list]
            mapped_labels_list1 = [torch.from_numpy(mapped_label).to(torch.int16) for mapped_label in mapped_labels_list1]

        elif datum_type == 'unlabeled':
            labels_list = [label.to(torch.int32) for label in labels_list]
            labels_list1 = [label.to(torch.int32) for label in labels_list1]
            mapped_labels_list = [mapped_label.to(torch.int32) for mapped_label in mapped_labels_list]
            mapped_labels_list1 = [mapped_label.to(torch.int32) for mapped_label in mapped_labels_list1]


        return ({
            'datum_type' : datum_type,
            'points': coords_list,
            'labels': labels_list,
            'selected_idx' : selected_idx_list,
            'pts_semantic_mask': mapped_labels_list,
            'idx': t_list
                },
                {
            'datum_type' : datum_type,
            'points': coords_list1,
            'labels': labels_list1,
            'selected_idx' : selected_idx_list1,
            'pts_semantic_mask': mapped_labels_list1,
            'idx': t_list1
                },
                )


