# Get the ID dataset
# Add degradations to the ID images, and get the corresponding predictions
# For each level in the hierarchy plot the degradation vs accuracy

import torch
import numpy as np
import torch.utils.data as data
from sklearn.metrics import accuracy_score

from dataset_utils import load_dataset
from config import get_cfg_defaults
from inference_utils import md_utils, metrics

cfg = get_cfg_defaults()

def get_topk_accuracy(output, target, label_dict, ks=(1,3,5)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    
    output = torch.tensor(output)
    target = torch.tensor(target)
    with torch.no_grad():
        maxk = max(ks)
        batch_size = target.size(0)
        # Get the class index of the top <maxk> scores for each element of the minibatch
        _, pred_ = output.topk(maxk, 1, True, True)
        actual_pred = [label_dict[int(j)] for i in pred_ for j in i]
        new_pred = torch.tensor(np.array(actual_pred).reshape(-1,maxk))
        
        '''
        pred = new_pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in ks:
            correct_k = correct[:k].reshape(-1).float()#.sum(0, keepdim=True)
            print(correct_k.shape)
            res.append(correct_k.mul_(1.0 / batch_size))
        return res, pred_
        '''
        results = []
        for k in  ks:
            correct = 0
            for i in range(output.shape[0]):
                target_i = target[i]
                pred = new_pred[i][:k]
                if target_i in pred:
                    correct += 1

            results.append(correct/output.shape[0])

        return results, pred_.numpy()

def get_hierarchy_predictions(t, gt, pred, label_dict, height_of_tree):

    root_name = t.name

    gt_node = t&str(gt)
    pred_node = t&str(pred)
    
    
    gt_node_ancestors = gt_node.get_ancestors()
    gt_node_path = get_path(gt_node.name, gt_node_ancestors, height_of_tree)
    pred_node_ancestors = pred_node.get_ancestors()
    pred_node_path = get_path(pred_node.name, pred_node_ancestors, height_of_tree)

    return gt_node_path, pred_node_path


def get_path(node_name, ancestors, height_of_tree):
    ancestors = [i.name for i in ancestors]
    total_no_of_levels = int(height_of_tree + 1)
    extra_node = (total_no_of_levels - len(ancestors)) * [node_name]
    ancestors = extra_node + ancestors
    return ancestors


def get_robustness_metrics(X, y, t, model, selected_eig_vectors, mean_train_emb, pc, train_label, num_eig, label_dict):

    # 1. Get the dataset with increasing level of blurriness and see how it impacts the accuracy
    # 2. Get the accuracy at different levels of the hierarchy
    for leaf in t:
        height_of_tree = t.get_distance('1', topology_only=True)
        break

    '''
    gaussian_kernel_size=[1,3,5,7]

    for kernel_size in gaussian_kernel_size:

        id_data_loader = load_dataset.LoadNoisyDataset(X, y, blur_kernel_size=kernel_size)
        id_data = data.DataLoader(id_data_loader, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False, num_workers=cfg.SYSTEM.NUM_WORKERS)
        id_emb, id_label = md_utils.extract_features(model, id_data)

        transformed_id_data = md_utils.transform_features(selected_eig_vectors, id_emb, mean_train_emb)
        id_md_matrix = md_utils.mahalanobis(transformed_id_data, pc, train_label)
        id_pred_prob = md_utils.get_md_prob(id_md_matrix, num_eig)
        id_pred_prob = md_utils.get_nearest_neighbour_softmax(id_md_matrix, num_eig)

        pred_class = np.argmin(id_md_matrix, axis=1)
        accuracy = metrics.get_accuracy(pred_class, id_label) 
        #print("Accuracy: %f" %(accuracy))

        hierarchy_gt_path = []
        hierarchy_pred_path = []

        for i in range(pred_class.shape[0]):
            gt_node_path, pred_node_path = get_hierarchy_predictions(t, id_label[i], pred_class[i], label_dict, height_of_tree)
            hierarchy_gt_path.append(gt_node_path)
            hierarchy_pred_path.append(pred_node_path)

        h_accuracy_layer = []
        tree_levels = int(height_of_tree + 1)
        avg_set_size = []

        for i in range(tree_levels):
            set_size = []
            gt_layer = [item[i] for item in hierarchy_gt_path]
            pred_layer = [item[i] for item in hierarchy_pred_path]

            for pred_ in pred_layer:
                leaves = (t&pred_).get_leaves()
                leaves_set = set(leaves)
                set_size.append(len(leaves_set))
            print("Avg set size is: ", np.array(set_size).mean())

            h_accuracy_layer.extend([accuracy_score(np.array(gt_layer), np.array(pred_layer))])
        print(h_accuracy_layer)


        #softmax_prob, label_database = md_utils.get_softmax_prob(model, id_data)
        #softmax_prob = np.array(softmax_prob)
        ks=(1,2,3)
        topK_accuracies, topK_predicted_classes = get_topk_accuracy(id_pred_prob, id_label, label_dict, ks=ks)
        print(topK_accuracies)
    
    '''
    '''
    rotation_angle=[0,20,40,60]

    for kernel_size in rotation_angle:

        id_data_loader = load_dataset.LoadNoisyDataset(X, y, rotation=kernel_size)
        id_data = data.DataLoader(id_data_loader, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False, num_workers=cfg.SYSTEM.NUM_WORKERS)
        id_emb, id_label = md_utils.extract_features(model, id_data)

        transformed_id_data = md_utils.transform_features(selected_eig_vectors, id_emb, mean_train_emb)
        id_md_matrix = md_utils.mahalanobis(transformed_id_data, pc, train_label)
        id_pred_prob = md_utils.get_md_prob(id_md_matrix, num_eig)
        id_pred_prob = md_utils.get_nearest_neighbour_softmax(id_md_matrix, num_eig)

        pred_class = np.argmin(id_md_matrix, axis=1)
        accuracy = metrics.get_accuracy(pred_class, id_label) 
        #print("Accuracy: %f" %(accuracy))

        hierarchy_gt_path = []
        hierarchy_pred_path = []

        for i in range(pred_class.shape[0]):
            gt_node_path, pred_node_path = get_hierarchy_predictions(t, id_label[i], pred_class[i], label_dict, height_of_tree)
            hierarchy_gt_path.append(gt_node_path)
            hierarchy_pred_path.append(pred_node_path)

        h_accuracy_layer = []
        tree_levels = int(height_of_tree + 1)
        avg_set_size = []

        for i in range(tree_levels):
            set_size = []
            gt_layer = [item[i] for item in hierarchy_gt_path]
            pred_layer = [item[i] for item in hierarchy_pred_path]

            for pred_ in pred_layer:
                leaves = (t&pred_).get_leaves()
                leaves_set = set(leaves)
                set_size.append(len(leaves_set))
            print("Avg set size is: ", np.array(set_size).mean())

            h_accuracy_layer.extend([accuracy_score(np.array(gt_layer), np.array(pred_layer))])
        print(h_accuracy_layer)


        #softmax_prob, label_database = md_utils.get_softmax_prob(model, id_data)
        #softmax_prob = np.array(softmax_prob)
        ks=(1,2,3)
        topK_accuracies, topK_predicted_classes = get_topk_accuracy(id_pred_prob, id_label, label_dict, ks=ks)
        print(topK_accuracies)
    
    '''
    
    random_erasing=[0, 0.2, 0.4, 0.6, 0.8, 1.0]

    for kernel_size in random_erasing:

        id_data_loader = load_dataset.LoadNoisyDataset(X, y, random_erasing_prob = kernel_size)
        id_data = data.DataLoader(id_data_loader, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False, num_workers=cfg.SYSTEM.NUM_WORKERS)
        id_emb, id_label = md_utils.extract_features(model, id_data)

        transformed_id_data = md_utils.transform_features(selected_eig_vectors, id_emb, mean_train_emb)
        id_md_matrix = md_utils.mahalanobis(transformed_id_data, pc, train_label)
        id_pred_prob = md_utils.get_md_prob(id_md_matrix, num_eig)
        id_pred_prob = md_utils.get_nearest_neighbour_softmax(id_md_matrix, num_eig)

        pred_class = np.argmin(id_md_matrix, axis=1)
        accuracy = metrics.get_accuracy(pred_class, id_label) 
        #print("Accuracy: %f" %(accuracy))

        hierarchy_gt_path = []
        hierarchy_pred_path = []

        for i in range(pred_class.shape[0]):
            gt_node_path, pred_node_path = get_hierarchy_predictions(t, id_label[i], pred_class[i], label_dict, height_of_tree)
            hierarchy_gt_path.append(gt_node_path)
            hierarchy_pred_path.append(pred_node_path)

        h_accuracy_layer = []
        tree_levels = int(height_of_tree + 1)
        avg_set_size = []

        for i in range(tree_levels):
            set_size = []
            gt_layer = [item[i] for item in hierarchy_gt_path]
            pred_layer = [item[i] for item in hierarchy_pred_path]

            for pred_ in pred_layer:
                leaves = (t&pred_).get_leaves()
                leaves_set = set(leaves)
                set_size.append(len(leaves_set))
            print("Avg set size is: ", np.array(set_size).mean())

            h_accuracy_layer.extend([accuracy_score(np.array(gt_layer), np.array(pred_layer))])
        print(h_accuracy_layer)


        #softmax_prob, label_database = md_utils.get_softmax_prob(model, id_data)
        #softmax_prob = np.array(softmax_prob)
        ks=(1,2,3)
        topK_accuracies, topK_predicted_classes = get_topk_accuracy(id_pred_prob, id_label, label_dict, ks=ks)
        print(topK_accuracies)
    

    '''
    gaussian_noise=[0, 0.2, 0.4, 0.6, 0.8, 1.0]

    for kernel_size in gaussian_noise:

        id_data_loader = load_dataset.LoadNoisyDataset(X, y, gaussian_noise_prob = kernel_size)
        id_data = data.DataLoader(id_data_loader, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False, num_workers=cfg.SYSTEM.NUM_WORKERS)
        id_emb, id_label = md_utils.extract_features(model, id_data)

        transformed_id_data = md_utils.transform_features(selected_eig_vectors, id_emb, mean_train_emb)
        id_md_matrix = md_utils.mahalanobis(transformed_id_data, pc, train_label)
        id_pred_prob = md_utils.get_md_prob(id_md_matrix, num_eig)
        id_pred_prob = md_utils.get_nearest_neighbour_softmax(id_md_matrix, num_eig)

        pred_class = np.argmin(id_md_matrix, axis=1)
        accuracy = metrics.get_accuracy(pred_class, id_label) 
        #print("Accuracy: %f" %(accuracy))

        hierarchy_gt_path = []
        hierarchy_pred_path = []

        for i in range(pred_class.shape[0]):
            gt_node_path, pred_node_path = get_hierarchy_predictions(t, id_label[i], pred_class[i], label_dict, height_of_tree)
            hierarchy_gt_path.append(gt_node_path)
            hierarchy_pred_path.append(pred_node_path)

        h_accuracy_layer = []
        tree_levels = int(height_of_tree + 1)
        avg_set_size = []

        for i in range(tree_levels):
            set_size = []
            gt_layer = [item[i] for item in hierarchy_gt_path]
            pred_layer = [item[i] for item in hierarchy_pred_path]

            for pred_ in pred_layer:
                leaves = (t&pred_).get_leaves()
                leaves_set = set(leaves)
                set_size.append(len(leaves_set))
            print("Avg set size is: ", np.array(set_size).mean())

            h_accuracy_layer.extend([accuracy_score(np.array(gt_layer), np.array(pred_layer))])
        print(h_accuracy_layer)


        #softmax_prob, label_database = md_utils.get_softmax_prob(model, id_data)
        #softmax_prob = np.array(softmax_prob)
        ks=(1,2,3)
        topK_accuracies, topK_predicted_classes = get_topk_accuracy(id_pred_prob, id_label, label_dict, ks=ks)
        print(topK_accuracies)
    '''

def get_new_robustness_metrics(X, y, t, model, selected_eig_vectors, mean_train_emb, pc, train_label, num_eig, label_dict):

    # 1. Get the dataset with increasing level of blurriness and see how it impacts the accuracy
    # 2. Get the accuracy at different levels of the hierarchy
    for leaf in t:
        height_of_tree = t.get_distance('1', topology_only=True)
        break

    # First get the metrics on the leaf nodes and retain only those samples that are accurately classified.
    id_data_loader = load_dataset.LoadNoisyDataset(X, y)
    id_data = data.DataLoader(id_data_loader, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False, num_workers=cfg.SYSTEM.NUM_WORKERS)
    id_emb, id_label = md_utils.extract_features(model, id_data)

    transformed_id_data = md_utils.transform_features(selected_eig_vectors, id_emb, mean_train_emb)
    id_md_matrix = md_utils.mahalanobis(transformed_id_data, pc, train_label)
    id_pred_prob = md_utils.get_md_prob(id_md_matrix, num_eig)
    id_pred_prob = md_utils.get_nearest_neighbour_softmax(id_md_matrix, num_eig)

    pred_class = np.argmin(id_md_matrix, axis=1)

    selected_samples_idx = np.argwhere(pred_class == id_label)
    
    selected_X = X[selected_samples_idx]
    selected_y = y[selected_samples_idx]
    new_X = [i[0] for i in selected_X]
    new_y = [i[0] for i in selected_y]




    '''
    random_erasing=[0, 0.2, 0.4, 0.6, 0.8, 1.0]

    for kernel_size in random_erasing:

        id_data_loader = load_dataset.LoadNoisyDataset(new_X, new_y, random_erasing_prob = kernel_size)
        id_data = data.DataLoader(id_data_loader, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False, num_workers=cfg.SYSTEM.NUM_WORKERS)
        id_emb, id_label = md_utils.extract_features(model, id_data)

        transformed_id_data = md_utils.transform_features(selected_eig_vectors, id_emb, mean_train_emb)
        id_md_matrix = md_utils.mahalanobis(transformed_id_data, pc, train_label)
        id_pred_prob = md_utils.get_md_prob(id_md_matrix, num_eig)
        id_pred_prob = md_utils.get_nearest_neighbour_softmax(id_md_matrix, num_eig)

        pred_class = np.argmin(id_md_matrix, axis=1)
        accuracy = metrics.get_accuracy(pred_class, id_label) 
        #print("Accuracy: %f" %(accuracy))

        hierarchy_gt_path = []
        hierarchy_pred_path = []

        for i in range(pred_class.shape[0]):
            gt_node_path, pred_node_path = get_hierarchy_predictions(t, id_label[i], pred_class[i], label_dict, height_of_tree)
            hierarchy_gt_path.append(gt_node_path)
            hierarchy_pred_path.append(pred_node_path)

        h_accuracy_layer = []
        tree_levels = int(height_of_tree + 1)
        avg_set_size = []

        for i in range(tree_levels):
            set_size = []
            gt_layer = [item[i] for item in hierarchy_gt_path]
            pred_layer = [item[i] for item in hierarchy_pred_path]

            for pred_ in pred_layer:
                leaves = (t&pred_).get_leaves()
                leaves_set = set(leaves)
                set_size.append(len(leaves_set))
            print("Avg set size is: ", np.array(set_size).mean())

            h_accuracy_layer.extend([accuracy_score(np.array(gt_layer), np.array(pred_layer))])
        print(h_accuracy_layer)


        #softmax_prob, label_database = md_utils.get_softmax_prob(model, id_data)
        #softmax_prob = np.array(softmax_prob)
        ks=(1,2,3)
        topK_accuracies, topK_predicted_classes = get_topk_accuracy(id_pred_prob, id_label, label_dict, ks=ks)
        print(topK_accuracies)
    '''

    gaussian_kernel_size=[1,3,5,7]

    for kernel_size in gaussian_kernel_size:

        id_data_loader = load_dataset.LoadNoisyDataset(new_X, new_y, blur_kernel_size=kernel_size)
        id_data = data.DataLoader(id_data_loader, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False, num_workers=cfg.SYSTEM.NUM_WORKERS)
        id_emb, id_label = md_utils.extract_features(model, id_data)

        transformed_id_data = md_utils.transform_features(selected_eig_vectors, id_emb, mean_train_emb)
        id_md_matrix = md_utils.mahalanobis(transformed_id_data, pc, train_label)
        id_pred_prob = md_utils.get_md_prob(id_md_matrix, num_eig)
        id_pred_prob = md_utils.get_nearest_neighbour_softmax(id_md_matrix, num_eig)

        pred_class = np.argmin(id_md_matrix, axis=1)
        accuracy = metrics.get_accuracy(pred_class, id_label) 
        #print("Accuracy: %f" %(accuracy))

        hierarchy_gt_path = []
        hierarchy_pred_path = []

        for i in range(pred_class.shape[0]):
            gt_node_path, pred_node_path = get_hierarchy_predictions(t, id_label[i], pred_class[i], label_dict, height_of_tree)
            hierarchy_gt_path.append(gt_node_path)
            hierarchy_pred_path.append(pred_node_path)

        h_accuracy_layer = []
        tree_levels = int(height_of_tree + 1)
        avg_set_size = []

        for i in range(tree_levels):
            set_size = []
            gt_layer = [item[i] for item in hierarchy_gt_path]
            pred_layer = [item[i] for item in hierarchy_pred_path]

            for pred_ in pred_layer:
                leaves = (t&pred_).get_leaves()
                leaves_set = set(leaves)
                set_size.append(len(leaves_set))
            print("Avg set size is: ", np.array(set_size).mean())

            h_accuracy_layer.extend([accuracy_score(np.array(gt_layer), np.array(pred_layer))])
        print(h_accuracy_layer)


        #softmax_prob, label_database = md_utils.get_softmax_prob(model, id_data)
        #softmax_prob = np.array(softmax_prob)
        ks=(1,2,3)
        topK_accuracies, topK_predicted_classes = get_topk_accuracy(id_pred_prob, id_label, label_dict, ks=ks)
        print(topK_accuracies)