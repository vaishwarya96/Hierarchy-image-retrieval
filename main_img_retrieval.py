import pickle
import os
import numpy as np
import torch
import torch.utils.data as data
import torch.nn as nn
import scipy
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import torchvision

from config import get_cfg_defaults
from models.efficientnet import EfficientNet_new, CUBNet
from dataset_utils import utils, load_dataset
from inference_utils import md_utils, metrics
from hierarchy_utils import img_retrieval as ir
from hierarchy_utils import h_utils
from hierarchy_utils.img_retrieval_metrics import HierarchyMetrics

cfg = get_cfg_defaults()

checkpoint = cfg.MODEL.CHECKPOINT_DIR
np.random.seed(cfg.SYSTEM.RANDOM_SEED)
torch.random.manual_seed(cfg.SYSTEM.RANDOM_SEED)

#Load the id map with the class label information
id_map = utils.get_id_map(cfg.DATASET.ID_MAP_PATH)

label_dict = pickle.load(open(os.path.join(checkpoint, 'label_dict.pkl'), 'rb'))
n_classes = len(label_dict)
print("number of classes is %d" %(n_classes))

#id_map = label_dict.copy()


#Load the model
model_path = os.path.join(checkpoint, cfg.MODEL.EXPERIMENT)
model = EfficientNet_new(num_classes = n_classes, in_channels=cfg.DATASET.NUM_CHANNELS)
model.load_state_dict(torch.load(model_path))
model = model.cuda()
model.eval()


#Load the train and test dataset
train_X, train_y, _ = utils.get_dataset(cfg.DATASET.DATASET_PATH, id_map)
train_data_loader = load_dataset.LoadDataset(train_X, train_y, train=False)
train_data = data.DataLoader(train_data_loader, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False, num_workers=cfg.SYSTEM.NUM_WORKERS)

test_X, test_y, _ = utils.get_dataset(cfg.INF.ID_TEST_DATASET, id_map)
test_data_loader = load_dataset.LoadDataset(id_X, id_y, train=False)
test_data = data.DataLoader(test_data_loader, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False, num_workers=cfg.SYSTEM.NUM_WORKERS)


#Extract the features and perform PCA
train_emb, train_label = md_utils.extract_features(model, train_data)
mean_train_emb = np.mean(np.array(train_emb).T, axis=1)
eigen_values, eigen_vectors, transformed_pts = md_utils.pca(train_emb)
explained_variances = eigen_values / np.sum(eigen_values)
cumsum = np.cumsum(explained_variances)
num_eig = int(np.argwhere(cumsum>cfg.INF.EXP_VAR_THRESHOLD)[0])

print("Number of principal components is %d" %(num_eig))

selected_eig_vectors = eigen_vectors[:,:num_eig]
pc = transformed_pts[:,:num_eig]

###Test data analysis###
id_emb, id_label = md_utils.extract_features(model, test_data)
#Convert to PCA frame
transformed_id_data = md_utils.transform_features(selected_eig_vectors, id_emb, mean_train_emb)
id_md_matrix = md_utils.mahalanobis(transformed_id_data, pc, train_label)
id_pred_prob = md_utils.get_md_prob(id_md_matrix, num_eig)
id_pred_prob = md_utils.get_nearest_neighbour_softmax(id_md_matrix, num_eig)

pred_class = np.argmin(id_md_matrix, axis=1)
accuracy = metrics.get_accuracy(pred_class, id_label) 
print("Accuracy: %f" %(accuracy))




#Get the hierarchy
label_dict_list = h_utils.create_xmeans_hierarchy(transformed_id_data, id_label, id_map, label_dict)
graph, root_node = h_utils.get_networkx_graph(label_dict_list, n_classes)
ete_tree = h_utils.get_ete_graph(graph, root_node)


#with open('ete_hierarchy_new.pkl', 'wb') as handle:
#    pickle.dump(ete_tree, handle)

#ete_tree = pickle.load(open('ete_hierarchy_new.pkl', 'rb'))


#taxonomy_file_path = cfg.HIER.HIERARCHY_FILE_PATH
#ete_tree = h_utils.get_taxonomy_cub(taxonomy_file_path, id_map)

#ete_tree = h_utils.create_dendrogram_hierarchy(transformed_id_data, id_label, id_map, label_dict)

for node in ete_tree.traverse("postorder"):
    if node.name == '':
        node.name = str(c)
        c += 1
    else:
        node.name = str(int(node.name))



#Get retrieved images
hr = ir.HierarchyRetrieval(ete_tree, n_classes)
retrieved = hr.get_retrieved_imgs_hierarchy(id_emb, pred_class)

#retrieved = ir.get_retrieved_imgs(id_emb)
#Calculate metrics



h_metrics = HierarchyMetrics(ete_tree)


#hprec, hmap = h_metrics.get_hierarchy_metrics(retrieved, id_label, topK_to_consider = (1, 5, 10))

topK_to_consider = (1, 5, 10, 20)

map_values = []
for k in topK_to_consider:
    map_values.append(h_metrics.calculate_map(retrieved, id_label, k))
print(map_values)



