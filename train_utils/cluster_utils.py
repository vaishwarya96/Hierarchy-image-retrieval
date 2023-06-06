import torch
import torch.nn as nn
from scipy.spatial.distance import pdist ,squareform
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer

from pyclustering.cluster.gmeans import gmeans
from pyclustering.cluster.kmeans import kmeans
import torch.utils.data as data
import numpy as np
import os

from config import get_cfg_defaults 
from dataset_utils import utils, load_dataset

cfg = get_cfg_defaults()


def modify_model(model, n_classes):
    '''
    Change the classification layer to accomodate 
    the change in the total number of classes after 
    clustering

    Input: model and the new number of classes n_classes
    Output: modified model with output containing n_classes predictions
    '''

    #model.linear = nn.Linear(640, n_classes)
    model.classification_layer = nn.Linear(1280, n_classes)
    return model


def create_cluster(label_database, features, paths, C_norm):
    
    '''
    Perform X-Means clustering based on the feature embeddings of each image

    Input: 
    X_train- paths corresponding to the images
    y_train- labels corresponding to the input images
    features- feature embeddings obtained from the network for the X_train images
    paths- paths corresponding to the images
    C_norm- Normalised confusion matrix 

    Output:
    '''
    
    label_dict = {}
    img_list = []
    label_list = []

    cnt = 0
    avg_eig = 0

    n_classes = np.max(label_database) + 1

    max_clusters = [1] * n_classes
    

   
    for i in range(n_classes):

        selected_features = features[np.where(label_database==i)]  
        selected_paths = paths[np.where(label_database==i)]    
        dist_mat = squareform(pdist(selected_features))   

        C_ = C_norm[i]
        C_sum = np.sum(C_)-C_[i]   #False negatives

        if C_sum > cfg.TRAIN.FNR:
            max_clusters[i] = cfg.TRAIN.MAX_CLUSTERS

        amount_initial_centers = 1
        initial_centers = kmeans_plusplus_initializer(dist_mat, amount_initial_centers).initialize()
        xmeans_instance = xmeans(dist_mat, initial_centers, max_clusters[i])
        xmeans_instance.process()
        
        clusters = xmeans_instance.get_clusters()
        centers = xmeans_instance.get_centers()
        
        #gmeans_instance = gmeans(dist_mat, tolerance=25).process()

        #clusters = gmeans_instance.get_clusters()
        #centers = gmeans_instance.get_centers()
        #kmeans_instance = kmeans(dist_mat, initial_centers).process()
        #clusters = kmeans_instance.get_clusters()
        #centers = kmeans_instance.get_centers()


        for j in range(len(clusters)):
            img_list.extend(selected_paths[clusters[j]])
            label_list.extend([cnt]*len(clusters[j]))
            label_dict[cnt] = i

            cnt += 1
            
        

    return img_list, label_list, label_dict



def extract_features(train_data, model):

    '''
    Extract feature embeddings from a CNN model

    Input:
    model - CNN model
    train_data_loader - train dataset
    
    Output:
    label_database - list of all labels 
    emb_databse - list of extracted feature embeddings
    path_database - list of paths of the images
    '''

    label_database = []
    emb_database = []
    path_database = []
    model.eval()
    with torch.no_grad():
        for it, (img, label, path) in enumerate(train_data):

            b_images = img.cuda()
            b_labels = label.cuda()
            emb, logits = model(b_images)
            #emb = torch.div(emb,torch.linalg.norm(emb, dim=1).view(-1,1))
            label_database.extend(label)
            emb_database.extend(emb.cpu().numpy())
            path_database.extend(path)

    return label_database, emb_database, path_database


def get_clustered_model_and_dataset(train_data, model, C_norm):

    '''
    Combines all the modules to get the final 
    cluster dataset and the model
    '''
    label_database, emb_database, path_database = extract_features(train_data, model)
    img_list, label_list, label_dict = create_cluster(np.array(label_database), np.array(emb_database), np.array(path_database), C_norm)

    label_counts = {}
    output_nc = len(label_dict)

    for i in range(output_nc):
        label_counts[i] = np.sum(np.array(label_list)==i)
    print("Number of clusters is", output_nc)
    model = modify_model(model, output_nc)
    model = model.cuda()
    cluster_data_loader = load_dataset.LoadDataset(img_list, label_list)
    cluster_data = data.DataLoader(cluster_data_loader, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=cfg.SYSTEM.NUM_WORKERS)
    
    return model, cluster_data, label_dict, label_counts, img_list, label_list

