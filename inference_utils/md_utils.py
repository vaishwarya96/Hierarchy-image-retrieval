import numpy as np
import scipy as sp
import torch
from scipy.stats import chi2
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, to_tree
from ete3 import Tree
import torch.nn as nn

def pca(embeddings):
    
    embeddings = np.array(embeddings)
    mean_values = np.mean(embeddings.T, axis=1)
    C = embeddings - mean_values
    covariance_mat = np.cov(C.T)
    eig_val, eig_vec = np.linalg.eig(covariance_mat)
    sorted_idx = np.argsort(eig_val)[::-1]
    eig_val = eig_val[sorted_idx]
    eig_vec = eig_vec.T[sorted_idx]
    transformed_pts = transform_features(eig_vec, embeddings, mean_values)

    return eig_val, eig_vec, transformed_pts


def transform_features(eig_vec, data_points, mean_values):

    data_points = np.array(data_points)
    C = data_points - mean_values
    transformed_points = eig_vec.T.dot(C.T).T
    return transformed_points

def mahalanobis(x, pc, label):
    label = np.array(label)
    n_classes = np.max(label) + 1
    mean_values = []
    distance_matrix = np.empty((x.shape[0], n_classes))
    num_eig = pc.shape[1]
    ''' 
    tied_cov = 0
    for i in range(n_classes):
        index_i = np.argwhere(label==i)
        feats = pc[index_i].squeeze(1)
        mean = np.mean(feats,axis=0)
        x_minus_mu = feats - np.mean(feats, axis=0)  
        tied_cov += np.dot(x_minus_mu.T, x_minus_mu)     
    tied_cov = tied_cov / label.shape[0]
    '''
    for i in range(n_classes):
        index_i = np.argwhere(label==i)
        feats = pc[index_i].squeeze(1)
        mean = np.mean(feats,axis=0)
        x_minus_mu = x - np.mean(feats, axis=0)
        cov = np.cov(pc.T) + np.diag([1e-20]*num_eig)
        #cov = tied_cov + np.diag([1e-10]*num_eig)
        inv_cov = sp.linalg.inv(cov)
        left_term = np.dot(x_minus_mu, inv_cov)
        mahal = np.dot(left_term, x_minus_mu.T)
        if isinstance(mahal, np.float):
            dist = mahal
        else:
            dist = mahal.diagonal()

        dist = np.sqrt(dist)
        distance_matrix[:,i] = dist
    
    return distance_matrix

def euclidean(x, pc, label):
    label = np.array(label)
    n_classes = np.max(label) + 1
    mean_values = []
    distance_matrix = np.empty((x.shape[0], n_classes))
    num_eig = pc.shape[1]
    
    for i in range(n_classes):
        index_i = np.argwhere(label==i)
        feats = pc[index_i].squeeze(1)
        mean = np.mean(feats,axis=0)
        x_minus_mu = x - np.mean(feats, axis=0)
        inv_cov = np.identity(num_eig)
        left_term = np.dot(x_minus_mu, inv_cov)
        mahal = np.dot(left_term, x_minus_mu.T)
        if isinstance(mahal, np.float):
            dist = mahal
        else:
            dist = mahal.diagonal()

        dist = np.sqrt(dist)
        distance_matrix[:,i] = dist
    
    return distance_matrix

def extract_features(model, dataset):
    label_database = []
    emb_database = []
    path_database = []

    with torch.no_grad():
        for it, (img, label, _) in enumerate(dataset):
            
            b_images = img.cuda()
            b_labels = label.cuda()

            emb, logits = model(b_images)
            label_database.extend(label.detach().cpu().numpy())
            emb_database.extend(emb.detach().cpu().numpy())

    return emb_database, label_database

def get_softmax_prob(model, dataset):
    label_database = []
    prob = []

    with torch.no_grad():
        for it, (img, label, _) in enumerate(dataset):

            b_images = img.cuda()
            b_labels = label.cuda()
            emb, logits = model(b_images)
            label_database.extend(label.detach().cpu().numpy())
            softmax_prob = nn.Softmax(dim=1)(logits)
            prob.extend(softmax_prob.detach().cpu().numpy())

    return prob, label_database


def extract_ood_features(model, dataset):
    label_database = []
    emb_database = []
    path_database = []

    with torch.no_grad():
        #for it, (img, label, _) in enumerate(dataset):
        for it, (img, label) in enumerate(dataset):
            
            b_images = img.cuda()
            b_labels = label.cuda()

            emb, logits = model(b_images)
            label_database.extend(label.detach().cpu().numpy())
            emb_database.extend(emb.detach().cpu().numpy())

    return emb_database, label_database

def extract_features_with_path(model, dataset):
    label_database = []
    emb_database = []
    path_database = []

    with torch.no_grad():
        for it, (img, label, path) in enumerate(dataset):
            
            b_images = img.cuda()
            b_labels = label.cuda()

            emb, logits = model(b_images)
            #emb = torch.div(emb,torch.linalg.norm(emb, dim=1).view(-1,1))
            label_database.extend(label.detach().cpu().numpy())
            emb_database.extend(emb.detach().cpu().numpy())
            path_database.extend(path)

    return emb_database, label_database, path_database

def get_md_prob(mahalanobis_distance, num_eig):
    p_values = 1-chi2.cdf(mahalanobis_distance**2, num_eig)
    n_classes = p_values.shape[1]
    #p_values = p_values/np.max(p_values, axis=1, keepdims=True)
    #norm_prob = p_values/(np.sum(p_values**2, axis=1, keepdims=True)+1e-20)
    #norm_prob = p_values*np.exp(n_classes * p_values)/np.sum(np.exp(n_classes * p_values), axis=1, keepdims=True)

    norm_prob = p_values*np.exp(n_classes * p_values)/np.sum(np.exp(n_classes * p_values), axis=1, keepdims=True)
    aleatoric_prob = norm_prob#p_values * norm_prob #norm_prob * p_values
    return p_values

def get_dict_keys(mydict, search_val):
    for key, val in mydict.items():
        if val == search_val:
            return key
        
def get_nearest_neighbours_delaunay(train_emb, train_label):
    train_label = np.array(train_label)
    n_classes = int(np.max(train_label) + 1)
    mean_points = []
    for i in range(n_classes):
        index_i = np.argwhere(train_label==i)
        feats = train_emb[index_i].squeeze(1)
        mean = np.mean(feats, axis=0)
        mean_points.append(mean)

    mean_points = np.array(mean_points)
    tri = Delaunay(mean_points)
    #plt.triplot(mean_points[:,0], mean_points[:,1], tri.simplices)
    #plt.plot(mean_points[:,0], mean_points[:,1], 'o')
    #plt.show()
    
    return tri

def get_nearest_neighbours_prob_avg(pred_prob):

    nearest_prob = (pred_prob > 0.75)
    selected_prob = pred_prob*nearest_prob
    sum_prob = np.sum(selected_prob, axis=1, keepdims=True)
    num_non_zero = np.count_nonzero(selected_prob, axis=1, keepdims=True)

    new_prob = sum_prob / (num_non_zero + 1e-20)
    new_prob = selected_prob * new_prob
    return new_prob

def get_nearest_neighbour_probability(md_matrix, ed_matrix, num_eig):
    '''
    pred_prob = get_md_prob(md_matrix, num_eig)
    exp_dist = np.exp(-ed_matrix**2/num_eig)
    normalised_prob = exp_dist / np.sum((exp_dist), axis=1, keepdims=True)

    return normalised_prob
    '''
    pred_prob = get_md_prob(md_matrix, num_eig)
    exp_dist = np.exp(-ed_matrix**2/num_eig)

    nearest_prob = (pred_prob > 0.75)
    selected_prob = pred_prob*nearest_prob
    sum_prob = np.sum(selected_prob, axis=1, keepdims=True)
    num_non_zero = np.count_nonzero(selected_prob, axis=1, keepdims=True)

    new_prob = sum_prob / (num_non_zero + 1e-20)
    new_prob = selected_prob * new_prob
    return new_prob

def get_nearest_neighbour_softmax(md_matrix, num_eig):
    pred_prob = get_md_prob(md_matrix, num_eig)
    exp_dist = np.exp(-md_matrix**2/2)
    softmax_prob = exp_dist / np.sum(exp_dist, axis=1, keepdims=True)
    softmax_prob = pred_prob * softmax_prob
    return softmax_prob    
    

def get_nearest_neighbours_prob_softmax(pred_prob):

    nearest_prob = (pred_prob > 0.75)
    selected_prob = pred_prob * nearest_prob
    num_non_zero = np.count_nonzero(selected_prob, axis=1, keepdims=True)
    norm_prob = selected_prob*np.exp(selected_prob)/np.sum(np.exp(selected_prob), axis=1, keepdims=True)

    return norm_prob

def get_nearest_neighbours_prob_softmax_1(pred_prob):

    pred_class = np.argmax(pred_prob, axis=1)   #Select the predicted class
    nearest_prob = (pred_prob > 0.75)   #Filter probabilities with values > 0.75
    selected_prob = pred_prob * nearest_prob
    prob_list = []
    
    num_samples = pred_class.shape[0]
    prob_list = [selected_prob[i][selected_prob[i]!=0] if len(selected_prob[i][selected_prob[i]!=0]) > 0 else [0] for i in range(num_samples)]
    softmax_prob_list = [np.exp(len(prob_list[i]) * prob_list[i])/np.sum(np.exp(len(prob_list[i]) * prob_list[i]), keepdims=True) for i in range(len(prob_list))]

    normalised_probability = [prob_list[i] * softmax_prob_list[i] for i in range(len(prob_list))]
    pred_prob = [np.max(i) for i in normalised_probability]

    pred_prob = np.array(pred_prob)

    return pred_class, pred_prob



def get_key(val, id_map):
    for key, value in id_map.items():
         if val == value:
             return key

def get_probabilistic_hierarchy(md, id_map, num_levels=3):

    Z = linkage(md, "ward")
    tree = to_tree(Z, False)
    newick = getNewick(tree, "", tree.dist)
    t = Tree(newick, format=1)

    n_classes = len(id_map)

    c = n_classes
    print(len(t.get_leaves()))

    #for node in t.traverse("postorder"):
    #    if node.name == '':
    #        node.name = str(c)
    #        c += 1
    #    else:
    #        node.name = node.name

    return t

def get_neighbouring_classes(pred_prob, id_map, num_levels=3):

    print(np.max(pred_prob[0]))
    exit(0)
    n_classes = len(id_map)
    t = get_probabilistic_hierarchy(md, id_map, num_levels)
    confused_dict = {}

    for i in range(n_classes):
        confused_taxa = []
        node = t&str(i)
        node_ancestors = node.get_ancestors()
        confusion_list = []
        for j in range(num_levels):
            ancestor = node_ancestors[j]
            confusion_taxa = ancestor.get_leaves()
            confusion_taxa = [n.name for n in confusion_taxa]
            confusion_taxa.remove(node.name)
            confusion_taxa = [get_key(int(txa), id_map) for txa in confusion_taxa]

            confusion_list.append(confusion_taxa)
            confused_dict[get_key(i, id_map)] = confusion_list
    #print(confused_dict)
    return confused_dict



def getNewick(node, newick, parentdist):
    if node.is_leaf():
        return "%s:%.2f%s" % (str(node.id), parentdist - node.dist, newick)
    else:
        if len(newick) > 0:
            newick = "):%.2f%s" % (parentdist - node.dist, newick)
        else:
            newick = ");"
        newick = getNewick(node.get_left(), newick, node.dist)
        newick = getNewick(node.get_right(), ",%s" % (newick), node.dist)
        newick = "(%s" % (newick)
        return newick
