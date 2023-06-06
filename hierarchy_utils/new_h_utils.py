import numpy as np
import igraph as ig
import networkx as nx
import matplotlib.pyplot as plt
import ete3
from scipy.cluster.hierarchy import dendrogram, linkage, to_tree
from ete3 import Tree

def bhattacharyya_gaussian_distance(distribution1: "dict", distribution2: "dict",) -> int:
    """ Estimate Bhattacharyya Distance (between Gaussian Distributions)
    
    Args:
        distribution1: a sample gaussian distribution 1
        distribution2: a sample gaussian distribution 2
    
    Returns:
        Bhattacharyya distance
    """
    mean1 = distribution1["mean"]
    cov1 = distribution1["covariance"]

    mean2 = distribution2["mean"]
    cov2 = distribution2["covariance"]

    cov = (1 / 2) * (cov1 + cov2)

    T1 = (1 / 8) * (
        np.sqrt((mean1 - mean2) @ np.linalg.inv(cov) @ (mean1 - mean2).T)[0][0]
    )

    T2 = (1 / 2) * np.log(
        np.linalg.det(cov) / (np.sqrt(np.linalg.det(cov1) * np.linalg.det(cov2)))
    )

    return T1 + T2

def robust_bhattacharya_distance(distribution1: "dict", distribution2: "dict", eps=1e-6):
    mu1 = distribution1["mean"]
    sigma1 = distribution1["covariance"]

    mu2 = distribution2["mean"]
    sigma2 = distribution2["covariance"]

    sigma = (sigma1 + sigma2)/2
    #sigma = np.identity(sigma.shape[0])
    u,d,v = np.linalg.svd(sigma)

    i = np.abs(d) > eps

    def logdet(s):
        return np.log(np.linalg.det(u[i] @ s @ v[i].T))

    logdet_sigma = np.sum(np.log(d[i]))
    logdet_sigma1 = logdet(sigma1)
    logdet_sigma2 = logdet(sigma2)

    q = logdet_sigma - logdet_sigma1/2 -logdet_sigma2/2
    dmu = (mu1 - mu2).T
    y = np.linalg.pinv(sigma, rcond=eps) @ dmu
    md = dmu * y
    
    final_distance = np.sum(md)/8 + q/2
    return final_distance  


def bhattacharya_coefficient(bd):
    return np.exp(-bd)

def get_distribution(embeddings, labels):

    embeddings = np.array(embeddings)
    labels = np.array(labels)
    n_classes = np.max(labels) + 1
    num_eig = embeddings.shape[1]
    distribution_params = []

    for i in range(n_classes):
        dict = {}
        index_i = np.argwhere(labels==i)
        feats = embeddings[index_i].squeeze(1)
        mean = np.mean(feats,axis=0).reshape(1,-1)
        cov = np.cov(embeddings, rowvar=0) #+ np.diag([1e-20]*num_eig)
        dict['mean'] = mean
        dict['covariance'] = cov
        distribution_params.append(dict)

    return distribution_params

def get_combinations(num_classes):
    return combinations(num_classes, 2)

def get_overlapping_region(distribution_params):
    num_classes = len(distribution_params)
    pdist = np.ones((num_classes, num_classes))

    for i in range(num_classes):
        for j in range(i+1, num_classes):
            #bd = bhattacharyya_gaussian_distance(distribution_params[i], distribution_params[j])
            bd = robust_bhattacharya_distance(distribution_params[i], distribution_params[j])
            bc = bhattacharya_coefficient(bd)
            pdist[i,j] = bc
            pdist[j,i] = 0

    return pdist

def prune_hierarchy(t):
    def get_non_leaves(node):
        if node.is_leaf():
            pass
        else:
            return node

    tree_leaves = [leaf for leaf in t]
    for node in t.traverse('preorder'):
        node_children = node.children
        for n in node_children:
            if n in tree_leaves:
                leaves = n.get_leaves()
                non_leaves = list(filter(get_non_leaves, node.traverse()))
                non_leaves.remove(node)
                for nl in non_leaves:
                    nl.delete()

    return t


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


def create_dendrogram_hierarchy(pc, train_label, id_map, label_dict):

    new_train_label = train_label.copy()
    new_train_label = np.array(new_train_label)
    train_label = np.array(train_label)
    cnt = len(label_dict)
    new_label_dict_list = []

    #while True:

    components = {}

    original_train_label = new_train_label.copy()

    #Get the mean and covariance of each class
    distribution_params = get_distribution(pc, new_train_label)

    #Compute Bhattacharya overlap coefficient and get the adjasency matrix
    overlap_matrix = get_overlapping_region(distribution_params)
    identity_matrix = np.identity(overlap_matrix.shape[0])
    adjacency_matrix = overlap_matrix #- identity_matrix

    taxa_names = list(id_map.keys())
    num_classes = identity_matrix.shape[0]
    new_taxa_numbers = np.arange(num_classes)
    actual_taxa_numbers = [label_dict[k] for k in new_taxa_numbers]
    actual_taxa_names = [taxa_names[i] for i in actual_taxa_numbers]

    Z = linkage(adjacency_matrix, 'ward')
    tree = to_tree(Z, False)
    newick = getNewick(tree, "", tree.dist)
    t = Tree(newick, format=1)
    t = prune_hierarchy(t)
    c = len(id_map)
    for node in t.traverse("postorder"):
        if node.name == '':
            node.name = str(c)
            c += 1
        else:
            node.name = node.name

    return t

def create_hierarchy(pc, train_label, id_map, label_dict):

    new_train_label = train_label.copy()
    new_train_label = np.array(new_train_label)
    train_label = np.array(train_label)
    cnt = len(label_dict)
    new_label_dict_list = []
    margin = 0.10
    original_num_classes = 0

    while True:
        new_num_classes = 0
        components = {}

        original_train_label = new_train_label.copy()

        #Get the mean and covariance of each class
        distribution_params = get_distribution(pc, new_train_label)

        #Compute Bhattacharya overlap coefficient and get the adjasency matrix
        overlap_matrix = get_overlapping_region(distribution_params)
        identity_matrix = np.identity(overlap_matrix.shape[0])
        adjacency_matrix = overlap_matrix #- identity_matrix

        #Consider only overlaps that are above a threshold
        new_adjacency_matrix = adjacency_matrix.copy()
        new_adjacency_matrix[np.where(adjacency_matrix < margin)] = 0

        for i in range(new_adjacency_matrix.shape[0]):
            adj_list = new_adjacency_matrix[i]
            overlapping_classes = np.argwhere(adj_list != 0).tolist()
            overlapping_classes = [item for sublist in overlapping_classes for item in sublist]
            present = False
            for elem in components.values():
                items = set(elem)
                if set(overlapping_classes).issubset(items):
                    present = True
                    break
            
            if not present:
                components[new_num_classes] = overlapping_classes
                new_num_classes += 1
        
        #margin+=0.05

        new_label_dict = {}
        for i in range(len(components)):
            clustered_labels = components[i]
            new_label_dict[i] = clustered_labels
            for j in clustered_labels:
                new_train_label[np.where(original_train_label == j)] = i
                
        cnt += len(components)
        
        print("New number of classes: %d" %(np.max(new_train_label) + 1))

        if original_num_classes == new_num_classes:
            new_label_dict_list.append({0:list(np.arange(len(components)))})
            return new_label_dict_list
        original_num_classes = new_num_classes
        new_label_dict_list.append(new_label_dict)
        print(new_label_dict_list)


def create_pairwise_hierarchy(pc, train_label, id_map, label_dict):

    #TODO
    new_train_label = train_label.copy()
    new_train_label = np.array(new_train_label)
    train_label = np.array(train_label)
    cnt = len(label_dict)
    new_label_dict_list = []
    margin = 0.1
    original_num_classes = 0

    while True:
        new_num_classes = 0
        components = {}

        original_train_label = new_train_label.copy()

        #Get the mean and covariance of each class
        distribution_params = get_distribution(pc, new_train_label)

        #Compute Bhattacharya overlap coefficient and get the adjasency matrix
        overlap_matrix = get_overlapping_region(distribution_params)
        identity_matrix = np.identity(overlap_matrix.shape[0])
        adjacency_matrix = overlap_matrix #- identity_matrix

        #Consider only overlaps that are above a threshold
        new_adjacency_matrix = adjacency_matrix.copy()
        new_adjacency_matrix[np.where(adjacency_matrix < margin)] = 0

        for i in range(new_adjacency_matrix.shape[0]):
            adj_list = new_adjacency_matrix[i]
            overlapping_classes = np.argwhere(adj_list != 0).tolist()
            overlapping_classes = [item for sublist in overlapping_classes for item in sublist]
            present = False
            for elem in components.values():
                items = set(elem)
                if set(overlapping_classes).issubset(items):
                    present = True
                    break
            
            if not present:
                components[new_num_classes] = overlapping_classes
                new_num_classes += 1
        
        margin-=0.02

        new_label_dict = {}
        for i in range(len(components)):
            clustered_labels = components[i]
            new_label_dict[i] = clustered_labels
            for j in clustered_labels:
                new_train_label[np.where(original_train_label == j)] = i
                
        cnt += len(components)
        
        print("New number of classes: %d" %(np.max(new_train_label) + 1))

        if original_num_classes == new_num_classes:
            new_label_dict_list.append({0:list(np.arange(len(components)))})
            return new_label_dict_list
        original_num_classes = new_num_classes
        new_label_dict_list.append(new_label_dict)  


def get_networkx_graph(label_dict_list, num_classes):

    G = nx.DiGraph()

    cnt = num_classes
    intermediate_dict = {i:i for i in range(num_classes)}
    for d in label_dict_list:
        for i in range(len(d)):
            for j in d[i]:
                G.add_edge(cnt,intermediate_dict[j])
                intermediate_dict[i] = cnt
            cnt += 1   

    return G, cnt-1 

def get_ete_graph(nx_graph, root_node):


    subtrees = {node:ete3.Tree(name=node) for node in nx_graph.nodes()}
    [*map(lambda edge:subtrees[edge[0]].add_child(subtrees[edge[1]]), nx_graph.edges())]
    tree = subtrees[root_node]
    #print(tree.get_ascii())
    #tree.show()
    #ts = ete3.TreeStyle()
    #ts.show_leaf_name = True
    #ts.branch_vertical_margin = 10 # 10 pixels between adjacent branches
    #ts.rotation = 90
    #ts.mode = "c"
    #ts.arc_start = -180 # 0 degrees = 3 o'clock
    #ts.arc_span = 180
    #tree.show(tree_style=ts)
    return tree

