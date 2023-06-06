import numpy as np
import igraph as ig
import networkx as nx
import matplotlib.pyplot as plt
import ete3
from scipy.cluster.hierarchy import dendrogram, linkage, to_tree
from ete3 import Tree
import pandas as pd
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from scipy.spatial.distance import pdist,squareform

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

    while True:

        original_train_label = new_train_label.copy()

        #Get the mean and covariance of each class
        distribution_params = get_distribution(pc, new_train_label)

        #Compute Bhattacharya overlap coefficient and get the adjasency matrix
        overlap_matrix = get_overlapping_region(distribution_params)
        identity_matrix = np.identity(overlap_matrix.shape[0])
        adjacency_matrix = overlap_matrix #- identity_matrix

        #Get the actual class from the clusters
        taxa_names = list(id_map.keys())
        num_classes = identity_matrix.shape[0]
        new_taxa_numbers = np.arange(num_classes)
        actual_taxa_numbers = [label_dict[k] for k in new_taxa_numbers]
        actual_taxa_names = [taxa_names[i] for i in actual_taxa_numbers]

        #Consider only overlaps that are above a threshold
        new_adjacency_matrix = adjacency_matrix.copy()
        new_adjacency_matrix[np.where(adjacency_matrix < margin)] = 0
        #new_adjacency_matrix.compress(np.all(s == 0, axis=0), axis=1)
        #margin -= 0.02

        

        #Create a graph from the new adjasency matrix and cluster them.
        conn_indices = np.where(new_adjacency_matrix)
        weights = new_adjacency_matrix[conn_indices]
        edges = zip(*conn_indices)
        G = ig.Graph(edges=edges, directed=True)
        #G.vs['label'] = actual_taxa_names
        G.es['weight'] = weights
        G.es['width'] = weights
        components = G.clusters(mode='weak')

        new_label_dict = {}
        for i in range(len(components)):
            clustered_labels = components[i]
            new_label_dict[i] = clustered_labels
            for j in clustered_labels:
                new_train_label[np.where(original_train_label == j)] = i
                
        cnt += len(components)
        
        print("New number of classes: %d" %(np.max(new_train_label) + 1))

        if np.max(original_train_label) == np.max(new_train_label):
            new_label_dict_list.append({0:list(np.arange(len(components)))})
            return new_label_dict_list
        new_label_dict_list.append(new_label_dict)


def create_xmeans_hierarchy(pc, train_label, id_map, label_dict):

    new_train_label = train_label.copy()
    new_train_label = np.array(new_train_label)
    train_label = np.array(train_label)
    cnt = len(label_dict)
    new_label_dict_list = []
    margin = 0.20
    th = 0#30

    while True:


        th = max(1, th)
        original_train_label = new_train_label.copy()

        #Get the mean and covariance of each class
        distribution_params = get_distribution(pc, new_train_label)

        #Compute Bhattacharya overlap coefficient and get the adjasency matrix
        overlap_matrix = get_overlapping_region(distribution_params)
        identity_matrix = np.identity(overlap_matrix.shape[0])
        adjacency_matrix = overlap_matrix #- identity_matrix

        #Get the actual class from the clusters
        taxa_names = list(id_map.keys())
        num_classes = identity_matrix.shape[0]
        new_taxa_numbers = np.arange(num_classes)
        actual_taxa_numbers = [label_dict[k] for k in new_taxa_numbers]
        actual_taxa_names = [taxa_names[i] for i in actual_taxa_numbers]

        #Consider only overlaps that are above a threshold
        new_adjacency_matrix = adjacency_matrix.copy()
        new_adjacency_matrix[np.where(adjacency_matrix < margin)] = 0
        #new_adjacency_matrix.compress(np.all(s == 0, axis=0), axis=1)
        margin -= 0.05

        '''
        amount_initial_centers = 1
        initial_centers = kmeans_plusplus_initializer(1-new_adjacency_matrix, amount_initial_centers).initialize()
        xmeans_instance = xmeans(1-new_adjacency_matrix, initial_centers, 10)
        xmeans_instance.process()
        
        clusters = xmeans_instance.get_clusters()
        centers = xmeans_instance.get_centers()

        print(clusters)
        exit(0)
        '''

        #Create a graph from the new adjasency matrix and cluster them.
        conn_indices = np.where(new_adjacency_matrix)
        weights = new_adjacency_matrix[conn_indices]
        edges = zip(*conn_indices)
        G = ig.Graph(edges=edges, directed=True)
        #G.vs['label'] = actual_taxa_names
        G.es['weight'] = weights
        G.es['width'] = weights
        components = G.clusters(mode='weak')

        new_label_dict = {}
        final_clusters = []
        for i in range(len(components)):
            clustered_labels = components[i]
            #reduced_adj_mat = new_adjacency_matrix[clustered_labels,:][:, clustered_labels]
            mean_values = [distribution_params[x]['mean'] for x in clustered_labels]


            if len(clustered_labels) > th:
                amount_initial_centers = th #int(len(clustered_labels)/2)
                mean_values = np.array(mean_values).squeeze(1)
                dist_mat = squareform(pdist(mean_values))
                initial_centers = kmeans_plusplus_initializer(dist_mat, amount_initial_centers).initialize()
                #kmeans_instance = kmedoids(reduced_adj_mat, initial_centers)
                #xmeans_instance = kmeans(dist_mat, initial_centers, 10)
                xmeans_instance = kmeans(dist_mat, initial_centers)
                xmeans_instance.process()
        
                clusters = xmeans_instance.get_clusters()
                for c_ in clusters:
                    c_list = [clustered_labels[c_l] for c_l in c_]
                    final_clusters.append(c_list)
            else:
                final_clusters.append(clustered_labels)
                #centers = xmeans_instance.get_centers()
                #for c_ in clusters:
                #    c_list = [clustered_labels[c_l] for c_l in c_]
                #    c_names = [taxa_names[g] for g in c_list]

        final_clusters.sort()
        new_label_dict = {}
        for i in range(len(final_clusters)):
            clustered_labels = final_clusters[i]
            new_label_dict[i] = clustered_labels
            for j in clustered_labels:
                new_train_label[np.where(original_train_label == j)] = i
                
        cnt += len(final_clusters)
        
        print("New number of classes: %d" %(np.max(new_train_label) + 1))

        if np.max(original_train_label) == np.max(new_train_label):
            new_label_dict_list.append({0:list(np.arange(len(final_clusters)))})
            return new_label_dict_list
        new_label_dict_list.append(new_label_dict)  
        th -= 2  




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
    return tree



def get_key(my_dict,val):
    for key,value in my_dict.items():
        if val == value:
            return key

def get_node(nodename):
    if nodename in nodes_by_name:
        return nodes_by_name[nodename]
    else:
        nodes_by_name[nodename] = Tree(name=nodename)
        return nodes_by_name[nodename]

nodes_by_name = {}

def get_hierarchy_diatoms(file_path, id_map):
    df = pd.read_excel(file_path, engine='openpyxl')
    df = df[df['Code'].isin(id_map.keys())]

    img_classes = list(id_map.keys())
    file_classes = list(df['Code'])
    code_set = set(df['Code'])
    genre_set = set(df['Genre'])
    df['Variété'] = df['Variété'].fillna('0')
    species_set = set(df['Genre']+'_'+df['Espèce'])
    #variety_set = set(df['Genre']+'_'+df['Espèce']+'_'+df['Variété'])

    top_node = {'diatom'}
    total_set = code_set | species_set | genre_set | top_node 
    num_species = len(total_set)
    print("Total number of nodes is: ", num_species)
    print("Number of leaf nodes is: ", len(code_set))

    label_dict1 = {i:val for i,val in enumerate(sorted(code_set))}
    label_dict2 = {i+len(label_dict1):val for i,val in enumerate(sorted(species_set)) }
    label_dict3 = {i+len(label_dict1)+len(label_dict2):val for i,val in enumerate(sorted(genre_set))}
    label_dict4 = {i+len(label_dict1)+len(label_dict2)+len(label_dict3):val for i,val in enumerate(sorted(top_node))}
    label_dict = {**label_dict1, **label_dict2, **label_dict3, **label_dict4}

    label_file = open('label_file.txt', 'w')
    relation_file = open('connections.txt', 'w')
    parent_child_file = open('parent_child.txt', 'w')

    for i in range(len(label_dict)):
        label_file.write(str(i)+" "+label_dict[i]+"\n")

    for i in range(len(id_map)):
        df_row = df[df['Code'].isin([get_key(id_map, i)])]
        df_list = df_row.values
        top_id = get_key(label_dict, 'diatom')
        #print(df_list[0,2]+'_'+df_list[0,3])
        genre_id = get_key(label_dict, df_list[0,2])
        species_id = get_key(label_dict, df_list[0,2]+'_'+df_list[0,3]) #Original was [0,2]
        code_id = get_key(label_dict, df_list[0,1])
        relation_file.write(str(i)+" "+str(top_id)+" "+str(genre_id)+" "+str(species_id)+" "+str(code_id)+"\n")
        parent_child_file.write(str(top_id)+" "+str(genre_id)+"\n")
        parent_child_file.write(str(genre_id)+" "+str(species_id)+"\n")
        parent_child_file.write(str(species_id)+" "+str(code_id)+"\n")

    label_file.close()
    relation_file.close()
    parent_child_file.close()

    adj_matrix = np.zeros((num_species, num_species))

    relation_file = open('connections.txt', 'r')
    count = 0
    for line in relation_file:
        ids = line.split()
        for i in range(1,len(ids)-1):
            adj_matrix[int(ids[i]),int(ids[i+1])] = 1

        count += 1

    G = nx.from_numpy_matrix(adj_matrix, create_using=nx.DiGraph())

    uniqlines = set(open('parent_child.txt').readlines())
    parent_child_file = open('parent_child.txt', 'w').writelines(set(uniqlines))

    parent_child = open('parent_child.txt', 'r')
    for line in parent_child:
        if not line: continue
        parent_name, child_name = line.split()
        parent = get_node(parent_name.strip())
        parent.add_child(get_node(child_name.strip()))

    t = parent.get_tree_root()

    c = len(id_map)
    n_classes = len(id_map)
    for node in t.traverse("postorder"):
        if node.name == '':
            node.name = str(c)
            c += 1
        else:
            node.name = node.name

    return t

def get_taxonomy_hierarchy(file_path, id_map):
    ete_tree = get_hierarchy_diatoms(file_path, id_map)
    return ete_tree


def get_taxonomy_cub(file_path, id_map):


    #File should be the same format as parent_child.txt in https://github.com/cvjena/semantic-embeddings.git

    uniqlines = set(open(file_path).readlines())
    parent_child_file = open(file_path, 'w').writelines(set(uniqlines))

    parent_child = open(file_path, 'r')
    for line in parent_child:
        if not line: continue
        parent_name, child_name = line.split()
        parent = get_node(parent_name.strip())
        parent.add_child(get_node(child_name.strip()))

    t = parent.get_tree_root()

    c = len(id_map)
    n_classes = len(id_map)
    for node in t.traverse("postorder"):
        if node.name == '':
            node.name = str(c)
            c += 1
        else:
            node.name = node.name

    return t   
