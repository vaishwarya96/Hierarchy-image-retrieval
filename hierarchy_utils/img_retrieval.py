import torch
from scipy.spatial.distance import pdist, squareform, cosine
import numpy as np

from inference_utils import md_utils

def get_retrieved_imgs(test_emb):

 
    dist_mat = squareform(pdist(test_emb, 'euclidean'))
    ranking = np.argsort(dist_mat, axis = -1)
    gen = ((i, ret.tolist()) for i, ret in enumerate(ranking))
    gen = dict(gen)

    return gen
    


def get_retrieved_imgs_cosine(test_emb):

 
    dist_mat = squareform(pdist(test_emb, 'cosine'))
    ranking = np.argsort(dist_mat, axis = -1)
    gen = ((i, ret.tolist()) for i, ret in enumerate(ranking))
    gen = dict(gen)

    return gen

def get_retrieved_imgs_random(test_emb):

    dist_mat = squareform(pdist(test_emb, 'cosine'))
    dist_mat = np.random.rand(dist_mat.shape[0],dist_mat.shape[1])
    ranking = np.argsort(dist_mat, axis = -1)
    gen = ((i, ret.tolist()) for i, ret in enumerate(ranking))
    gen = dict(gen)

    return gen


    

class HierarchyRetrieval(object):

    def __init__(self, ete_tree, n_classes):

        object.__init__(self)
        self.heights = {}
        self.depths = {}
        self.lcs_cache = {}
        self.wup_cache = {}
        self.n_classes = n_classes
        self.t = ete_tree
        self.max_height = self.get_height(self.t)
        self.get_node_heights()
        self.get_node_depths()

    def get_height(self, n):

        leaves = n.get_leaves()
        dists = [n.get_distance(i, topology_only=True)+0 for i in leaves]
        height = max(dists)
        return height

    def get_lcs(self, a, b):
        a = str(a)
        b = str(b)
        a = self.t&a
        b = self.t&b
        lcs = self.t.get_common_ancestor(a,b)
        self.lcs_cache[(int(a.name), int(b.name))] = self.lcs_cache[(int(b.name), int(a.name))] = lcs.name


    def get_node_heights(self):

        for node in self.t.traverse("postorder"):
            self.heights[node.name] = self.get_height(node)

    def get_node_depths(self):

        for node in self.t.traverse("postorder"):
            self.depths[node.name] = self.get_depth(node)

    def get_depth(self, n):

        root_node = self.t.name
        dist = self.t.get_distance(n.name, topology_only=True)
        return dist


    def wup_similarity(self, a, b):
        
        a = str(a)
        b = str(b)
        a = self.t&a
        b = self.t&b
        #if (a,b) not in self._wup_cache:
        #lcs = self.t.get_common_ancestor(a,b)
        lcs = self.lcs_cache[(int(b.name), int(a.name))]
        ds = self.depths[lcs]
        d1 = self.depths[a.name]
        d2 = self.depths[b.name]
        self.wup_cache[(int(a.name),int(b.name))] = self.wup_cache[(int(b.name),int(a.name))] = (2.0 * ds) / (d1 + d2)
        #return (2.0 * ds) / (d1 + d2)


    def get_retrieved_imgs_hierarchy(self, test_emb, pred_class):

        dist_mat = np.empty((len(test_emb), len(test_emb)))
        for i in range(len(test_emb)):
            print(i)
            for j in range(i+1, len(test_emb)):
                sample1 = test_emb[i]
                sample2 = test_emb[j]
                lbl1 = pred_class[i]
                lbl2 = pred_class[j]
                cosine_dist = cosine(sample1, sample2)
                if (lbl1, lbl2) not in self.lcs_cache:
                    self.get_lcs(lbl1,lbl2)
                hierarchy_height = self.heights[self.lcs_cache[(lbl1,lbl2)]]/self.max_height

                #if (lbl1, lbl2) not in self.wup_cache:
                #    self.get_lcs(lbl1,lbl2)
                #    self.wup_similarity(lbl1, lbl2)
                #hierarchy_height = 1 - self.wup_cache[(lbl1, lbl2)]
                dist_mat[i,j] = dist_mat[j,i] = 0*cosine_dist + 1* hierarchy_height

        ranking = np.argsort(dist_mat, axis = -1)
        gen = ((i, ret.tolist()) for i, ret in enumerate(ranking))
        gen = dict(gen)   

        return gen


class CombinedHierarchyRetrieval(object):

    def __init__(self, visual_ete_tree, semantic_ete_tree, n_classes):

        object.__init__(self)
        self.visual_heights = {}
        self.visual_depths = {}
        self.visual_lcs_cache = {}
        self.visual_wup_cache = {}
        self.semantic_heights = {}
        self.semantic_depths = {}
        self.semantic_lcs_cache = {}
        self.semantic_wup_cache = {}       
        self.n_classes = n_classes
        self.visual_t = visual_ete_tree
        self.semantic_t = semantic_ete_tree
        self.max_visual_height = self.get_height(self.visual_t)
        self.max_semantic_height = self.get_height(self.semantic_t)
        self.get_node_heights()
        self.get_node_depths()

    def get_height(self, n):

        leaves = n.get_leaves()
        dists = [n.get_distance(i, topology_only=True)+0 for i in leaves]
        height = max(dists)
        return height

    def get_visual_lcs(self, a, b):
        a = str(a)
        b = str(b)
        a = self.visual_t&a
        b = self.visual_t&b
        lcs = self.visual_t.get_common_ancestor(a,b)
        self.visual_lcs_cache[(int(a.name), int(b.name))] = self.visual_lcs_cache[(int(b.name), int(a.name))] = lcs.name

    def get_semantic_lcs(self, a, b):
        a = str(a)
        b = str(b)
        a = self.semantic_t&a
        b = self.semantic_t&b
        lcs = self.semantic_t.get_common_ancestor(a,b)
        self.semantic_lcs_cache[(int(a.name), int(b.name))] = self.semantic_lcs_cache[(int(b.name), int(a.name))] = lcs.name


    def get_node_heights(self):

        for node in self.visual_t.traverse("postorder"):
            self.visual_heights[node.name] = self.get_height(node)
        for node in self.semantic_t.traverse("postorder"):
            self.semantic_heights[node.name] = self.get_height(node)

    def get_node_depths(self):

        for node in self.visual_t.traverse("postorder"):
            self.visual_depths[node.name] = self.get_visual_depth(node)
        for node in self.semantic_t.traverse("postorder"):
            self.semantic_depths[node.name] = self.get_semantic_depth(node)

    def get_visual_depth(self, n):

        root_node = self.visual_t.name
        dist = self.visual_t.get_distance(n.name, topology_only=True)
        return dist

    def get_semantic_depth(self, n):

        root_node = self.semantic_t.name
        dist = self.semantic_t.get_distance(n.name, topology_only=True)
        return dist


    def visual_wup_similarity(self, a, b):
        
        a = str(a)
        b = str(b)
        a = self.visual_t&a
        b = self.visual_t&b
        #if (a,b) not in self._wup_cache:
        #lcs = self.t.get_common_ancestor(a,b)
        lcs = self.visual_lcs_cache[(int(b.name), int(a.name))]
        ds = self.visual_depths[lcs]
        d1 = self.visual_depths[a.name]
        d2 = self.visual_depths[b.name]
        self.visual_wup_cache[(int(a.name),int(b.name))] = self.visual_wup_cache[(int(b.name),int(a.name))] = (2.0 * ds) / (d1 + d2)
        #return (2.0 * ds) / (d1 + d2)

    def semantic_wup_similarity(self, a, b):
        
        a = str(a)
        b = str(b)
        a = self.semantic_t&a
        b = self.semantic_t&b
        #if (a,b) not in self._wup_cache:
        #lcs = self.t.get_common_ancestor(a,b)
        lcs = self.semantic_lcs_cache[(int(b.name), int(a.name))]
        ds = self.semantic_depths[lcs]
        d1 = self.semantic_depths[a.name]
        d2 = self.semantic_depths[b.name]
        self.semantic_wup_cache[(int(a.name),int(b.name))] = self.semantic_wup_cache[(int(b.name),int(a.name))] = (2.0 * ds) / (d1 + d2)

    def get_retrieved_imgs_hierarchy(self, test_emb, pred_class):

        dist_mat = np.empty((len(test_emb), len(test_emb)))
        for i in range(len(test_emb)):
            print(i)
            for j in range(i+1, len(test_emb)):
                sample1 = test_emb[i]
                sample2 = test_emb[j]
                lbl1 = pred_class[i]
                lbl2 = pred_class[j]
                cosine_dist = cosine(sample1, sample2)
                if (lbl1, lbl2) not in self.visual_lcs_cache:
                    self.get_visual_lcs(lbl1,lbl2)
                    self.get_semantic_lcs(lbl1,lbl2)
                visual_hierarchy_height = self.visual_heights[self.visual_lcs_cache[(lbl1,lbl2)]]/self.max_visual_height
                semantic_hierarchy_height = self.semantic_heights[self.semantic_lcs_cache[(lbl1,lbl2)]]/self.max_semantic_height

                #if (lbl1, lbl2) not in self.wup_cache:
                #    self.get_lcs(lbl1,lbl2)
                #    self.wup_similarity(lbl1, lbl2)
                #hierarchy_height = 1 - self.wup_cache[(lbl1, lbl2)]
                dist_mat[i,j] = dist_mat[j,i] = cosine_dist + 1* visual_hierarchy_height + 1*semantic_hierarchy_height 

        ranking = np.argsort(dist_mat, axis = -1)
        gen = ((i, ret.tolist()) for i, ret in enumerate(ranking))
        gen = dict(gen)   

        return gen


