import numpy as np
import torch
import ete3
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, zero_one_loss, precision_recall_fscore_support

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

def get_adaptive_hierarchy_sets(t, pred_prob, pred, target, label_dict, height_of_tree):

    tree_levels = int(height_of_tree + 1)
    nodes_at_each_level = [[] for i in range(int(height_of_tree))]

    for node in t.traverse("postorder"):
        dist = height_of_tree - t.get_distance(node, topology_only=True) -1
        nodes_at_each_level[int(dist)].append(node.name)
    
    nodes_at_each_level = [sorted(list(set(l))) for l in nodes_at_each_level]

    for i in range(pred.shape[0]):
        probs = pred_prob[i]

        #Assign weights to each node
        for node in t.traverse("postorder"):
            leaves = node.get_leaves()
            leaf_prob = []
            for leaf in leaves:
                leaf_prob.append(probs[int(leaf.name)])
            node.weight = min(leaf_prob)      

        for j in range(len(nodes_at_each_level)):
            node_list = nodes_at_each_level[j]
            node_weight = [(t&node).weight for node in node_list]
            
            #Select the weights that are greater than 0.99
            #If more than two elements have weight greater than this value, go to higher level
            #Otherwise break the loop and select the node corresponding to the largest weight
            #Get the 

           
    

def get_temp_adaptive_sets(t, pred_prob, pred, target, label_dict, height_of_tree):

    sorted_prob = np.sort(pred_prob, axis=1)[:,::-1]
    ambiguous = (sorted_prob[:,0] - sorted_prob[:,1])<0.70
    
    hierarchy_gt_path = []
    hierarchy_pred_path = []

    for i in range(pred.shape[0]):
        #if ambiguous[i]:
        #    gt_node_path, pred_node_path = get_hierarchy_predictions(t, target[i], pred[i], label_dict, height_of_tree)
        #    hierarchy_gt_path.append(gt_node_path)
        #    hierarchy_pred_path.append(pred_node_path)
        
        #if sorted_prob[i,0] < 0.99:
        #    continue
        #else:
        #    gt_node_path, pred_node_path = [str(pred[i])]*int(height_of_tree+1), [str(target[i])]*int(height_of_tree+1)#get_hierarchy_predictions(t, target[i], pred[i], label_dict, 1)
        #    hierarchy_gt_path.append(gt_node_path)
        #    hierarchy_pred_path.append(pred_node_path)

        if sorted_prob[i,0] > 0.90:
            #gt_node_path, pred_node_path = [str(pred[i])]*int(height_of_tree+1), [str(target[i])]*int(height_of_tree+1)#get_hierarchy_predictions(t, target[i], pred[i], label_dict, 1)
            #hierarchy_gt_path.append(gt_node_path)
            #hierarchy_pred_path.append(pred_node_path)
            gt_node_path, pred_node_path = get_hierarchy_predictions(t, target[i], pred[i], label_dict, height_of_tree)
            hierarchy_gt_path.append(gt_node_path)
            hierarchy_pred_path.append(pred_node_path)
            
        else:
            #gt_node_path, pred_node_path = get_hierarchy_predictions(t, target[i], pred[i], label_dict, height_of_tree)
            #hierarchy_gt_path.append(gt_node_path)
            #hierarchy_pred_path.append(pred_node_path)
            continue

        
    return hierarchy_gt_path, hierarchy_pred_path




def get_adaptive_sets(t, pred_prob, pred, target, label_dict, height_of_tree):


    tree_levels = int(height_of_tree + 1)
    prob_level = pred_prob.copy()

    
    length = []
    correct = 0
    cnt = 0
    for i in range(pred.shape[0]):
        probs = prob_level[i]
        sorted_prob = np.sort(probs)[::-1]
        indices = np.argwhere(probs > 0.99).tolist()
        if len(indices) > 0:
            cnt += 1
        if target[i] in indices:
            correct += 1
        count = sum(probs > 0.99)
        length.append(count)
    print("The new set size is ", np.array(length).mean() )
    print("accuracy is ", correct/cnt)
    
    '''
        
        for node in t.traverse("postorder"):
            leaves = node.get_leaves()
            leaf_prob = []
            for leaf in leaves:
                leaf_prob.append(probs[int(leaf.name)])
                node.weight = min(leaf_prob) 

    '''
        
        #for j in range(height_of_tree):

    #if (sorted_prob[0] - sorted_prob[1]) < 0.05:



        #        pred_class = np.argmax(probs)
        #        pred_node = t&str(pred_class)

        #else:
        #    break
            
        
        

    '''
    ambiguous = (pred_prob[:,0] - pred_prob[:,1])<0.05

    hierarchy_gt_path = []
    hierarchy_pred_path = []

    for i in range(pred.shape[0]):
        if ambiguous[i]:
            gt_node_path, pred_node_path = get_hierarchy_predictions(t, target[i], pred[i], label_dict, height_of_tree)
            hierarchy_gt_path.append(gt_node_path)
            hierarchy_pred_path.append(pred_node_path)
        else:
            gt_node_path, pred_node_path = [str(pred[i])]*int(height_of_tree+1), [str(target[i])]*int(height_of_tree+1)#get_hierarchy_predictions(t, target[i], pred[i], label_dict, 1)
            hierarchy_gt_path.append(gt_node_path)
            hierarchy_pred_path.append(pred_node_path)
        
    return hierarchy_gt_path, hierarchy_pred_path
    '''

def get_hierarchy_metrics(output, target, label_dict, t):

    ks=(1,2,3)
    topK_accuracies, topK_predicted_classes = get_topk_accuracy(output, target, label_dict, ks=ks)

    for leaf in t:
        height_of_tree = t.get_distance('1', topology_only=True)
        break
    pred = np.argmax(output, axis=1)
    pred_prob = np.sort(output, axis=1)[:,::-1]
    pred_class = np.argsort(output, axis=1)[:,::-1]
    hierarchy_gt_path = []
    hierarchy_pred_path = []
    
    for i in range(pred.shape[0]):
        gt_node_path, pred_node_path = get_hierarchy_predictions(t, target[i], pred[i], label_dict, height_of_tree)
        #print("GT: ", gt_node_path)
        #print("Pred: ", pred_node_path)
        #print("Prob: ", pred_prob[i,:3])
        #print("Pred class: ", pred_class[i,:3])
        hierarchy_gt_path.append(gt_node_path)
        hierarchy_pred_path.append(pred_node_path)

    h_accuracy_layer = []
    h_prec_layer = []
    h_rec_layer = []
    h_f_layer = []
    h_conf_mat = []


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
            

        #conf_mat = [confusion_matrix(np.array(gt_layer), np.array(pred_layer))]
        #h_conf_mat.extend(conf_mat)
        h_accuracy_layer.extend([accuracy_score(np.array(gt_layer), np.array(pred_layer))])
        #h_prec_layer.extend([precision_score(np.array(gt_layer), np.array(pred_layer), average='macro')])
        #h_rec_layer.extend([recall_score(np.array(gt_layer), np.array(pred_layer), average='macro')])
        #h_f_layer.extend([f1_score(np.array(gt_layer), np.array(pred_layer), average='macro')])


    print("Each hierarchy level ", h_accuracy_layer)
    get_adaptive_sets(t, output, pred, target, label_dict, height_of_tree)

    hierarchy_gt_path, hierarchy_pred_path = get_temp_adaptive_sets(t, output, pred, target, label_dict, height_of_tree)


    '''
    hierarchy_gt_path, hierarchy_pred_path = get_adaptive_hierarchy_sets(t, output, pred, target, label_dict, height_of_tree)
    '''
    #accuracy_levels = get_accuracy_at_each_level(ete_tree, target, topK_predicted_classes, 1)
    h_accuracy_layer = []
    h_prec_layer = []
    h_rec_layer = []
    h_f_layer = []
    h_conf_mat = []


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
            

        #conf_mat = [confusion_matrix(np.array(gt_layer), np.array(pred_layer))]
        #h_conf_mat.extend(conf_mat)
        h_accuracy_layer.extend([accuracy_score(np.array(gt_layer), np.array(pred_layer))])
        #h_prec_layer.extend([precision_score(np.array(gt_layer), np.array(pred_layer), average='macro')])
        #h_rec_layer.extend([recall_score(np.array(gt_layer), np.array(pred_layer), average='macro')])
        #h_f_layer.extend([f1_score(np.array(gt_layer), np.array(pred_layer), average='macro')])


    print("Each hierarchy level ", h_accuracy_layer)   


    return topK_accuracies




