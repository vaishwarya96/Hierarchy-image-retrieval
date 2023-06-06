import numpy as np
import types
from sklearn.metrics import average_precision_score
import ete3

class HierarchyMetrics(object):

    def __init__(self, ete_tree):

        object.__init__(self)
        self._wup_cache = {}
        self._lcs_cache = {}
        self.heights = {}
        self.t = ete_tree
        self.max_height = self.get_height(self.t)
        self.get_node_heights()

    def get_depth(self, n):

        root_node = self.t.name
        dist = self.t.get_distance(n.name, topology_only=True)
        return dist

    def get_height(self, n):

        leaves = n.get_leaves()
        dists = [n.get_distance(i, topology_only=True)+0 for i in leaves]
        height = max(dists)
        return height

    def get_node_heights(self):

        for node in self.t.traverse("postorder"):
            self.heights[node.name] = self.get_height(node)


    def wup_similarity(self, a, b):
        
        a = str(a)
        b = str(b)
        a = self.t&a
        b = self.t&b
        #if (a,b) not in self._wup_cache:
        lcs = self.t.get_common_ancestor(a,b)
        ds = self.get_depth(lcs)
        d1 = self.get_depth(a)
        d2 = self.get_depth(b)
        self._wup_cache[(int(a.name),int(b.name))] = self._wup_cache[(int(b.name),int(a.name))] = (2.0 * ds) / (d1 + d2)
        return (2.0 * ds) / (d1 + d2)
        #return self._wup_cache[(a,b)]

    def lcs_height(self, a, b):

        a = str(a)
        b = str(b)
        a = self.t&a
        b = self.t&b
        lcs = self.t.get_common_ancestor(a,b)
        lcs_height_ = self.get_height(lcs) #/ self.max_height
        return lcs_height_

    def lcs(self, a, b):

        a_tree = str(a)
        b_tree = str(b)
        a_tree = self.t&a_tree
        b_tree = self.t&b_tree
        if (a,b) not in self._lcs_cache:

            self._lcs_cache[(a,b)] = self._lcs_cache[(b,a)] = self.t.get_common_ancestor(a_tree,b_tree)

        return self._lcs_cache

    def get_lcs(self, a, b):

        a = str(a)
        b = str(b)
        a = self.t&a
        b = self.t&b

        return self.t.get_common_ancestor(a,b)


    def hierarchical_precision(self, retrieved, labels, ks=[1, 10, 50, 100], compute_ahp=False, compute_ap=False, ignore_qids=True, all_ids=None):

        if isinstance(ks, int):
            ks = [ks]
        kmax = max(ks)
        if not isinstance(compute_ahp, bool):
            kmax = max(kmax, int(compute_ahp))

        prec = { 'P@{} ({})'.format(k, type) : {} for k in ks for type in ('WUP', 'LCS_HEIGHT') }
        if compute_ahp:
            ahp_suffix = '' if isinstance(compute_ahp, bool) else '@{}'.format(compute_ahp)
            prec['AHP{} (WUP)'.format(ahp_suffix)] = {}
            prec['AHP{} (LCS_HEIGHT)'.format(ahp_suffix)] = {}
        if compute_ap:
            prec['AP'] = {}
        
        best_wup_cum = {}
        best_lcs_cum = {}


        for qid, ret in (retrieved if isinstance(retrieved, types.GeneratorType) else retrieved.items()):

            
            lbl = labels[qid]
            
            # Append missing images to the end of the ranking for proper determination of the optimal ranking
            if all_ids and (len(ret) < len(all_ids)):
                sret = set(ret)
                ret = ret + [id for id in all_ids if id not in sret]

            # Compute WUP similarity and determine optimal ranking for this label
            if (lbl not in best_wup_cum) or (compute_ahp is True):
                # We inlined the cache lookup from self.wup_similarity() here to reduce unnecessary function calls.
                wup = [self._wup_cache[(lbl, labels[r])] if (lbl, labels[r]) in self._wup_cache else self.wup_similarity(lbl, labels[r]) for r in ret]
                if lbl not in best_wup_cum:
                    best_wup_cum[lbl] = np.cumsum(sorted(wup, reverse = True))
            else:
                wup = [self._wup_cache[(lbl, labels[r])] if (lbl, labels[r]) in self._wup_cache else self.wup_similarity(lbl, labels[r]) for r in ret[:kmax+1]]
            
            # Compute LCS height based similarity and determine optimal ranking for this label
            if (lbl not in best_lcs_cum) or (compute_ahp is True):
                # We inline self.lcs_height() here to reduce function calls.
                # We also don't need to check whether the class pair is cached in self._lcs_cache, since we computed the WUP before which does that implicitly.
                #lcs = (1.0 - np.array([self.heights[self._lcs_cache[(lbl, labels[r])]] for r in ret]) / self.max_height).tolist()
                lcs = (1.0 - np.array([self.heights[self.get_lcs(lbl, labels[r]).name] for r in ret[:kmax+1]]) / self.max_height).tolist()
                if lbl not in best_lcs_cum:
                    best_lcs_cum[lbl] = np.cumsum(sorted(lcs, reverse = True))
            else:
                #lcs = (1.0 - np.array([self.heights[self._lcs_cache[(lbl, labels[r])]] for r in ret[:kmax+1]]) / self.max_height).tolist()
                lcs = (1.0 - np.array([self.heights[self.get_lcs(lbl, labels[r]).name] for r in ret[:kmax+1]]) / self.max_height).tolist()
            
            # Remove query from retrieval list
            cum_best_wup = best_wup_cum[lbl]
            cum_best_lcs = best_lcs_cum[lbl]
            if ignore_qids:
                try:
                    qid_ind = ret.index(qid)
                    if qid_ind < len(wup):
                        del wup[qid_ind]
                        del lcs[qid_ind]
                        cum_best_wup = np.concatenate((cum_best_wup[:qid_ind], cum_best_wup[qid_ind+1:] - 1.0))
                        cum_best_lcs = np.concatenate((cum_best_lcs[:qid_ind], cum_best_lcs[qid_ind+1:] - 1.0))
                except ValueError:
                    pass
            
            # Compute hierarchical precision for several cut-off points
            for k in ks:
                prec['P@{} (WUP)'.format(k)][qid]        = sum(wup[:k]) / cum_best_wup[k-1]
                prec['P@{} (LCS_HEIGHT)'.format(k)][qid] = sum(lcs[:k]) / cum_best_lcs[k-1]
            if compute_ahp:
                if isinstance(compute_ahp, bool):
                    prec['AHP (WUP)'][qid]        = np.trapz(np.cumsum(wup) / cum_best_wup, dx=1./len(wup))
                    prec['AHP (LCS_HEIGHT)'][qid] = np.trapz(np.cumsum(lcs) / cum_best_lcs, dx=1./len(lcs))
                else:
                    prec['AHP{} (WUP)'.format(ahp_suffix)][qid] = np.trapz(np.cumsum(wup[:compute_ahp]) / cum_best_wup[:compute_ahp], dx=1./compute_ahp)
                    prec['AHP{} (LCS_HEIGHT)'.format(ahp_suffix)][qid] = np.trapz(np.cumsum(lcs[:compute_ahp]) / cum_best_lcs[:compute_ahp], dx=1./compute_ahp)
            if compute_ap:
                prec['AP'][qid] = average_precision_score(
                    [labels[r] == lbl for r in ret if (not ignore_qids) or (r != qid)],
                    [-i for i, r in enumerate(ret) if (not ignore_qids) or (r != qid)]
                )
        
        return { metric : sum(values.values()) / len(values) for metric, values in prec.items() }, prec


    '''
    def get_hierarchy_metrics(self, retrieved, labels, topK_to_consider = (1, 5, 10)):
        
        kmax = max(topK_to_consider)
        labels = np.array(labels)
        num_logged = 0
        norm_mistakes_accum = 0.0
        flat_accuracy_accums = np.zeros(len(topK_to_consider), dtype=np.float)
        hdist_accums = np.zeros(len(topK_to_consider))
        hdist_top_accums = np.zeros(len(topK_to_consider))
        hdist_mistakes_accums = np.zeros(len(topK_to_consider))
        hprecision_accums = np.zeros(len(topK_to_consider))
        hmAP_accums = np.zeros(len(topK_to_consider))

        topK_hdist = np.empty([len(labels), topK_to_consider[-1]])

        best_wup_cum = {}
        best_lcs_cum = {}

        for qid, ret in retrieved.items():

            lbl = labels[qid]
            if lbl not in best_wup_cum:
                lcs = (1.0 - np.array([self.heights[self.get_lcs(lbl, labels[r]).name] for r in ret[:kmax+1]]) / self.max_height).tolist()
                best_lcs_cum[lbl] = np.cumsum(sorted(lcs, reverse = True))
            
            cum_best_lcs = best_lcs_cum[lbl]

            qid_ind = ret.index(qid)
            if qid_ind < len(lcs):

                del lcs[qid_ind]
                cum_best_lcs = np.concatenate((cum_best_lcs[:qid_ind], cum_best_lcs[qid_ind+1:] - 1.0))



        for i in range(len(labels)):
            for j in range(max(topK_to_consider)):
                class_idx_ground_truth = labels[i]
                retrieved_labels = labels[retrieved[i]][j]
                topK_hdist[i,j] = self.lcs_height(class_idx_ground_truth, retrieved_labels)
        
        mistakes_ids = np.where(topK_hdist[:, 0] != 0)[0]
        norm_mistakes_accum += len(mistakes_ids)
        topK_hdist_mistakes = topK_hdist[mistakes_ids, :]
        topK_hsimilarity = 1 - topK_hdist #/ max_dist
        topK_AP = [np.sum(topK_hsimilarity[:, :k]) / cum_best_lcs[k-1] for k in range(1, max(topK_to_consider) + 1)]
        for i in range(len(topK_to_consider)):

            hdist_accums[i] += np.mean(topK_hdist[:, : topK_to_consider[i]])
            hdist_top_accums[i] += np.mean([np.min(topK_hdist[b, : topK_to_consider[i]]) for b in range(len(labels))])
            hdist_mistakes_accums[i] += np.sum(topK_hdist_mistakes[:, : topK_to_consider[i]])
            hprecision_accums[i] += topK_AP[topK_to_consider[i] - 1]
            hmAP_accums[i] += np.mean(topK_AP[: topK_to_consider[i]])

        return hprecision_accums/len(labels), hmAP_accums/len(labels)
    '''

    def get_hierarchy_metrics(self, retrieved, labels, topK_to_consider=(1, 5, 10)):

        kmax = max(topK_to_consider)
        labels = np.array(labels)
        num_logged = 0
        norm_mistakes_accum = 0.0
        flat_accuracy_accums = np.zeros(len(topK_to_consider), dtype=np.float)
        hdist_accums = np.zeros(len(topK_to_consider))
        hdist_top_accums = np.zeros(len(topK_to_consider))
        hdist_mistakes_accums = np.zeros(len(topK_to_consider))
        hprecision_accums = np.zeros(len(topK_to_consider))
        hmAP_accums = np.zeros(len(topK_to_consider))

        topK_hdist = np.empty([len(labels), topK_to_consider[-1]])

        best_wup_cum = {}
        best_lcs_cum = {}

        for qid, ret in retrieved.items():

            lbl = labels[qid]
            if lbl not in best_wup_cum:
                lcs = (1.0 - np.array([self.heights[self.get_lcs(lbl, labels[r]).name] for r in ret[:kmax + 1]]) / self.max_height).tolist()
                best_lcs_cum[lbl] = np.cumsum(sorted(lcs, reverse=True))

            cum_best_lcs = best_lcs_cum[lbl]

            qid_ind = ret.index(qid)
            if qid_ind < len(lcs):

                del lcs[qid_ind]
                cum_best_lcs = np.concatenate((cum_best_lcs[:qid_ind], cum_best_lcs[qid_ind + 1:] - 1.0))

        for i in range(len(labels)):
            for j in range(max(topK_to_consider)):
                class_idx_ground_truth = labels[i]
                retrieved_labels = labels[retrieved[i]][j]
                topK_hdist[i, j] = self.lcs_height(class_idx_ground_truth, retrieved_labels)

        mistakes_ids = np.where(topK_hdist[:, 0] != 0)[0]
        norm_mistakes_accum += len(mistakes_ids)
        topK_hdist_mistakes = topK_hdist[mistakes_ids, :]
        topK_hsimilarity = 1 - topK_hdist  # / max_dist
        topK_AP = [np.sum(topK_hsimilarity[:, :k]) / cum_best_lcs[k - 1] for k in range(1, max(topK_to_consider) + 1)]
        for i in range(len(topK_to_consider)):

            hdist_accums[i] += np.mean(topK_hdist[:, : topK_to_consider[i]])
            hdist_top_accums[i] += np.mean([np.min(topK_hdist[b, : topK_to_consider[i]]) for b in range(len(labels))])
            hdist_mistakes_accums[i] += np.sum(topK_hdist_mistakes[:, : topK_to_consider[i]])
            hprecision_accums[i] += topK_AP[topK_to_consider[i] - 1]
            hmAP_accums[i] += np.mean(topK_AP[: topK_to_consider[i]])

        return hprecision_accums / len(labels), hmAP_accums / len(labels)
    '''
    def apk(self, actual, predicted, k):

        if isinstance(actual, int):
            actual = [actual]
        if not actual:
            return 0.0

        if len(predicted)>k:
            predicted = predicted[:k]

        score = 0.0
        num_hits = 0.0

        for i,p in enumerate(predicted):

            # first condition checks whether it is valid prediction
            # second condition checks if prediction is not repeated
            if p in [actual]:
                if p not in predicted[:i]:
                    num_hits += 1.0
                    score += num_hits / (i+1.0)

        return score / k

    def mapk(self, retrieved, labels, topK_to_consider=1):
        
        labels = np.array(labels)
        result = list(retrieved.values())
        retrieved = np.array(result)
        retrieved = labels[retrieved]
        #Need to convert query ids to label values
        map_values = []

        for a,p in zip(labels, retrieved):

            map_values.append([np.mean([self.apk(a,p,k) for a,p in zip(labels, retrieved)])])

        return map_values
    '''

    def calculate_ap(self, retrieved, labels, k):

        relevant_count = 0
        precision_sum = 0

        for i in range(k):
            if retrieved[i] == labels:
                relevant_count += 1
                precision_sum += relevant_count / (i + 1)

        ap = precision_sum / k
        return ap

    def calculate_map(self, retrieved, labels, k):

        ap_sum = 0
        query_count = 0

        labels = np.array(labels)
        result = list(retrieved.values())
        retrieved = np.array(result)
        retrieved = labels[retrieved]
        retrieved = retrieved[:,1:]

        for a,p in zip(labels, retrieved):
            ap = self.calculate_ap(p, a, k)
            ap_sum += ap
            query_count += 1
        
        map = ap_sum/query_count

        return map


                
        

        

