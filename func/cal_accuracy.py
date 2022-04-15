import numpy as np
import copy
import scipy.sparse as sparse

class IOU_and_Dice_Accuracy():
    def __init__(self, tissue_gt, tissue_pred):
        super(IOU_and_Dice_Accuracy, self).__init__()
        self.gt=tissue_gt
        self.pred=tissue_pred
    
    def cal_accuracy(self): # one cell maps to one cell
        self.pred[self.pred>0]=self.pred[self.pred>0]+np.max(self.gt)
        
        gt_unique_values = np.unique(self.gt).tolist()
        if 0 in gt_unique_values:
            gt_unique_values.remove(0)
        if -1 in gt_unique_values:
            gt_unique_values.remove(-1)
        accuracy_record_1 = -1 * np.ones((len(gt_unique_values), 2))
        value_record = -1 * np.ones((len(gt_unique_values), 1))
        for idx_gt,value in enumerate(gt_unique_values):
            print('progress: '+str((idx_gt/len(gt_unique_values)*100))+'%', end='\r')
            pos_x, pos_y, pos_z = np.where(self.gt==value)
            gt_x_min = np.min(pos_x)
            gt_x_max = np.max(pos_x)
            gt_y_min = np.min(pos_y)
            gt_y_max = np.max(pos_y)
            gt_z_min = np.min(pos_z)
            gt_z_max = np.max(pos_z)
            
            candidates = np.unique(self.pred[gt_x_min:gt_x_max, gt_y_min:gt_y_max, gt_z_min:gt_z_max]).tolist()
            if 0 in candidates:
                candidates.remove(0)
            if -1 in candidates:
                candidates.remove(-1)
            
            for idx_pred,candidate in enumerate(candidates):
                pos_x, pos_y, pos_z = np.where(self.pred==candidate)
                if idx_pred==0:
                    pred_x_min = np.min(pos_x)
                    pred_x_max = np.max(pos_x)
                    pred_y_min = np.min(pos_y)
                    pred_y_max = np.max(pos_y)
                    pred_z_min = np.min(pos_z)
                    pred_z_max = np.max(pos_z)
                else:
                    pred_x_min = np.min([pred_x_min, np.min(pos_x)])
                    pred_x_max = np.max([pred_x_max, np.max(pos_x)])
                    pred_y_min = np.min([pred_y_min, np.min(pos_y)])
                    pred_y_max = np.max([pred_y_max, np.max(pos_y)])
                    pred_z_min = np.min([pred_z_min, np.min(pos_z)])
                    pred_z_max = np.max([pred_z_max, np.max(pos_z)])
            
            x_min = np.min([gt_x_min, pred_x_min])
            x_max = np.max([gt_x_max, pred_x_max])
            y_min = np.min([gt_y_min, pred_y_min])
            y_max = np.max([gt_y_max, pred_y_max])
            z_min = np.min([gt_z_min, pred_z_min])
            z_max = np.max([gt_z_max, pred_z_max])
            
            gt_crop=copy.deepcopy(self.gt[x_min:x_max, y_min:y_max, z_min:z_max])
            pred_crop=copy.deepcopy(self.pred[x_min:x_max, y_min:y_max, z_min:z_max])
            
            masks_gt_crop = np.expand_dims(gt_crop==value, axis=3)
            masks_pred_crop = self.get_masks(pred_crop, candidates)
            
            overlaps, dices = self.compute_overlaps_masks(masks_gt_crop, masks_pred_crop)
            idx = np.argmax(overlaps)

            iou = overlaps[0,idx]
            dice = dices[0,idx]
            
            accuracy_record_1[idx_gt,:] = iou, dice
            value_record[idx_gt,:] = value
        
        accuracy_record_2 = -1 * np.ones((len(gt_unique_values), 4))
        used_candidates_record = []
        
        value_descending_order=np.argsort(1-accuracy_record_1[:,0])
        del accuracy_record_1
        for idx_gt,value in enumerate(value_record[value_descending_order]):
            print('progress: '+str((idx_gt/len(value_record)*100))+'%', end='\r')
            pos_x, pos_y, pos_z = np.where(self.gt==value)
            gt_x_min = np.min(pos_x)
            gt_x_max = np.max(pos_x)
            gt_y_min = np.min(pos_y)
            gt_y_max = np.max(pos_y)
            gt_z_min = np.min(pos_z)
            gt_z_max = np.max(pos_z)
            
            candidates = np.unique(self.pred[gt_x_min:gt_x_max, gt_y_min:gt_y_max, gt_z_min:gt_z_max]).tolist()
            if 0 in candidates:
                candidates.remove(0)
            if -1 in candidates:
                candidates.remove(-1)
            if used_candidates_record!=[]:
                for item in used_candidates_record:
                    if item in candidates:
                        candidates.remove(item)
            
            for idx_pred,candidate in enumerate(candidates):
                pos_x, pos_y, pos_z = np.where(self.pred==candidate)
                if idx_pred==0:
                    pred_x_min = np.min(pos_x)
                    pred_x_max = np.max(pos_x)
                    pred_y_min = np.min(pos_y)
                    pred_y_max = np.max(pos_y)
                    pred_z_min = np.min(pos_z)
                    pred_z_max = np.max(pos_z)
                else:
                    pred_x_min = np.min([pred_x_min, np.min(pos_x)])
                    pred_x_max = np.max([pred_x_max, np.max(pos_x)])
                    pred_y_min = np.min([pred_y_min, np.min(pos_y)])
                    pred_y_max = np.max([pred_y_max, np.max(pos_y)])
                    pred_z_min = np.min([pred_z_min, np.min(pos_z)])
                    pred_z_max = np.max([pred_z_max, np.max(pos_z)])
            
            x_min = np.min([gt_x_min, pred_x_min])
            x_max = np.max([gt_x_max, pred_x_max])
            y_min = np.min([gt_y_min, pred_y_min])
            y_max = np.max([gt_y_max, pred_y_max])
            z_min = np.min([gt_z_min, pred_z_min])
            z_max = np.max([gt_z_max, pred_z_max])
            
            gt_crop=copy.deepcopy(self.gt[x_min:x_max, y_min:y_max, z_min:z_max])
            pred_crop=copy.deepcopy(self.pred[x_min:x_max, y_min:y_max, z_min:z_max])
            
            if candidates!=[]:
                masks_gt_crop = np.expand_dims(gt_crop==value, axis=3)
                masks_pred_crop = self.get_masks(pred_crop, candidates)

                overlaps, dices = self.compute_overlaps_masks(masks_gt_crop, masks_pred_crop)
                idx = np.argmax(dices)

                iou = overlaps[0,idx]
                dice = dices[0,idx]
                used_candidates_record.append(candidates[idx])

                accuracy_record_2[idx_gt,:] = candidates[idx], value, iou, dice
            else:
                accuracy_record_2[idx_gt,:] = -1, value, 0, 0
        
        for i in np.arange(accuracy_record_2.shape[0]):
            if accuracy_record_2[i,0]!=-1:
                self.pred[self.pred==accuracy_record_2[i,0]]=accuracy_record_2[i,1]
            
        return accuracy_record_2[:,1:]
    
    def cal_accuracy_II(self): #not necessary one cell maps to one cell
        self.pred[self.pred>0]=self.pred[self.pred>0]+np.max(self.gt)
        
        gt_unique_values = np.unique(self.gt).tolist()
        if 0 in gt_unique_values:
            gt_unique_values.remove(0)
        if -1 in gt_unique_values:
            gt_unique_values.remove(-1)
        accuracy_record = -1 * np.ones((len(gt_unique_values), 4))
        for idx_gt,value in enumerate(gt_unique_values):
            print('progress: '+str((idx_gt/len(gt_unique_values)*100))+'%', end='\r')
            pos_x, pos_y, pos_z = np.where(self.gt==value)
            gt_x_min = np.min(pos_x)
            gt_x_max = np.max(pos_x)
            gt_y_min = np.min(pos_y)
            gt_y_max = np.max(pos_y)
            gt_z_min = np.min(pos_z)
            gt_z_max = np.max(pos_z)
            
            candidates = np.unique(self.pred[gt_x_min:gt_x_max, gt_y_min:gt_y_max, gt_z_min:gt_z_max]).tolist()
            if 0 in candidates:
                candidates.remove(0)
            if -1 in candidates:
                candidates.remove(-1)
            
            for idx_pred,candidate in enumerate(candidates):
                pos_x, pos_y, pos_z = np.where(self.pred==candidate)
                if idx_pred==0:
                    pred_x_min = np.min(pos_x)
                    pred_x_max = np.max(pos_x)
                    pred_y_min = np.min(pos_y)
                    pred_y_max = np.max(pos_y)
                    pred_z_min = np.min(pos_z)
                    pred_z_max = np.max(pos_z)
                else:
                    pred_x_min = np.min([pred_x_min, np.min(pos_x)])
                    pred_x_max = np.max([pred_x_max, np.max(pos_x)])
                    pred_y_min = np.min([pred_y_min, np.min(pos_y)])
                    pred_y_max = np.max([pred_y_max, np.max(pos_y)])
                    pred_z_min = np.min([pred_z_min, np.min(pos_z)])
                    pred_z_max = np.max([pred_z_max, np.max(pos_z)])
            
            x_min = np.min([gt_x_min, pred_x_min])
            x_max = np.max([gt_x_max, pred_x_max])
            y_min = np.min([gt_y_min, pred_y_min])
            y_max = np.max([gt_y_max, pred_y_max])
            z_min = np.min([gt_z_min, pred_z_min])
            z_max = np.max([gt_z_max, pred_z_max])
            
            gt_crop=copy.deepcopy(self.gt[x_min:x_max, y_min:y_max, z_min:z_max])
            pred_crop=copy.deepcopy(self.pred[x_min:x_max, y_min:y_max, z_min:z_max])
            
            if candidates!=[]:
                masks_gt_crop = np.expand_dims(gt_crop==value, axis=3)
                masks_pred_crop = self.get_masks(pred_crop, candidates)

                overlaps, dices = self.compute_overlaps_masks(masks_gt_crop, masks_pred_crop)
                idx = np.argmax(overlaps)

                iou = overlaps[0,idx]
                dice = dices[0,idx]

                accuracy_record[idx_gt,:] = candidates[idx], value, iou, dice
            else:
                accuracy_record[idx_gt,:] = -1, value, 0, 0
        
        for i in np.arange(accuracy_record.shape[0]):
            if accuracy_record[i,0]!=-1:
                self.pred[self.pred==accuracy_record[i,0]]=accuracy_record[i,1]
            
        return accuracy_record[:,1:]
    
    def get_masks(self, tissue, candidates):
        """Generate masks for each single cell given the 3d segmentation results
        Input: tissue: [Height, Width, Depth]  dtype=int
        Output: masks: [Height, Width, Depth, instances]   dtype=boolean"""
        data = np.array(tissue)
        x,y,z = data.shape
        count = len(candidates)
        masks = np.empty((x,y,z,count), dtype=np.bool_)
        for i, candidate in enumerate(candidates):
            masks[:,:,:,i] = (data==candidate)
        return masks
    
    def compute_overlaps_masks(self, masks1, masks2):
        """  Computes IoU overlaps between two sets of masks.
        params:
        masks1, masks2: [Height, Width, Depth, instances]  dtype=boolean
        overlaps: [instances1, instances2]  dtype=float
        dices: [instances1, instances2]  dtype=float
        """
        # If either set of masks is empty return empty result
        if masks1.shape[-1] == 0 or masks2.shape[-1] == 0:
            return np.zeros((masks1.shape[-1], masks2.shape[-1]))
        # flatten masks and compute their areas
        masks1 = np.reshape(masks1, (-1, masks1.shape[-1])).astype(np.float32)
        masks2 = np.reshape(masks2, (-1, masks2.shape[-1])).astype(np.float32)
        area1 = np.sum(masks1, axis=0)
        area2 = np.sum(masks2, axis=0)

        # intersections and union
        intersections = np.dot(masks1.T, masks2)
        union = area1[:, None] + area2[None, :] - intersections
        overlaps = intersections / union
        dices = 2 * intersections / (union + intersections)
        
        return overlaps, dices

# Evaluation code courtesy of Juan Nunez-Iglesias, taken from
# https://github.com/janelia-flyem/gala/blob/master/gala/evaluate.py

def VOI(reconstruction, groundtruth, ignore_reconstruction=[0], ignore_groundtruth=[0]):
    """Return the conditional entropies of the variation of information metric. [1]
    Let X be a reconstruction, and Y a ground truth labelling. The variation of
    information between the two is the sum of two conditional entropies:
        VI(X, Y) = H(X|Y) + H(Y|X).
    The first one, H(X|Y), is a measure of oversegmentation, the second one,
    H(Y|X), a measure of undersegmentation. These measures are referred to as
    the variation of information split or merge error, respectively.
    Parameters
    ----------
    seg : np.ndarray, int type, arbitrary shape
        A candidate segmentation.
    gt : np.ndarray, int type, same shape as `seg`
        The ground truth segmentation.
    ignore_seg, ignore_gt : list of int, optional
        Any points having a label in this list are ignored in the evaluation.
        By default, only the label 0 in the ground truth will be ignored.
    Returns
    -------
    (split, merge) : float
        The variation of information split and merge error, i.e., H(X|Y) and H(Y|X)
    References
    ----------
    [1] Meila, M. (2007). Comparing clusterings - an information based
    distance. Journal of Multivariate Analysis 98, 873-895.
    """
    (hyxg, hxgy) = split_vi(reconstruction, groundtruth, ignore_reconstruction, ignore_groundtruth)
    return (hxgy, hyxg)


def split_vi(x, y=None, ignore_x=[0], ignore_y=[0]):
    """Return the symmetric conditional entropies associated with the VI.
    The variation of information is defined as VI(X,Y) = H(X|Y) + H(Y|X).
    If Y is the ground-truth segmentation, then H(Y|X) can be interpreted
    as the amount of under-segmentation of Y and H(X|Y) is then the amount
    of over-segmentation.  In other words, a perfect over-segmentation
    will have H(Y|X)=0 and a perfect under-segmentation will have H(X|Y)=0.
    If y is None, x is assumed to be a contingency table.
    Parameters
    ----------
    x : np.ndarray
        Label field (int type) or contingency table (float). `x` is
        interpreted as a contingency table (summing to 1.0) if and only if `y`
        is not provided.
    y : np.ndarray of int, same shape as x, optional
        A label field to compare to `x`.
    ignore_x, ignore_y : list of int, optional
        Any points having a label in this list are ignored in the evaluation.
        Ignore 0-labeled points by default.
    Returns
    -------
    sv : np.ndarray of float, shape (2,)
        The conditional entropies of Y|X and X|Y.
    See Also
    --------
    vi
    """
    _, _, _ , hxgy, hygx, _, _ = vi_tables(x, y, ignore_x, ignore_y)
    # false merges, false splits
    return np.array([hygx.sum(), hxgy.sum()])


def vi_tables(x, y=None, ignore_x=[0], ignore_y=[0]):
    """Return probability tables used for calculating VI.
    If y is None, x is assumed to be a contingency table.
    Parameters
    ----------
    x, y : np.ndarray
        Either x and y are provided as equal-shaped np.ndarray label fields
        (int type), or y is not provided and x is a contingency table
        (sparse.csc_matrix) that may or may not sum to 1.
    ignore_x, ignore_y : list of int, optional
        Rows and columns (respectively) to ignore in the contingency table.
        These are labels that are not counted when evaluating VI.
    Returns
    -------
    pxy : sparse.csc_matrix of float
        The normalized contingency table.
    px, py, hxgy, hygx, lpygx, lpxgy : np.ndarray of float
        The proportions of each label in `x` and `y` (`px`, `py`), the
        per-segment conditional entropies of `x` given `y` and vice-versa, the
        per-segment conditional probability p log p.
    """
    if y is not None:
        pxy = contingency_table(x, y, ignore_x, ignore_y)
    else:
        cont = x
        total = float(cont.sum())
        # normalize, since it is an identity op if already done
        pxy = cont / total

    # Calculate probabilities
    px = np.array(pxy.sum(axis=1)).ravel()
    py = np.array(pxy.sum(axis=0)).ravel()
    # Remove zero rows/cols
    nzx = px.nonzero()[0]
    nzy = py.nonzero()[0]
    nzpx = px[nzx]
    nzpy = py[nzy]
    nzpxy = pxy[nzx, :][:, nzy]

    # Calculate log conditional probabilities and entropies
    lpygx = np.zeros(np.shape(px))
    lpygx[nzx] = xlogx(divide_rows(nzpxy, nzpx)).sum(axis=1).ravel()
                        # \sum_x{p_{y|x} \log{p_{y|x}}}
    hygx = -(px*lpygx) # \sum_x{p_x H(Y|X=x)} = H(Y|X)

    lpxgy = np.zeros(np.shape(py))
    lpxgy[nzy] = xlogx(divide_columns(nzpxy, nzpy)).sum(axis=0).ravel()
    hxgy = -(py*lpxgy)

    return [pxy] + list(map(np.asarray, [px, py, hxgy, hygx, lpygx, lpxgy]))


def contingency_table(seg, gt, ignore_seg=[0], ignore_gt=[0], norm=True):
    """Return the contingency table for all regions in matched segmentations.
    Parameters
    ----------
    seg : np.ndarray, int type, arbitrary shape
        A candidate segmentation.
    gt : np.ndarray, int type, same shape as `seg`
        The ground truth segmentation.
    ignore_seg : list of int, optional
        Values to ignore in `seg`. Voxels in `seg` having a value in this list
        will not contribute to the contingency table. (default: [0])
    ignore_gt : list of int, optional
        Values to ignore in `gt`. Voxels in `gt` having a value in this list
        will not contribute to the contingency table. (default: [0])
    norm : bool, optional
        Whether to normalize the table so that it sums to 1.
    Returns
    -------
    cont : scipy.sparse.csc_matrix
        A contingency table. `cont[i, j]` will equal the number of voxels
        labeled `i` in `seg` and `j` in `gt`. (Or the proportion of such voxels
        if `norm=True`.)
    """
    segr = seg.ravel()
    gtr = gt.ravel()
    ignored = np.zeros(segr.shape, np.bool)
    data = np.ones(len(gtr))
    for i in ignore_seg:
        ignored[segr == i] = True
    for j in ignore_gt:
        ignored[gtr == j] = True
    data[ignored] = 0
    cont = sparse.coo_matrix((data, (segr, gtr))).tocsc()
    if norm:
        cont /= float(cont.sum())
    return cont


def divide_columns(matrix, row, in_place=False):
    """Divide each column of `matrix` by the corresponding element in `row`.
    The result is as follows: out[i, j] = matrix[i, j] / row[j]
    Parameters
    ----------
    matrix : np.ndarray, scipy.sparse.csc_matrix or csr_matrix, shape (M, N)
        The input matrix.
    column : a 1D np.ndarray, shape (N,)
        The row dividing `matrix`.
    in_place : bool (optional, default False)
        Do the computation in-place.
    Returns
    -------
    out : same type as `matrix`
        The result of the row-wise division.
    """
    if in_place:
        out = matrix
    else:
        out = matrix.copy()
    if type(out) in [sparse.csc_matrix, sparse.csr_matrix]:
        if type(out) == sparse.csc_matrix:
            convert_to_csc = True
            out = out.tocsr()
        else:
            convert_to_csc = False
        row_repeated = np.take(row, out.indices)
        nz = out.data.nonzero()
        out.data[nz] /= row_repeated[nz]
        if convert_to_csc:
            out = out.tocsc()
    else:
        out /= row[np.newaxis, :]
    return out


def divide_rows(matrix, column, in_place=False):
    """Divide each row of `matrix` by the corresponding element in `column`.
    The result is as follows: out[i, j] = matrix[i, j] / column[i]
    Parameters
    ----------
    matrix : np.ndarray, scipy.sparse.csc_matrix or csr_matrix, shape (M, N)
        The input matrix.
    column : a 1D np.ndarray, shape (M,)
        The column dividing `matrix`.
    in_place : bool (optional, default False)
        Do the computation in-place.
    Returns
    -------
    out : same type as `matrix`
        The result of the row-wise division.
    """
    if in_place:
        out = matrix
    else:
        out = matrix.copy()
    if type(out) in [sparse.csc_matrix, sparse.csr_matrix]:
        if type(out) == sparse.csr_matrix:
            convert_to_csr = True
            out = out.tocsc()
        else:
            convert_to_csr = False
        column_repeated = np.take(column, out.indices)
        nz = out.data.nonzero()
        out.data[nz] /= column_repeated[nz]
        if convert_to_csr:
            out = out.tocsr()
    else:
        out /= column[:, np.newaxis]
    return out


def xlogx(x, out=None, in_place=False):
    """Compute x * log_2(x).
    We define 0 * log_2(0) = 0
    Parameters
    ----------
    x : np.ndarray or scipy.sparse.csc_matrix or csr_matrix
        The input array.
    out : same type as x (optional)
        If provided, use this array/matrix for the result.
    in_place : bool (optional, default False)
        Operate directly on x.
    Returns
    -------
    y : same type as x
        Result of x * log_2(x).
    """
    if in_place:
        y = x
    elif out is None:
        y = x.copy()
    else:
        y = out
    if type(y) in [sparse.csc_matrix, sparse.csr_matrix]:
        z = y.data
    else:
        z = y
    nz = z.nonzero()
    z[nz] *= np.log2(z[nz])
    return y

def adapted_rand(seg, gt, all_stats=False):
    """Compute Adapted Rand error as defined by the SNEMI3D contest [1]
    Formula is given as 1 - the maximal F-score of the Rand index
    (excluding the zero component of the original labels). Adapted
    from the SNEMI3D MATLAB script, hence the strange style.
    Parameters
    ----------
    seg : np.ndarray
        the segmentation to score, where each value is the label at that point
    gt : np.ndarray, same shape as seg
        the groundtruth to score against, where each value is a label
    all_stats : boolean, optional
        whether to also return precision and recall as a 3-tuple with rand_error
    Returns
    -------
    are : float
        The adapted Rand error; equal to $1 - \frac{2pr}{p + r}$,
        where $p$ and $r$ are the precision and recall described below.
    prec : float, optional
        The adapted Rand precision. (Only returned when `all_stats` is ``True``.)
    rec : float, optional
        The adapted Rand recall.  (Only returned when `all_stats` is ``True``.)
    References
    ----------
    [1]: http://brainiac2.mit.edu/SNEMI3D/evaluation
    """
    # Evaluation code courtesy of Juan Nunez-Iglesias, taken from
    # https://github.com/janelia-flyem/gala/blob/master/gala/evaluate.py

    # segA is truth, segB is query
    segA = np.ravel(gt)
    segB = np.ravel(seg)
    n = segA.size

    n_labels_A = int(np.amax(segA) + 1)
    n_labels_B = int(np.amax(segB) + 1)

    ones_data = np.ones(n)

    p_ij = sparse.csr_matrix((ones_data, (segA[:], segB[:])), shape=(n_labels_A, n_labels_B))

    a = p_ij[1:n_labels_A, :]
    b = p_ij[1:n_labels_A, 1:n_labels_B]
    c = p_ij[1:n_labels_A, 0].todense()
    d = b.multiply(b)

    a_i = np.array(a.sum(1))
    b_i = np.array(b.sum(0))

    sumA = np.sum(a_i * a_i)
    sumB = np.sum(b_i * b_i) + (np.sum(c) / n)
    sumAB = np.sum(d) + (np.sum(c) / n)

    precision = sumAB / sumB
    recall = sumAB / sumA

    fScore = 2.0 * precision * recall / (precision + recall)
    are = 1.0 - fScore

    if all_stats:
        return (are, precision, recall)
    else:
        return are