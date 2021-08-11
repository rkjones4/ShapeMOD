from ShapeAssembly import hier_execute
import sa_utils as utils
import torch
import os
import sys
import math
import faiss
import numpy as np
from valid import check_stability, check_rooted

device = torch.device("cuda")

class SimpChamferLoss(torch.nn.Module):
    def __init__(self, device):
        super(SimpChamferLoss, self).__init__()
        self.dimension = 3
        self.gpu_id = torch.cuda.current_device()
        self.res = faiss.StandardGpuResources()

    def build_nn_index(self, database):
        """
        :param database: numpy array of Nx3
        :return: Faiss index, in CPU
        """
        index_cpu = faiss.IndexFlatL2(self.dimension)        
        index = faiss.index_cpu_to_gpu(self.res, self.gpu_id, index_cpu)
        index.add(database)
        return index

    def search_nn(self, index, query, k):
        D, I = index.search(query, 1)
        return np.sqrt(D)

    def getAvgDist(self, index, query):
        D, I = index.search(query, 2)
        m_d = math.sqrt(np.percentile(D[:,1],90))
        return m_d
    
    def calc_metrics(self, predict_pc, gt_pc, threshes):
        """
        :param predict_pc: Bx3xM Variable in GPU
        :param gt_pc: Bx3xN Variable in GPU
        :return:
        """

        predict_pc_size = predict_pc.size()
        gt_pc_size = gt_pc.size()

        predict_pc_np = np.ascontiguousarray(
            torch.transpose(predict_pc.data.clone(), 1, 2).cpu().numpy()
        )  # BxMx3
        gt_pc_np = np.ascontiguousarray(
            torch.transpose(gt_pc.data.clone(), 1, 2).cpu().numpy()
        )  # BxNx3

    
        index_predict = self.build_nn_index(predict_pc_np[0])
        index_gt = self.build_nn_index(gt_pc_np[0])

        fwd_dist = self.search_nn(index_gt, predict_pc_np[0], 1)
        bwd_dist = self.search_nn(index_predict, gt_pc_np[0], 1)
        
        cd = (fwd_dist.mean() / 2) + (bwd_dist.mean() / 2)

        ones = np.ones(fwd_dist.shape)

        fscores = []
        for thresh in threshes:
            if thresh == 'def':
                thresh = self.getAvgDist(index_gt, gt_pc_np[0])            
            precision = (100 / ones.shape[0]) * np.sum(ones[fwd_dist <= thresh])
            recall = (100 / ones.shape[0]) * np.sum(ones[bwd_dist <= thresh])
            fs = (2*precision*recall) / (precision + recall + 1e-8)
            fscores.append(fs)
            
        return [cd] + fscores


class SimpCPUChamferLoss(torch.nn.Module):
    def __init__(self):
        super(SimpCPUChamferLoss, self).__init__()
        self.dimension = 3

    def build_nn_index(self, database):
        """
        :param database: numpy array of Nx3
        :return: Faiss index, in CPU
        """
        index = faiss.IndexFlatL2(self.dimension)        
        index.add(database)
        return index

    def search_nn(self, index, query, k):
        D, I = index.search(query, 1)
        return np.sqrt(D)

    def getAvgDist(self, index, query):
        D, I = index.search(query, 2)
        m_d = math.sqrt(np.percentile(D[:,1],90))
        return m_d
    
    def calc_metrics(self, predict_pc, gt_pc, threshes):
        """
        :param predict_pc: Bx3xM Variable in GPU
        :param gt_pc: Bx3xN Variable in GPU
        :return:
        """

        predict_pc_size = predict_pc.size()
        gt_pc_size = gt_pc.size()

        predict_pc_np = np.ascontiguousarray(
            torch.transpose(predict_pc.data.clone(), 1, 2).cpu().numpy()
        )  # BxMx3
        gt_pc_np = np.ascontiguousarray(
            torch.transpose(gt_pc.data.clone(), 1, 2).cpu().numpy()
        )  # BxNx3

    
        index_predict = self.build_nn_index(predict_pc_np[0])
        index_gt = self.build_nn_index(gt_pc_np[0])

        fwd_dist = self.search_nn(index_gt, predict_pc_np[0], 1)
        bwd_dist = self.search_nn(index_predict, gt_pc_np[0], 1)
        
        cd = (fwd_dist.mean() / 2) + (bwd_dist.mean() / 2)

        ones = np.ones(fwd_dist.shape)

        fscores = []
        for thresh in threshes:
            if thresh == 'def':
                thresh = self.getAvgDist(index_gt, gt_pc_np[0])            
            precision = (100 / ones.shape[0]) * np.sum(ones[fwd_dist <= thresh])
            recall = (100 / ones.shape[0]) * np.sum(ones[bwd_dist <= thresh])
            fs = (2*precision*recall) / (precision + recall + 1e-8)
            fscores.append(fs)
            
        return [cd] + fscores


chamfer = SimpCPUChamferLoss()
    
def getSampMetrics(verts, faces, t_samps):
    p_samps = torch.clamp(
        utils.sample_surface(faces, verts.unsqueeze(0), t_samps.shape[0], False)[0],
        -1, 1)    
            
    samp_metrics = chamfer.calc_metrics(
        p_samps.T.unsqueeze(0).float(),
        t_samps.T.unsqueeze(0).float(),
        [.05, .03, .01, 'def']
    )
    return {
        'cd': samp_metrics[0],
        'fscore-05': samp_metrics[1],
        'fscore-03': samp_metrics[2],
        'fscore-01': samp_metrics[3],
        'fscore-def': samp_metrics[4],
    }
        
def getShapeIoU(cubes, gt_cubes):
    pvoxels = shape_voxelize(cubes)
    tvoxels = shape_voxelize(gt_cubes)
    
    iou = 100 * (
        (pvoxels & tvoxels).sum().item() 
        / (pvoxels | tvoxels).sum().item()
    )        
    return iou

def recon_metrics(
    recon_sets, outpath, exp_name, name, epoch, VERBOSE, num_gen
):
    misses = 0.
    results = {
        'iou': [],
        'cd': [],
        'fscore-def': [],
        'fscore-01': [],
        'fscore-03': [],
        'fscore-05': [],
        'rooted': [],
        'stable': []
    }

    count = 0
    
    for prog, gt_prog, prog_ind, gt_pts in recon_sets:                    
        
        gt_verts, gt_faces, gt_cubes = hier_execute(gt_prog, return_all = True)
        
        try:
            verts, faces, cubes = hier_execute(prog, return_all = True)            
            assert not torch.isnan(verts).any(), 'saw nan vert'

            try:
                if check_rooted(verts, faces):
                    results['rooted'].append(1.)
                else:
                    results['rooted'].append(0.)
                
                if check_stability(verts, faces):
                    results['stable'].append(1.)
                else:
                    results['stable'].append(0.)
                    
            except Exception as e:
                print(f"Failed rooted/stable with {e} ???")
                
        except Exception as e:
            misses += 1.
            if VERBOSE:
                print(f"failed recon metrics for {prog_ind} with {e}")
            continue
                        
        gt_objs = os.listdir(f"{outpath}/{exp_name}/objs/gt/")                        
        try:
            sm = getSampMetrics(verts, faces, gt_pts)
            for k, v in sm.items():
                if v is not None:
                    results[k].append(v)
        except Exception as e:
            if VERBOSE:
                print(f"failed Samp Metrics for {prog_ind} with {e}")     
        
        if count >= num_gen:
            continue

        if f"{prog_ind}.obj" not in gt_objs:
            utils.writeObj(gt_verts, gt_faces, f"{outpath}/{exp_name}/objs/gt/{prog_ind}.obj")
            if 'dsl_prog' in gt_prog:
                utils.writeHierProg(gt_prog, 'dsl_prog', f"{outpath}/{exp_name}/programs/gt/{prog_ind}.txt")
            else:
                utils.sawriteHierProg(gt_prog, f"{outpath}/{exp_name}/programs/gt/{prog_ind}.txt")
        try:
            utils.writeObj(
                verts, faces, f"{outpath}/{exp_name}/objs/{name}/{epoch}_{prog_ind}.obj"
            )
            if 'dsl_prog' in prog:
                utils.writeHierProg(
                    prog, 'dsl_prog', f"{outpath}/{exp_name}/programs/{name}/{epoch}_{prog_ind}.txt"
                )
            else:
                utils.sawriteHierProg(
                    prog, f"{outpath}/{exp_name}/programs/{name}/{epoch}_{prog_ind}.txt"
                )
            count += 1
            
        except Exception as e:
            if VERBOSE:
                print(f"Failed writing prog/obj for {prog_ind} with {e}")
        
    for key in results:
        if len(results[key]) > 0:
            res = torch.tensor(results[key]).mean().item()
        else:
            res = 0.

        results[key] = res
        
    return results, misses


