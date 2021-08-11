from ShapeAssembly import hier_execute, ShapeAssembly, make_hier_prog
import sa_utils as utils
import torch
import os
import sys
from valid import check_stability, check_rooted
from pointnet_fd import get_fd
from recon_metrics import chamfer, CD_MULT
from tqdm import tqdm
import faiss
import numpy as np

NUM_SAMPS = 2500
CD_NUM_SAMPS = 1024
device = torch.device("cuda")
#device = torch.device("cpu")
MAX_ROOTED_STABLE = 200
MAX_VARI = 200

sa = ShapeAssembly()

class CDPairs(torch.nn.Module):
    def __init__(self, device, mem = 100 * 1024 * 1024):
        super(CDPairs, self).__init__()
        self.gpu_id = torch.cuda.current_device()
        self.res = faiss.StandardGpuResources()
        self.res.noTempMemory()
        self.res.setTempMemory(mem)
        self.dimension = 3
        
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
        D, I = index.search(query, k)
        return np.sqrt(D)

    # B x D x 3 inputs
    def calc_cd(self, source, target, skip_same=False):
        source = np.ascontiguousarray(source.cpu().numpy())
        target = np.ascontiguousarray(target.cpu().numpy())

        d_source = []
        d_target = []
        
        for i in range(source.shape[0]):
            d_source.append(self.build_nn_index(source[i]))
        
        for i in range(target.shape[0]):
            d_target.append(self.build_nn_index(target[i]))

        min_dists = []
        for i in tqdm(range(source.shape[0])):
            min_dist = 1e8
            for j in range(target.shape[0]):
                if i == j and skip_same:
                    continue
                fwd_dist = self.search_nn(d_target[j], source[i], 1)
                bwd_dist = self.search_nn(d_source[i], target[j], 1)
                dist = fwd_dist.mean() + bwd_dist.mean()
                if dist < min_dist:
                    min_dist = dist
            
            min_dists.append(min_dist)

        return torch.tensor(min_dists).mean().item()

cdpairs = CDPairs(device)
    
def getGTSamples(progs, num):
    data = []
    for prog in tqdm(progs[:num]):
        verts, faces = hier_execute(prog)
        verts = verts.to(device)
        faces = faces.to(device)
        try:
            samps = utils.sample_surface(faces, verts.unsqueeze(0), NUM_SAMPS, False).squeeze()
            data.append(samps)
        except Exception as e:
            print(f"Failed GT samples with {e}")
    return data
        
def getSamples(meshes, VERBOSE, num_samps = NUM_SAMPS):
    data = []
    for verts, faces in tqdm(meshes):
        verts = verts.to(device)
        faces = faces.to(device)
        try:
            samps = utils.sample_surface(faces, verts.unsqueeze(0), num_samps, False).squeeze()
            data.append(samps)
        except Exception as e:
            if VERBOSE:
                print(f"couldn't sample with {e}")
    return data

# Helper function for keeping consistent train/val splits
def getInds(train_ind_file):
    inds = set()
    with open(train_ind_file) as f:
        for line in f:
            inds.add(line.strip())
    return inds

def get_gt_surf_samples(category, num = 1000):

    train_ind_file = f'data_splits/{category}/train.txt'
    val_ind_file = f'data_splits/{category}/val.txt'

    os.system('mkdir gen_comps')
    os.system('mkdir gen_comps/gt_val')
    os.system('mkdir gen_comps/gt_train/')
    os.system(f'mkdir gen_comps/gt_val/{category}')
    os.system(f'mkdir gen_comps/gt_train/{category}')
    
    gt_train_files = set(os.listdir(f'gen_comps/gt_train/{category}/'))
    gt_val_files = set(os.listdir(f'gen_comps/gt_val/{category}/'))
    
    train_samples = []
    val_samples = []

    train_inds = list(getInds(train_ind_file))
    val_inds = list(getInds(val_ind_file))

    train_samples = []
    val_samples = []
    
    for ind in tqdm(train_inds[:num]):
        if ind+'.obj' in gt_train_files:
            verts, faces = utils.loadObj(f'gen_comps/gt_train/{category}/{ind}.obj')
            verts = torch.tensor(verts).float().to(device)
            faces = torch.tensor(faces).long().to(device)
        else:
            lines = sa.load_lines(f'data/{category}/{ind}.txt')
            hier_prog = make_hier_prog(lines)
            verts, faces = hier_execute(hier_prog)
            utils.writeObj(verts, faces, f'gen_comps/gt_train/{category}/{ind}.obj')
        train_samples.append((verts, faces))
    
    for ind in tqdm(val_inds[:num]):
        if ind+'.obj' in gt_val_files:
            verts, faces = utils.loadObj(f'gen_comps/gt_val/{category}/{ind}.obj')
            verts = torch.tensor(verts).float().to(device)
            faces = torch.tensor(faces).long().to(device)
        else:
            lines = sa.load_lines(f'data/{category}/{ind}.txt')
            hier_prog = make_hier_prog(lines)
            verts, faces = hier_execute(hier_prog)
            utils.writeObj(verts, faces, f'gen_comps/gt_val/{category}/{ind}.obj')
        val_samples.append((verts, faces))
    

    train_surf_samples = getSamples(
        train_samples[:num], True
    )

    val_surf_samples = getSamples(
        val_samples[:num], True
    )
    
    return train_surf_samples, val_surf_samples

def getMinDist(p, samples):
    min_dist = 1e8
    for s in samples:
        dist = chamfer.calc_metrics(
            p.squeeze().T.unsqueeze(0).cpu(),
            s.squeeze().T.unsqueeze(0).cpu(),
            []
        )[0]
        min_dist = min(dist.item(), min_dist)
    return min_dist

def gen_metrics(
    gen_progs, outpath, exp_name, epoch, VERBOSE, num_write, train_samps, val_samps
):
    misses = 0.
    results = {
        'num_parts': [],        
        'rootedness': [],
        'stability': [],
        'gen': [],
        'cov': [],
        'var': [],
    }

    samples = []

    print("DOING ROOTED AND STABLE")
    
    for i, prog in enumerate(gen_progs):
        try:
            verts, faces = hier_execute(prog)            
            assert not torch.isnan(verts).any(), 'saw nan vert'

            if i < num_write:
                utils.writeObj(verts, faces, f"{outpath}/{exp_name}/objs/gen/{epoch}_{i}.obj")
                if 'dsl_prog' in prog:
                    utils.writeHierProg(prog, 'dsl_prog', f"{outpath}/{exp_name}/programs/gen/{epoch}_{i}.txt")
                else:
                    utils.sawriteHierProg(prog, f"{outpath}/{exp_name}/programs/gen/{epoch}_{i}.txt")
            
            results['num_parts'].append(verts.shape[0] / 8.0)
            samples.append((verts, faces))
            
        except Exception as e:
            misses += 1.
            if VERBOSE:
                print(f"failed gen metrics for {i} with {e}")
            continue

        if i < MAX_ROOTED_STABLE:
            try:
                if check_rooted(verts, faces):
                    results['rootedness'].append(1.)
                else:
                    results['rootedness'].append(0.)

                if check_stability(verts, faces):
                    results['stability'].append(1.)
                else:
                    results['stability'].append(0.)

            except Exception as e:
                if VERBOSE:
                    print(f"failed rooted/stable with {e}")

                
    for key in results:
        if len(results[key]) > 0:
            res = torch.tensor(results[key]).mean().item()
        else:
            res = 0.

        results[key] = res

    print("GETTING SAMPLES")
    gen_samps = getSamples(samples, VERBOSE)
        
    try:
        assert len(gen_samps) > 0, 'no gen samps'
        print("CALC GEN")
        gen = cdpairs.calc_cd(
            torch.stack([g[:CD_NUM_SAMPS] for g in gen_samps[:MAX_VARI]]),
            torch.stack([g[:CD_NUM_SAMPS] for g in train_samps[:MAX_VARI]]),
            False
        )
        print("CALC COV")
        cov = cdpairs.calc_cd(
            torch.stack([g[:CD_NUM_SAMPS] for g in val_samps[:MAX_VARI]]),
            torch.stack([g[:CD_NUM_SAMPS] for g in gen_samps[:MAX_VARI]]),
            False
        )
        print("CALC VAR")
        var = cdpairs.calc_cd(
            torch.stack([g[:CD_NUM_SAMPS] for g in gen_samps[:MAX_VARI]]),
            torch.stack([g[:CD_NUM_SAMPS] for g in gen_samps[:MAX_VARI]]),
            True
        )
        
        results['gen'] = gen
        results['cov'] = cov
        results['var'] = var
    
    except Exception as e:
        results['gen'] = 1.0
        results['cov'] = 1.0
        results['var'] = 1.0
        if VERBOSE:
            print(f"failed NN comparisons with {e}")
    
    try:
        results['val_fd'] = get_fd(
            [g.cpu() for g in gen_samps],
            [v.cpu() for v in val_samps],
            None)

    except Exception as e:
        results['val_fd'] = 100.

        if VERBOSE:
            print(f"failed getting val variance with {e}")
        
    try:
        results['train_fd'] = get_fd(
            [g.cpu() for g in gen_samps],
            [t.cpu() for t in train_samps],
            None)
        
    except Exception as e:
        results['train_fd'] = 100.
        
        if VERBOSE:
            print(f"failed getting train variance with {e}")
        
    return results, misses

if __name__ == '__main__':
    from ShapeAssembly import ShapeAssembly, make_hier_prog, hier_execute
    import sys
    import time
    sa = ShapeAssembly()
    meshes = []
    for ind in list(os.listdir('data/chair'))[:int(sys.argv[1])]:
        lines = sa.load_lines(f'data/chair/{ind}')
        hier_prog = make_hier_prog(lines)
        verts, faces = hier_execute(hier_prog)
        meshes.append((verts, faces))
        
    gen_samps = getSamples(meshes, True)
    t = time.time()
    gen = cdpairs.calc_cd(torch.stack(gen_samps), torch.stack(gen_samps), True)
    print(gen)
    print(time.time() - t)
