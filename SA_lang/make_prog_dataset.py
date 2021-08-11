from tqdm import tqdm
import sys
sys.path.append('../')
sys.path.append('parse_files')
import pickle
from json_parse import loadAllData
from sa_prog_order import get_all_valid_orders, canonicalize
from ShapeMOD import ProgNode, OrderedProg

def main(in_path, out_name, max_progs):
    inds, nodes = loadAllData(in_path, max_progs)
    
    programs = []
    
    for ind, node in tqdm(list(zip(inds, nodes))):        
        save_oprogs = []
        
        orders = get_all_valid_orders(node)                        
        for o in orders:
            co, (fparams, lines, ret), (canon_lines, canon_ret) = canonicalize(o, node)
            
            op = OrderedProg(
                co,
                lines,                
                ret,
                fparams,
            )
            op.canon_info = (o, canon_lines, canon_ret)
            save_oprogs.append(op)            

        pn = ProgNode(ind, save_oprogs)
        pn.children_names = node['children_names']        
        programs.append(pn)

    pickle.dump(programs, open(out_name, 'wb'))                        
                          
if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
