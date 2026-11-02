"""H22 runner: same capture/predict/HTTP contract, separate locked weights."""
import sys
sys.dont_write_bytecode=True
import run as base

EXPECTED_H22_LOCK_SHA='BUILD_REPLACES_WITH_LOCK_SHA'

def load_model(folder,device):
    base.verify(folder)
    import torch
    import large_inventory_model as model
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    if device=='cuda':
        torch.cuda.set_per_process_memory_fraction(.4);torch.backends.cuda.matmul.allow_tf32=False
    return model.Predictor(folder,device)

def model_info(predictor):
    count=lambda x:sum(p.numel() for p in x.parameters())
    gcn=count(predictor.models['recent'][0]);flow=count(predictor.flows[0])
    return {'model':'H22 GCN + cached-condition Flow Matching','model_lock_sha256':EXPECTED_H22_LOCK_SHA,'gcn_parameters_per_seed':gcn,'flow_parameters_per_seed':flow,'seeds':[11,29],'all_loaded_parameters':2*(gcn+flow),'fp32_weight_bytes':8*(gcn+flow),'trained_stations':predictor.n,'horizon_seconds':600,'particles':128,'ode_steps':16,'weights_updated':False}

base.LOCK_SHA=EXPECTED_H22_LOCK_SHA
base.load_model=load_model
base.model_info=model_info
if __name__=='__main__':base.main()
