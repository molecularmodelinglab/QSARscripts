from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator as rdFpGen
from rdkit import DataStructs
from scipy import sparse
import numpy as np

# this is kat’s work. :3

def get_fp(smiles, fp_type='morgan', counts=False, bits=1024, 
           radius=2, chiral=False, sparsed=False):
    
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return False
    
    gen = None
    
    # Extended-connectivity fingerprint (Topological circular)
    if fp_type=='morgan':
        gen = rdFpGen.GetMorganGenerator(radius=radius,fpSize=bits,includeChirality=chiral)      
    # Functional class fingerprint (Topological circular)
    if fp_type=='fcfp':
        gen = rdFpGen.GetMorganGenerator(radius=radius,fpSize=bits,includeChirality=chiral, 
                                         atomInvariantsGenerator=rdFpGen.GetMorganFeatureAtomInvGen())
    # Daylight-esque fingerprint (Topological path-based)
    if fp_type=='rdkit':
        # maxPath = diameter = radius * 2
        gen = rdFpGen.GetRDKitFPGenerator(maxPath=radius*2,fpSize=bits,numBitsPerFeature=2)
    # "based on the atomic environments and shortest path separations of every atom pair" 
    if fp_type=='atom_pair':
        gen = rdFpGen.GetAtomPairGenerator(maxDistance=radius,fpSize=bits,includeChirality=chiral)
    
    if counts==True:
        _fp = gen.GetCountFingerprint(mol)
    else:
        _fp = gen.GetFingerprint(mol)
        
    fp = np.zeros(bits, dtype=np.int32)
    DataStructs.ConvertToNumpyArray(_fp, fp)
    
    if sparsed:
        return sparse.csr_matrix(fp)
    else:
        return fp
    
    
from joblib import Parallel, delayed

def get_fps_in_parallel(smiles, fp_type='morgan', counts=False, bits=1024, 
           radius=2, chiral=False, sparsed=False):
    parallelizer = Parallel(n_jobs=-1, backend= 'multiprocessing' )
    fp_tasks = (delayed(get_fp)(sm,fp_type,counts,bits,radius,chiral) for sm in smiles)
    fps = parallelizer(fp_tasks)
    fps = np.vstack(fps)
    return fps
