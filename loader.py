import json
import numpy as np
import os
from typing import List
from numpy.typing import NDArray

from arcle.loaders import ARCLoader
from arcle.loaders import MiniARCLoader

class SizeConstrainedLoader(ARCLoader):
    def __init__(self, size, train=True) -> None:
        self.size = size
        super().__init__(train=train)
    
    def parse(self, **kwargs):
        
        dat = []

        for p in self._pathlist:
            with open(p) as fp:
                problem = json.load(fp)

                ti: List[NDArray] = []
                to: List[NDArray] = []
                ei: List[NDArray] = []
                eo: List[NDArray] = []


                for d in problem['train']:
                    inp = np.array(d['input'],dtype=np.uint8)
                    oup = np.array(d['output'],dtype=np.uint8)
                    if inp.shape[0] > self.size or inp.shape[1] > self.size or oup.shape[0] > self.size or oup.shape[1] > self.size:
                        continue
                    ti.append(inp)
                    to.append(oup)

                for d in problem['test']:
                    inp = np.array(d['input'],dtype=np.uint8)
                    oup = np.array(d['output'],dtype=np.uint8)
                    if inp.shape[0] > self.size or inp.shape[1] > self.size or oup.shape[0] > self.size or oup.shape[1] > self.size:
                        continue
                    ei.append(inp)
                    eo.append(oup)

                if len(ti) == 0:
                    continue

                desc = {'id': os.path.basename(fp.name).split('.')[0]}
                dat.append((ti,to,ei,eo,desc))
                
        return dat
    
class MiniARCLoader(MiniARCLoader):
    def __init__(self) -> None:
        super().__init__()
    
    def parse(self, **kwargs):
        
        dat = []

        for p in self._pathlist:
            with open(p) as fp:
                fpdata = fp.read().replace('null', '"0"')
                problem = json.loads(fpdata)

                ti: List[NDArray] = []
                to: List[NDArray] = []
                ei: List[NDArray] = []
                eo: List[NDArray] = []

                for d in problem['train']:
                    ti.append(np.array(d['input'],dtype=np.uint8))
                    to.append(np.array(d['output'],dtype=np.uint8))
                
                for d in problem['test']:
                    ei.append(np.array(d['input'],dtype=np.uint8))
                    eo.append(np.array(d['output'],dtype=np.uint8))

                fns = os.path.basename(fp.name).split('_')
                desc = {'id': fns[-1].split('.')[-2], 'description': ' '.join(fns[0:-1]).strip() }

                dat.append((ti,to,ei,eo,desc))
                
        return dat
                