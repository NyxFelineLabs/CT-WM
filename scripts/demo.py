"""Small CPU demo of the semantic KPM -> OLLA -> RC loop; synthetic inputs only."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
sys.path.insert(0,str(ROOT/'src'))
from e2_sim import PythonLink,VirtualE2Node,E2SMKPMIndication as KPM,E2SMRCControl as RC
from e2_xapps import OLLAXApp

def main(steps):
    if not 1<=steps<=128:
        raise ValueError('steps must be 1..128')
    frame=np.zeros((200,13),np.float32)
    frame[:,0]=14;frame[:,1]=10;frame[:,5]=20;frame[:,7]=-88
    frame[:,9]=17+2*np.sin(np.arange(200)/12)
    frame[:,10]=1;frame[:,11]=.1
    node=VirtualE2Node(PythonLink());app=OLLAXApp(node.link)
    app.reset(node.reset(frame,64,42));se=0
    for _ in range(steps):
        kpm=KPM.read(node.last_indication.wire())
        control,_=app.decide(kpm)
        _,metrics=node.execute(RC.read(control.wire()))
        se+=metrics['SE']
        assert metrics['PRB']==20
    print(json.dumps(dict(synthetic_demo=True,not_a_benchmark=True,steps=steps,
        fixed_prb=20,final_mcs=metrics['MCS'],offset_db=app.offset.offset_db,
        cumulative_se=se,interface='local semantic JSON only'),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=16)
    main(p.parse_args().steps)
