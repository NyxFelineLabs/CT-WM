import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments'),str(ROOT/'scripts')]
from simple_olla import SimpleOLLA
from e2_sim import PythonLink,VirtualE2Node,E2SMKPMIndication as KPM,E2SMRCControl as RC
from e2_xapps import OLLAXApp
from wm_ms3m.core import WMMS3MModel
from planner_runtime import PriorMean,verify
import prepare_data

class OLLATests(unittest.TestCase):
    def test_negative_feedback_and_target_balance(self):
        o=SimpleOLLA();threshold=np.arange(29,dtype=float)
        before=o.select_mcs(17,threshold)
        o.update(False)
        self.assertGreater(o.offset_db,0)
        self.assertLess(o.select_mcs(17,threshold),before)
        o=SimpleOLLA()
        for ack in [True]*9+[False]:
            o.update(ack)
        self.assertAlmostEqual(o.offset_db,0)

    def test_actual_feedback_type_and_bounds(self):
        o=SimpleOLLA()
        with self.assertRaises(TypeError):o.update(.1)
        for _ in range(100):o.update(False)
        self.assertEqual(o.offset_db,20)

class InterfaceTests(unittest.TestCase):
    def setUp(self):
        self.link=PythonLink()
        self.frame=np.zeros((200,13),np.float32)
        self.frame[:,0]=14;self.frame[:,1]=10;self.frame[:,5]=20
        self.frame[:,7]=-88;self.frame[:,9]=17

    def test_json_roundtrip_and_own_feedback_reference(self):
        node=VirtualE2Node(self.link);app=OLLAXApp(self.link)
        app.reset(node.reset(self.frame,64,42))
        for _ in range(128):
            k=KPM.read(node.last_indication.wire())
            rc,_=app.decide(k);rc=RC.read(rc.wire())
            wire,row=node.execute(rc)
            self.assertEqual(row['PRB'],20)
            self.assertEqual(row['SE'],row['reference_SE'])
            self.assertEqual(len(KPM.read(wire).ack_nack),20)
        with self.assertRaises(ValueError):node.execute(rc)

    def test_invalid_action_rejected(self):
        with self.assertRaises(ValueError):RC('node',1,0,1,0,29)
        with self.assertRaises(TypeError):RC('node',1,0,1,.5,14)

class FrozenExecutorTests(unittest.TestCase):
    def test_cached_kernels_preserve_prior_mean(self):
        torch.manual_seed(7)
        model=WMMS3MModel(in_dim=13,act_dim=1,d_model=8,z_dim=2,L=4,
            n_layers=1,state_dim=4,n_components=2,full_dim=13,dropout=0).eval()
        meta=dict(x_mean=[0.]*13,x_std=[1.]*13,y_full_mean=[0.]*13,y_full_std=[1.]*13)
        bundle=types.SimpleNamespace(model=model,meta=meta,cfg=types.SimpleNamespace(device='cpu'),
                                     x_scaler=types.SimpleNamespace(transform=lambda x:x))
        before={k:v.clone() for k,v in model.state_dict().items()}
        executor=PriorMean(bundle)
        result=verify(bundle,executor,[np.random.default_rng(0).normal(size=(4,13)).astype(np.float32)])
        self.assertEqual(result['max_absolute_physical_output_error'],0)
        for key,value in model.state_dict().items():
            torch.testing.assert_close(before[key],value,rtol=0,atol=0)

class DataPreparationTests(unittest.TestCase):
    def test_windows_and_split_boundaries_stay_within_files(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'data/raw').mkdir(parents=True)
            records=[]
            for i in range(10):
                name=f'{i:02d}.csv'
                values=np.full((140,13),float(i))
                frame=pd.DataFrame(values,columns=prepare_data.COLUMNS)
                frame.insert(0,'time',np.arange(140)+1000*i)
                frame.to_csv(root/'data/raw'/name,index=False)
                records.append(dict(file=name,sha256='unused',split='train' if i<7 else 'val' if i==7 else 'test'))
            (root/'data/manifest.reference.json').write_text(json.dumps(dict(files=records)))
            old=prepare_data.ROOT
            try:
                prepare_data.ROOT=root
                manifest=prepare_data.prepare()
            finally:
                prepare_data.ROOT=old
            self.assertEqual(manifest['counts'],dict(train=756,val=108,test=216))
            windows=np.load(root/'data/x.npy',mmap_mode='r')
            targets=np.load(root/'data/y.npy',mmap_mode='r')
            for i,r in enumerate(manifest['files']):
                self.assertTrue(np.all(windows[r['sample_start']:r['sample_stop']]==i))
                self.assertTrue(np.all(targets[r['sample_start']:r['sample_stop']]==i))

if __name__=='__main__':unittest.main()
