"""Exact body ordering and force decomposition; fixtures are not native physics."""
from types import SimpleNamespace as NS
import sys,numpy as np,torch,pytest
from CrowdSim.humanoid.runtime.native_contact_report import read_static_report

def fixture(monkeypatch,static=0.):
 monkeypatch.setitem(sys.modules,'warp',NS(to_torch=lambda x:x))
 names=[f'Body{i}' for i in range(24)]
 paths=[[f'/World/envs/env_{a}/Robot/{n}' for n in names] for a in range(4)]
 own=torch.zeros(4,24,3);own[:,3,0]=40.
 floor=torch.zeros(4,24,3);floor[:,4,2]=600.
 peer=torch.zeros(4,24,3);peer[:,7,1]=12.
 net=floor+own+peer;net[0,9,0]+=static
 sensors={}
 for i,n in enumerate(names):
  matrix=torch.zeros(4,1,25,3);matrix[:,0,0]=floor[:,i];matrix[:,0,1]=own[:,i]
  sensors[n]=NS(data=NS(force_matrix_w=matrix,net_forces_w=net[:,i,None]))
 views=[]
 for a in range(4):
  matrix=torch.zeros(24,74,3);matrix[:,2]=peer[a].flip(0)
  view=NS(sensor_paths=paths[a][::-1],get_contact_force_matrix=lambda dt,m=matrix:m)
  views.append((a,view,2,[i for i in range(4) if i!=a],24,[],24))
 monitor=NS(world=NS(num_envs=4,mapping=NS(body_names=names)),peer=NS(views=views,manifest={'actor_body_paths':paths}))
 return NS(),NS(_contact_sensor_map=sensors),monitor

@pytest.mark.parametrize('static',[0.,17.])
def test_residual_subtracts_floor_peer_and_own_in_actual_body_order(monkeypatch,static):
 provider,sim,monitor=fixture(monkeypatch,static)
 force=read_static_report(provider,simulator=sim,monitor=monitor,dt=1/120,physics_step=1)
 assert np.allclose(force,[static,0.,0.,0.])
 assert provider._native_contact_stats['floor_positive_substeps']==1
 assert len(provider._native_residual_rows)==1
 with pytest.raises(RuntimeError,match='Skipped/repeated'):
  read_static_report(provider,simulator=sim,monitor=monitor,dt=1/120,physics_step=1)

def test_residual_rejects_foreign_body_identity(monkeypatch):
 provider,sim,monitor=fixture(monkeypatch)
 monitor.peer.views[0][1].sensor_paths[0]='/World/Foreign'
 with pytest.raises(ValueError,match='identities'):
  read_static_report(provider,simulator=sim,monitor=monitor,dt=1/120,physics_step=1)
