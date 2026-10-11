#!/usr/bin/env python3
"""Build the pinned optional ORCA dependency using the configured SDK Python.

CMake and a C++ compiler are required. Build tools and the resulting extension
stay in the project; no packages or files are installed into the SDK.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from social_navigation.external_runtime import read_binding

COMMIT='c2c46ba8d59556aa10faf03479293236efea154d'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'.runtime/dependencies/Python-RVO2')
    args=parser.parse_args()
    binding=read_binding(ROOT)
    python=binding['python']
    destination=ROOT/'.runtime/rvo2_native'
    if destination.exists():parser.error('Existing native dependency preserved; choose a fresh checkout for rebuilding')
    for tool in ('uv','cmake','c++'):
        if shutil.which(tool) is None:parser.error('Required build tool missing: '+tool)
    source=args.source.expanduser().resolve()
    if not source.exists():
        source.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(['git','clone','https://github.com/sybrenstuvel/Python-RVO2.git',str(source)],check=True)
        subprocess.run(['git','-C',str(source),'checkout','--detach',COMMIT],check=True)
    if subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()!=COMMIT:
        raise RuntimeError('RVO2 source commit differs from the registered ORCA dependency')
    if subprocess.check_output(['git','-C',str(source),'diff','HEAD','--'],text=True).strip():
        raise RuntimeError('RVO2 tracked source files changed')
    candidate=Path(tempfile.mkdtemp(prefix='.rvo2_native-',dir=ROOT/'.runtime'))
    tools=candidate/'build_tools'
    try:
        subprocess.run(['uv','pip','install','--python',python,'--target',str(tools),
                        '--no-deps','Cython==3.0.11','setuptools==81.0.0'],check=True)
        code=('import sys,site,runpy;site.addsitedir(sys.argv[1]);sys.path.insert(0,sys.argv[1]);'
              'sys.argv=["setup.py","build_ext","--build-lib",sys.argv[2],"--build-temp",sys.argv[3]];'
              'runpy.run_path("setup.py",run_name="__main__")')
        subprocess.run([python,'-I','-c',code,str(tools),str(candidate),str(candidate/'temp')],cwd=source,check=True)
        modules=list(candidate.glob('rvo2*.so'))
        if len(modules)!=1:raise RuntimeError('Expected exactly one compiled RVO2 extension')
        probe=('import sys,json;sys.path.insert(0,sys.argv[1]);import rvo2;'
               's=rvo2.PyRVOSimulator(.1,10,10,2,2,.4,2);i=s.addAgent((0,0));'
               's.setAgentPrefVelocity(i,(1,0));s.doStep();p=s.getAgentPosition(i);'
               'assert p[0]>0 and p[1]==0;print(json.dumps({"python":sys.executable,"position":p}))')
        result=json.loads(subprocess.check_output([python,'-I','-c',probe,str(candidate)],text=True))
        binary=modules[0]
        receipt={'schema':'social_navigation.rvo2_native.v1','source_commit':COMMIT,
                 'source':str(source),'sdk_python':python,'cython':'3.0.11','setuptools':'81.0.0',
                 'binary':binary.name,'sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),
                 'probe':result,'sdk_modified':False}
        shutil.copyfile(source/'LICENSE',candidate/'LICENSE')
        shutil.rmtree(tools);shutil.rmtree(candidate/'temp')
        (candidate/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        candidate.rename(destination)
        print(json.dumps(receipt))
    except BaseException:
        shutil.rmtree(candidate,ignore_errors=True)
        raise
    return 0


if __name__=='__main__':raise SystemExit(main())
