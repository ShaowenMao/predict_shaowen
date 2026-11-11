"""Compile the original precision sources unchanged, using only an ABI adapter."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parent
numeric=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=False)
compiler=os.environ.get('CXX','g++');objects=[];hashes={}
for name,symbol in [('phase_precision','legacy_mex'),('phase_precision_repair','repair_mex')]:
    source=numeric/'vendor'/(name+'.cpp');hashes[str(source)]=hashlib.sha256(source.read_bytes()).hexdigest()
    obj=out/(name+'.o');objects.append(str(obj))
    subprocess.run([compiler,'-std=c++11','-O2','-fPIC','-fno-fast-math','-ffp-contract=off',
        '-I'+str(root/'bridge'),'-DmexFunction='+symbol,'-c',str(source),'-o',str(obj)],check=True)
library=out/('phase_bridge.dll' if os.name=='nt' else 'libphase_bridge.so')
subprocess.run([compiler,'-std=c++11','-O2','-fPIC','-fno-fast-math','-ffp-contract=off','-shared',
    str(root/'bridge/adapter.cpp'),*objects,'-lquadmath','-o',str(library)],check=True)
with (out/'build.json').open('x') as f:
    json.dump(dict(compiler=subprocess.check_output([compiler,'--version'],text=True),
                   numerical_source_sha256=hashes,library=str(library),
                   library_sha256=hashlib.sha256(library.read_bytes()).hexdigest()),f,indent=2)
print(library)
