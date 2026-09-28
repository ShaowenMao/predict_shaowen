"""Bound read-cache residency during an audit, without changing source bytes."""
import hashlib
import json
import os
from pathlib import Path


def discard_consumed_pages(stream):
    if hasattr(os,'posix_fadvise'):
        stop=(stream.tell()//4096)*4096
        if stop:
            os.posix_fadvise(stream.fileno(),0,stop,os.POSIX_FADV_DONTNEED)


def normalized_file(path,header_size,edits):
    actual=hashlib.sha256();normalized=hashlib.sha256();size=0
    with Path(path).open('rb') as stream:
        head=stream.read(header_size);assert len(head)==header_size
        actual.update(head);modified=bytearray(head)
        for start,stop,value in edits:
            assert len(value)==stop-start
            modified[start:stop]=value
        normalized.update(modified);size+=len(head)
        for block in iter(lambda:stream.read(1024**2),b''):
            actual.update(block);normalized.update(block);size+=len(block)
            discard_consumed_pages(stream)
    return actual.hexdigest(),normalized.hexdigest(),size,head


def scientific(x):
    if isinstance(x,dict):
        return {k:scientific(v) for k,v in x.items() if k!='seconds' and not k.endswith('_seconds')}
    if isinstance(x,list):return [scientific(v) for v in x]
    return x


def scientific_hash(path):
    h=hashlib.sha256()
    with Path(path).open() as f:
        for line in f:
            h.update(json.dumps(scientific(json.loads(line)),sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n')
    return h.hexdigest()
