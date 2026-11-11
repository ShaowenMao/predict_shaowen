import common as C

TERMINAL={'FAILED','COMPLETED','TIMEOUT','OUT_OF_MEMORY','CANCELLED','NODE_FAIL','PREEMPTED'}

def terminal_jobs(text,expected):
    expected=set(expected);found={}
    for line in text.splitlines():
        fields=line.split('|')
        if len(fields)<3:continue
        raw,canonical,state=[x.strip() for x in fields[:3]]
        if raw not in expected:continue
        C.require(raw not in found,'Duplicate raw task accounting')
        C.require(state.split()[0].rstrip('+') in TERMINAL,'Previous diagnostic not terminal')
        found[raw]=canonical
    C.require(set(found)==expected,'Previous diagnostic accounting missing')
    return found
