"""Validate a preserved passing primary, not a failed or uncertain reference."""
import math
import common as C

def strict(metrics,mass,tight=False):
    target=1e-11 if tight else 1e-9
    C.require(len(metrics)==8 and all(math.isfinite(x) for x in metrics+[mass]),'Invalid audit')
    m=metrics
    C.require(m[2]>0 and m[0]<=min(1e-10,target) and max(m[4],m[5],mass)<=target
        and m[6]>=-1e-9 and m[7]<=1+1e-9,'Unchanged scientific audit failed')
    return m[2]

def passing_primary(d,row):
    C.require(not d['passed'] and d['failed_stage']=='reference' and d['key']==row['key'],'Wrong failed stage/key')
    C.require(d['source']==row['saved_source'] and d['mode']==row['saved_mode'],'Primary source changed')
    C.require(d['parts_hash']==row['parts_hash'] and d['operator_sha256']==row['operator_sha256'],'Primary operator changed')
    C.require((d['primary_target'],d['reference_target'],d['agreement_target'])==(1e-9,1e-11,1e-5),'Changed targets')
    successes=[a for a in d['attempts'] if a['passed']]
    C.require(len(successes)==1 and not successes[0]['tight'] and successes[0]['shift']==row['primary_shift'],'Ambiguous primary')
    h=successes[0]['history'][-1]
    C.require(len(h)==10,'Primary history format')
    q=strict(h[1:9],h[9])
    audits=d['independent_checks']
    C.require(len(audits)==1 and audits[0]['stage']==row['primary_stage'],'Independent primary identity')
    a=audits[0]
    C.require(a['independent_passed'] and a['independent_q_difference']<=1e-5,'Missing passing primary audit')
    C.require(strict(a['independent_metrics'],a['independent_mass'])==q,'Primary flux mismatch')
    C.require(d['production_writes']==d['completed_phase_calls_repeated']==0,'Prior diagnostic side effects')
    return q
