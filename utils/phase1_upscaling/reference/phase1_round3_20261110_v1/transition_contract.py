"""Explicit successor authority; keep original pilot seals and assignments intact."""
import common as C

ORIGIN='e39642918493925c6c122f783be2cd35e153fc51db9838ab5169f8241e95d613'


def validate_authority(scope, original, adoption, qualification, source, lock):
    C.require(original['approved_by_user'] and original['source']==ORIGIN,'Original recovery authority changed')
    C.require(adoption['approved_by_user'] and adoption['previous_source']==ORIGIN and adoption['source']==source,'Invalid successor authority')
    C.require(qualification['passed'] and qualification['source']==source,'Successor not qualified')
    for field in ('method','configuration'):
        expected=lock[field+'_sha256']
        C.require(qualification[field]==adoption[field]==expected,'Scientific identity changed')
    expected=source if scope['phase']=='solve' else ORIGIN
    C.require(scope['source']==expected,'Scope/source transition mismatch')


def require_held_gate(fields, job, dependency, script):
    C.require(fields.get('JobId')==job and fields.get('JobState')=='PENDING' and fields.get('Reason')=='JobHeldUser','Old handoff is not held/pending')
    C.require(fields.get('Command')==script,'Old handoff launcher changed')
    dep=fields.get('Dependency','')
    C.require(dep=='(null)' or dep.split('(',1)[0] in ('afterany:'+dependency,'afterany:'+dependency+'_*'),'Old handoff dependency changed')


def retired_gate(raw, job):
    rows=[r.strip().split('|') for r in raw.splitlines() if r.strip()]
    exact=[r for r in rows if r[0].strip()==job]
    C.require(len(exact)==1,'Old handoff accounting unavailable/ambiguous')
    C.require(exact[0][1].strip().split()[0]=='CANCELLED','Old handoff not retired')


def validate_pilot_boundary(state, launch, solve_exists, pilot_proof_exists):
    C.require(state==dict(cursor=1,active=['s01_c003_pc06_ca40_IND_001']),'Pilot boundary changed')
    C.require(launch==dict(job='28458640',gate='28458641',packets=2),'Pilot assignment changed')
    C.require(not solve_exists and not pilot_proof_exists,'Old continuation already advanced; reconcile first')
