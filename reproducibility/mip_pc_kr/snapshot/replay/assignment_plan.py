"""Case-independent assignment joins; no replay, numerical cache hit or submission."""
import hashlib
import json
import math
from pathlib import PurePosixPath
import re


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def exact_int(value,label):
    if isinstance(value,bool) or not re.fullmatch(r'\d+',str(value)):
        raise ValueError('Non-exact integer '+label)
    return int(value)


def hexid(value,length=64):
    if not isinstance(value,str) or not re.fullmatch('[0-9a-f]{'+str(length)+'}',value):
        raise ValueError('Invalid source/identity digest')
    return value


def relative_checkpoint(value):
    if not isinstance(value,str):raise ValueError('Checkpoint path must be text')
    p=PurePosixPath(value)
    if not isinstance(value,str) or '\\' in value or ':' in value or p.is_absolute() or '..' in p.parts:
        raise ValueError('Checkpoint must stay under frozen data root')
    if len(p.parts)<2 or p.parts[0]!='data' or p.name!='predict_runs.mat':
        raise ValueError('Unexpected checkpoint layout')
    return str(p)


def identity(row,summary,sources):
    for left,right in [('geology_id','GeologyId'),('window','Window'),('task_id','TaskId'),
                       ('task_key_sha256','TaskKeySha256'),('checkpoint_sha256','CheckpointSha256'),
                       ('checkpoint_relative_path','CheckpointRelativePath'),
                       ('predict_code_commit','PredictCodeCommit'),('predict_method_config_hash','PredictMethodConfigHash'),
                       ('sampling_code_commit','SamplingCodeCommit'),('sampling_method_config_hash','SamplingMethodConfigHash'),
                       ('sampling_case_id','SamplingCaseId'),('phase','SamplingPhase'),
                       ('case_type','SamplingCaseType'),('design_version','SamplingDesignVersion'),
                       ('adapter_schema_version','SamplingAdapterSchemaVersion')]:
        if row[left]!=summary[right]:raise ValueError('Assignment/production summary mismatch: '+left)
    numbers={}
    for left,right in [('case_id','Level3CaseId'),('slice_index','SliceIndex'),
                       ('selected_sample_index','SelectedSampleIndex'),('exact_replay_seed','ReplaySeed')]:
        a=exact_int(row[left],left);b=exact_int(summary[right],right)
        if a!=b:raise ValueError('Assignment/production integer mismatch: '+left)
        numbers[left]=a
    sampling_seed=exact_int(row['sampler_random_seed'],'sampler_random_seed')
    if sampling_seed!=exact_int(row['draw_seed'],'draw_seed') or sampling_seed!=exact_int(summary['SamplingSeed'],'SamplingSeed'):
        raise ValueError('Original sampler seed mismatch')
    if exact_int(row['predict_realization_id'],'predict_realization_id')!=numbers['selected_sample_index'] or exact_int(summary['PredictRealizationId'],'PredictRealizationId')!=numbers['selected_sample_index']:
        raise ValueError('Original realization index mismatch')
    if row['phase']!='phase1' or not (row['case_type']=='independent_full' and 1<=numbers['case_id']<=12 or row['case_type']=='high_state_stress' and numbers['case_id']==103):
        raise ValueError('Unsupported original campaign selection')
    if numbers['selected_sample_index']<1 or not 0<numbers['exact_replay_seed']<2**53:
        raise ValueError('Unsupported checkpoint replay identity')
    if row['predict_code_commit']!=sources['physics_commit'] or row['predict_method_config_hash']!=sources['physics_method_sha256']:
        raise ValueError('Frozen physics contract mismatch')
    key=hexid(row['task_key_sha256'])
    if row['task_id']!='rpc_'+key[:32]:raise ValueError('Upstream task label mismatch')
    checkpoint=relative_checkpoint(row['checkpoint_relative_path'])
    checkpoint_hash=hexid(row['checkpoint_sha256'])
    properties=[float(row['perm_'+axis]) for axis in ('kxx','kyy','kzz')]
    if not all(math.isfinite(x) and x>0 for x in properties):raise ValueError('Invalid intrinsic permeability')
    pv,vol=float(summary['PoreVolume']),float(summary['BulkVolume'])
    cells=exact_int(summary['NumCells'],'NumCells')
    if not math.isfinite(pv) or not math.isfinite(vol) or not 0<pv<vol or cells<1:
        raise ValueError('Invalid production grid summary')
    # This proposes a replay grouping only; actual binary identity is still required.
    payload=dict(schema=1,upstream_task_key=key,geology=row['geology_id'],window=row['window'],
        checkpoint_relative_path=checkpoint,checkpoint_sha256=checkpoint_hash,
        sample=numbers['selected_sample_index'],seed=numbers['exact_replay_seed'],
        source_seed_base=exact_int(row['source_seed_base'],'source_seed_base'),
        intrinsic_perm_md_hex=[x.hex() for x in properties],num_cells=cells,
        pv_hex=pv.hex(),bulk_volume_hex=vol.hex(),source_contract=sources)
    if payload['seed']<payload['source_seed_base']:raise ValueError('Replay seed precedes base seed')
    return payload,numbers


def plan_case(case,records,sources,policy):
    if len(records)!=522:raise ValueError('Exactly 522 placements required')
    for key in ('archive_sha256','prepare_slot_sha256','replay_orchestrator_sha256','physics_method_sha256'):
        hexid(sources[key])
    hexid(sources['physics_commit'],40)
    if policy['scenario']['clay_contact_angle_deg']!=60 or not policy['scenario']['physical_prebreakthrough_saturation']:
        raise ValueError('WRR60 physical pre-breakthrough contract required')
    if policy['scenario']['saved_pressure_points']!=100 or policy['scenario']['predict_fault_hysteresis']:
        raise ValueError('Candidate sampling/hysteresis contract mismatch')
    case_key=case['case_key'];geology=case['geology_id'];realization=exact_int(case['realization_id'],'realization_id')
    if case_key!=f'{geology}_case{realization:02d}':raise ValueError('Manifest case identity mismatch')
    groups={};placements=[];seen=set();task_payloads={}
    for record in records:
        row,summary=record['assignment'],record['production_summary']
        if row['geology_id']!=geology or exact_int(row['case_id'],'case_id')!=realization:
            raise ValueError('Mixed geology or case in placement manifest')
        if row['window'] not in [f'famp{i}' for i in range(1,7)]:raise ValueError('Invalid window')
        window=int(row['window'][4:]);sl=exact_int(row['slice_index'],'slice')
        slot=(window-1)*87+sl
        if not 1<=sl<=87 or exact_int(record['slot'],'slot')!=slot or slot in seen:
            raise ValueError('Duplicate or inconsistent slot mapping')
        seen.add(slot)
        payload,numbers=identity(row,summary,sources)
        if record['checkpoint_sha256']!=payload['checkpoint_sha256']:raise ValueError('Record checkpoint mismatch')
        key=digest(payload);upstream=row['task_key_sha256']
        if upstream in task_payloads and task_payloads[upstream]!=key:
            raise ValueError('Same upstream task has conflicting physical identity')
        task_payloads[upstream]=key
        group=groups.setdefault(key,dict(candidate_replay_key=key,identity=payload,slots=[],selection=row))
        group['slots'].append(slot)
        placements.append(dict(slot=slot,saturation_region=slot+8,window=window,slice=sl,
            sample=numbers['selected_sample_index'],seed=numbers['exact_replay_seed'],
            sampler_seed=str(exact_int(row['sampler_random_seed'],'sampler_random_seed')),
            candidate_replay_key=key,assignment_sha256=digest(row),
            production_summary_sha256=digest(summary)))
    if seen!=set(range(1,523)):raise ValueError('Incomplete full-fault coverage')
    ordered=sorted(groups.values(),key=lambda g:min(g['slots']))
    batches={}
    for g in ordered:
        x=g['identity'];group_key=(geology,realization,x['window'],x['checkpoint_sha256'],x['checkpoint_relative_path'])
        batches.setdefault(group_key,[]).append(g['candidate_replay_key'])
    return dict(schema=1,case_key=case_key,task=exact_int(case['task'],'task'),
        policy_sha256=digest(policy),source_contract=sources,
        placements=sorted(placements,key=lambda p:p['slot']),candidate_replay_groups=ordered,
        checkpoint_batches=[dict(geology=k[0],case_id=k[1],window=k[2],checkpoint_sha256=k[3],
            checkpoint_relative_path=k[4],candidate_replay_keys=v) for k,v in sorted(batches.items())],
        numerical_cache_hits=0,replay_reuse_approved=False,dispatch_approved=False,
        generic_replay_executor_implemented=False,production_modified=False,
        pending=['runtime_source_qualification','checkpoint_accepted_seed_attempt_validation',
                 'geometry_replay_canaries','binary_identity_verification','resource_and_storage_admission',
                 'WRR60_transform_and_numerical_qualification','export_and_importer_roundtrip'])


def verify_placement_inputs(plan,receipts):
    """Bind every placement to its exact planned replay identity, not a sample label."""
    expected={g['candidate_replay_key'] for g in plan['candidate_replay_groups']}
    if set(receipts)!=expected:raise ValueError('Missing or extraneous replay receipts')
    for key,receipt in receipts.items():
        if receipt.get('candidate_replay_key')!=key or receipt.get('complete') is not True:
            raise ValueError('Incomplete or mismatched preparation cannot be reused')
        for name in ('replay_file_sha256','fine_fields_sha256','architecture_sha256','source_contract_sha256'):
            hexid(receipt[name])
        if receipt['source_contract_sha256']!=digest(plan['source_contract']):
            raise ValueError('Preparation source mismatch')
    return [dict(**p,preparation_receipt=receipts[p['candidate_replay_key']]) for p in plan['placements']]
