"""Validate the new cache proof while retaining exact v3 saved-flow recovery."""
from common import require

LEGACY_RUNNER='4497359d39e01a47ce830ca4495a9682ba330b210eaa0856b785c45761d551c5'
CACHE_FILES=('DrainageCache.jl','CachedAdaptiveCurve.jl')
CACHED_PREDECESSORS={
    'facf4d6fb3f93f144c19dfd6862e7cf45c24803b6333b4d69d6031304adc2c6a',
    '988abcc32f15861aa95e6662c932e28b897c4bd616f42893897d944cbaaff019',
    '709f7267bd1233138a1d1e50e7999f536343f566ea01e51a352922398fdabbd8',
    'c5b9ae6c9e49f6ff8a21f66aaba7724235a761de63e03bd04c1a96e93dc1bccf',
}

def verify_cache_evidence(summary,receipt,source,runner_pin,science):
    if 'state_access' not in summary:
        require(source['runner_source']==LEGACY_RUNNER,'Missing state-cache evidence from new runner')
        require('state_cache_implementation' not in receipt,'Mixed legacy/cache proof')
        return 'legacy_saved_flow'
    require(source['runner_source'] in CACHED_PREDECESSORS|{runner_pin},'Wrong cached-flow runner source')
    require(summary['state_access']=='verified_per_core_cache_v1','Unknown state access')
    require(summary['cached_states_verified']==100 and summary['cached_queries_verified']==summary['kr_states'],'Incomplete cache verification')
    require(summary['replay_states_checked']==0 and summary['reference_states_checked']==0,'Unexpected qualification replay in worker')
    require(summary['cache_payload_bytes']==100*(26*summary['cells']+16) and summary['cache_payload_bytes']<=512*1024**2,'Cache memory bound')
    require(receipt['state_cache_implementation']=={n:science[n] for n in CACHE_FILES},'Cache implementation changed')
    return 'verified_cache'
