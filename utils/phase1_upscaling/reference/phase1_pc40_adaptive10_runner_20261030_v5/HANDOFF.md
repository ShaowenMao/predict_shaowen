# Phase 1 future-job-only cache transition

Applied 2026-10-30 23:32 EDT (2026-10-31 03:32 UTC). Production now routes future
submissions through this v5 runner. The original eight active case assignments
remain on v3. Running numerical jobs were not interrupted; queued v3 workers are
explicitly dispatched to their original frozen runner when they start.

## Pins and unchanged science

- v5 LOCK: c5b9ae6c9e49f6ff8a21f66aaba7724235a761de63e03bd04c1a96e93dc1bccf.
- Qualified v4 packet runner: 709f7267bd1233138a1d1e50e7999f536343f566ea01e51a352922398fdabbd8.
- Original v3 runner: 4497359d39e01a47ce830ca4495a9682ba330b210eaa0856b785c45761d551c5.
- Frozen configuration: 7cbd42d913d516c765a6e3dff9f06994814e8b83ea67b2f96c98ae08289b1f36.
- Scientific method: ea1919d05292929667a2e33f38accd9cf0d935cdecafc5ba349083960867b832.

The v4 numerical worker body, flow script, scientific bundle, preparation,
export/native checks, planning, compact storage and Slurm admission helper are
unchanged. Runtime changes are confined to legacy assignment routing, admission
authorization and a launcher with an explicit frozen code path. New work keeps
32 cores/packet, 17 packets/case, the continuously replenished eight-case window,
2 CPUs / 18 GiB / 2 hours, no array throttle, mit_normal and both advanced account
and QoS mit_amf_advanced_cpu. No new manager, monitor or competing pump exists.

## Qualification and actual cutover

- All 43 tests passed locally and on compute job 27415550 (COMPLETED 0:0).
- Original v4 real 32-core checkpoint/resume/archive/no-op qualification remains
  the numerical qualification; v5 tests prove the scientific body is identical.
- Mixed old/new Slurm fixture: nine cases, eighteen dummy tasks. A deliberately
  queued old task retained v3 after the switch; the ninth case used v5; one
  intentional task failure did not block continuation. No scientific credit.
- Fixture transition 27415987 and final audit 27416806 both COMPLETED 0:0.
  qualification/FIXTURE_PASS.json records the successful lifecycle.
- One-time production cutover 27416915 COMPLETED 0:0 in six seconds. Only pending
  handoffs were briefly held; all eight were released to normal dependencies.
- Boundary cursor 125; eight legacy active cases retained. No plan, intent, ACK,
  original authorization, accepted archive or result receipt was rewritten.
  Only the mutable RUN.json route changed, supported by additive immutable
  CACHE_ADOPTION.json and CACHE_ADOPTION_APPLIED.json in the production root.
- Post-cutover: all eight legacy assignment fingerprints unchanged; 102 numerical
  workers running, 12 pending, eight dependency-pending handoffs, 99 full cases
  in the completion index. No new v5 production case had been admitted at that
  observation: it will start automatically at the next available case slot.
- No unsuccessful compute recovery attempts. The unrelated old campaign remains
  paused. Existing held/incomplete cases were not reopened or recalculated.

Production root: /orcd/data/juanes/001/shaowen/phase1_pc40_adaptive10_campaign_20261030_v1.
Runner root: /orcd/data/juanes/001/shaowen/phase1_pc40_adaptive10_runner_20261030_v5.

Do not rerun the production seed or cutover. Existing Slurm gates provide the
continuation. Do not remove v3 while any pre-switch worker can still start.
Future submissions use the pinned v5 launcher, so later changes to RUN.json
cannot silently reroute them. Preserve all source packages and qualification
evidence. Any interrupted submission must be reconciled using its existing
immutable intent/ACK rather than submitted again.
