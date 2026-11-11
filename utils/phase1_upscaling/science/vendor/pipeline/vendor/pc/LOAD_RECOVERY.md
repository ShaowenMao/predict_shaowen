# Isolated package-load recovery

Pilot 22392491 failed at package loading before numerical tests or any real-core
Pc calculation. Its JULIA_LOAD_PATH contained only @stdlib, which omitted even
the explicitly selected local project. Array 22392492 was automatically
cancelled without running.

The job environment now uses @:@stdlib: the selected project plus standard
libraries, still excluding the user's global environment. The --project argument
continues to select this sealed release. No package installation is required.

Only validation/job.sh changes the runtime behavior. All src/, vendor/, the
standalone CLI and numerical validation scripts remain byte-for-byte unchanged.
A regression test verifies project visibility without adding the user environment.
The original source and failed outputs are retained. There are no completed
cluster numerical comparisons to rerun.
