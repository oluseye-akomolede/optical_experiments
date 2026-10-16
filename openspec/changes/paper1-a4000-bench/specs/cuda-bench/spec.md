# cuda-bench

## ADDED Requirements

### Requirement: The CUDA bench image SHALL carry exactly the files the bench script imports

`dockerfiles/maxwell_solver/Dockerfile.bench-cuda` SHALL copy the repo's own import closure of
`scripts/paper1_nanowire.py` and nothing else, in the repository's relative layout, so the
image runs the same module search path the bench run was measured on. The image SHALL set no
`PYTHONPATH`: the script inserts its own `parent.parent/dockerfiles/maxwell_solver` into
`sys.path`, which resolves only if `scripts/` sits beside `dockerfiles/maxwell_solver/`.

#### Scenario: the closure is re-derived, never retyped
- **WHEN** `scripts/paper1_nanowire.py` gains a local import
- **THEN** the file list in the Dockerfile is stale by construction, and the derivation
  command recorded in the Dockerfile (the `-I -c` one-liner that loads the script and lists
  every loaded module whose file is in the repo) is what names the files to copy again

#### Scenario: the image answers --help from the bench script
- **WHEN** the built image runs with `--help` (the image's `ENTRYPOINT` is
  `python3 -u scripts/paper1_nanowire.py`, working directory `/opt/app`)
- **THEN** the script's own usage prints, so the imports it makes at module level resolved
  from the copied tree

### Requirement: The image SHALL pin its CUDA base and every package to the versions the bench path has been run with

The base tag SHALL be pinned at or below the CUDA a target node reports, and the four python
packages SHALL be pinned to the versions under which the bench's own numbers were produced, so
an A4000 reading is comparable with the Orin's.

#### Scenario: the pins name their source
- **WHEN** the pins are checked
- **THEN** the base is `nvidia/cuda:12.6.3-runtime-ubuntu24.04` (the A4000 nodes report
  `nvidia.com/cuda.runtime-version.full=12.7`, driver 565.57.01) and the packages are
  `cupy-cuda12x==14.2.0`, `numpy==2.2.6`, `scipy==1.15.3`, `trimesh==5.1.0` — read off the
  Jetson venv that ran r20, not chosen from a release page

#### Scenario: cupy imports without a GPU
- **WHEN** the image is run on a host with no GPU
- **THEN** `python3 -c "import cupy; print(cupy.__version__)"` prints the pinned version, so
  the image's risk (a wheel that cannot be installed or imported) is settled off the cluster

### Requirement: The A4000 Job SHALL be pinned to one GPU of one product and SHALL request exactly what it limits

`manifests/bench/paper1-a4000-job.yaml` SHALL select the GPU product and ask for one GPU,
SHALL set `requests` equal to `limits` for `nvidia.com/gpu`, `cpu` and `memory`, and SHALL set
the container's BLAS/OpenMP thread counts to its own CPU limit (a container CPU limit does not
change `os.cpu_count()`, so an unset thread count starts one thread per host core and is
throttled). Which node carries the Job SHALL be left to the scheduler and the card's own
allocatable, and the file SHALL say where a host pin would go.

This requirement SHALL NOT be read as admissibility. The namespace's quota admits no GPU
today, and this change SHALL NOT raise it.

#### Scenario: the GPU quota is a live obstacle, not a met precondition
- **WHEN** the namespace's ResourceQuota `mse-power-cap` is read
- **THEN** `requests.nvidia.com/gpu` is hard 0 / used 0, and a 1-GPU pod is refused
  (`exceeded quota: mse-power-cap, requested: requests.nvidia.com/gpu=1, used: 0, limited: 0`)
- **AND** raising it 0 → 1 SHALL be the owner's decision and the lead's apply, never this
  change's own act
- **AND** `kubectl apply --dry-run=server` SHALL NOT be read as admissibility for this
  manifest: ResourceQuota admission is exempt from dry-run, so dry-run green is not a pass

#### Scenario: the memory limit is derived from the priced grid
- **WHEN** the priced dx-10 record with the boundary held (14,048,300 cells) is compared with
  the memory limit
- **THEN** the estimator's own storage for it is 1,754,244,508 bytes = 1,673.0 MiB = 1.63 GiB:
  20.4 % of the 8Gi request (8,192 MiB, so the request is 4.9x the estimate) and 10.2 % of the
  card's 16,376 MiB -- both denominators named, because "15 % of the card" was wrong

#### Scenario: the args are the launcher and they name cupy
- **WHEN** the Job's `args` are read
- **THEN** they carry the priced record in full (`--wavelength`, `--dx`, `--mirror-pad`,
  `--periods`, `--xp cupy`, and the open boundary held at r20's own distance through `--pad-nm`
  and `--pml-nm`) plus the three raised bench ceilings `--max-cells`, `--max-steps` and
  `--max-seconds`, because on the cluster there is no `run_cw.sh`: the manifest is the launcher,
  and the script's default `--xp numpy` would run at about a third of the table's rate with
  nothing to refuse it

### Requirement: The Job's wall bound SHALL be its own activeDeadlineSeconds, derived from the same estimator's seconds

The Job SHALL carry `activeDeadlineSeconds` derived, and written out in the file, from the
pricing of the exact record its `args` ask for, with the margin named. A raised
`--max-seconds` SHALL NOT be presented as that bound: the product does not stop a run that has
started, so the deadline is the only live timer on the cluster.

#### Scenario: the deadline follows the record's own pricing
- **WHEN** the CW record's deadline is checked against its pricing
- **THEN** `1.5 × 17,925 s (the table's 13,662.4 s at the Orin's measured factor 1.312) + 300 s
  setup = 27,188 s`, rounded up to the next half hour, is the 28,800 s in the file, and the
  file states this arithmetic

#### Scenario: the run's own rate is visible before the deadline
- **WHEN** the Job starts
- **THEN** the script's `progress` lines report Mcell-steps/s within the first minutes (the
  Orin read 30.3 on cupy for r20), so a machine slower than the margin is seen long before the
  deadline kills the run

### Requirement: A finished or failed Job's log SHALL survive as the result

The Job SHALL NOT retry (`backoffLimit: 0`), SHALL NOT restart its container
(`restartPolicy: Never`), and SHALL keep its pod for at least a week after it finishes
(`ttlSecondsAfterFinished: 604800`), because a multi-hour run cancelled at the deadline leaves
its partial log as the only measurement there is.

#### Scenario: a cancelled run is still readable
- **WHEN** the Job is killed by its deadline
- **THEN** the pod is not deleted while the queue row that launched it is still open, so the
  progress lines up to the kill can be read

#### Scenario: the args are refused without the cell ceiling and admitted with it
- **WHEN** the manifest's dx-10 args are run against the shipped code
- **THEN** they are refused with the cell limit (`The grid has 14,048,300 cells, over the limit
  of 8,000,000`) before the scene is built, and with the file's own `--max-cells 15000000` the
  same args are admitted -- which is the resolution `tasks.md` records: the owner's decision
  was the third bench ceiling (`--max-cells`), not a coarser rung, because at the held boundary
  the dx-12 rung is 8,952,300 cells and still over the cap
