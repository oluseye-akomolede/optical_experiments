# The Jetson GPU worker

One long-lived process on the Orin that claims GPU jobs from the web app, runs
the production `solve()` on them, and uploads the results. It is the thing that
was missing between "the app can route a job to the Jetson" (part 2/3) and "a
job actually runs on the Jetson".

It is **warm on purpose**: the module imports and the CUDA context are created
once at startup and reused for every job. Re-initialising per job is the cost
this design exists to avoid, which is also why a failed job must never take the
process down.

## What the lead does to deploy it

Three steps. Replace `<user>` with the desktop user the Orin runs as.

**1. Put the whole reviewed tree and the unit where the unit expects them.**

The unit runs `~/mse-worker/app/worker/mse_worker.py`, and the worker imports
the solver from `../dockerfiles/maxwell_solver` **relative to its own file**
(`_SOLVER_DIR = Path(__file__).resolve().parent.parent / "dockerfiles" /
"maxwell_solver"`). So the archive must be the whole tree: copying
`mse_worker.py` on its own puts the script at `~/mse-worker/mse_worker.py`,
which makes `_SOLVER_DIR` resolve to `~/dockerfiles/maxwell_solver` — a
directory that does not exist, and the failure appears at import time as a
missing module.

```sh
mkdir -p ~/mse-worker ~/.config/systemd/user
git archive --format=tar <reviewed-sha> | tar -x -C ~/mse-worker/app --strip-components=0
# ...or from a checkout:  git -C <checkout> archive <sha> | tar -x -C ~/mse-worker/app
cp worker/mse-worker.service ~/.config/systemd/user/
```

Two directories are symlinked rather than copied, so the deploy names the bench
artefacts instead of duplicating them:

```sh
ln -sfn ~/mse-bench/gpu-venv          ~/mse-worker/venv       # the GPU venv
ln -sfn ~/mse-bench/cudss-jetson-071  ~/cudss-0.7.1           # the Jetson cuDSS tree
```

`~/cudss-0.7.1` is the path the unit's `LD_LIBRARY_PATH` and `CUDSS_MT_LIB`
already name, which is why the symlink is at the home directory and not inside
`~/mse-worker`. It must be NVIDIA's **Jetson aarch64** cuDSS 0.7.1 — the pip
`nvidia-cudss-cu12` wheel is an SBSA build and aborts in `cudss.create()`; the
same goes for the pip cublas/nvrtc wheels.

**2. Create the env file the token lives in — mode 600.**

```sh
install -d -m 700 ~/.config/mse-worker
cat > ~/.config/mse-worker/env <<'EOF'
MSE_URL=https://mse.taildfdd3b.ts.net
MSE_WORKER_TOKEN=<the token the app was given>
EOF
chmod 600 ~/.config/mse-worker/env
```

The worker **refuses to start** if this file is readable by anyone else, and
refuses if `MSE_URL` or `MSE_WORKER_TOKEN` is missing. Both refusals are
deliberate: a worker that starts, silently claims nothing and looks healthy is
how the Jetson sits idle for a day with nobody noticing. The token is never
logged, never in an exception message and never in the `Config` repr.

**3. Start it.**

```sh
systemctl --user daemon-reload
systemctl --user enable --now mse-worker
journalctl --user -u mse-worker -f
```

It is a **user** unit, so it stops when the user logs out unless lingering is
on. **Lingering is currently OFF on the Orin** (`loginctl show-user <user> -p
Linger` reports `Linger=no`), and the one-time fix needs root:

```sh
sudo loginctl enable-linger <user>
```

Our rules forbid us from running `sudo`, so **whether to enable it is the
owner's call, not the deployer's**. Until it is enabled the worker runs only
while that desktop session is logged in, and a logout silently stops it —
which matters because the failure is quiet: the Jetson simply stops claiming
jobs and the queue backs up with nothing reporting an error. If the owner does
not want lingering, the worker needs a login session kept open instead.

## Running it by hand

```sh
~/mse-worker/venv/bin/python ~/mse-worker/app/worker/mse_worker.py --once  # one job
~/mse-worker/venv/bin/python ~/mse-worker/app/worker/mse_worker.py         # the loop
```

By hand it must be that interpreter explicitly. The script's
`#!/usr/bin/env python3` shebang resolves to the system python, which has no
cupy and no nvmath, so running it directly dies at import with a traceback that
looks like a code fault rather than a deployment one.

## The bench lock

The worker takes an exclusive `flock` on `~/mse-bench/.gpu.lock` and holds it
for the whole time it is solving; it does not even *claim* a job while someone
else holds that lock. The lead's own benchmark runs take the same lock.

**The lock must be free before the service will take work.** If the worker is
idle while a bench run is in progress, that is the normal state, not a fault —
`journalctl --user -u mse-worker` says so explicitly (`... is held; not
claiming`). `flock` and not a sentinel file, because the kernel drops it when
the holder dies: a SIGKILLed worker or a crashed bench run cannot wedge the GPU
permanently.

## Why the versions in the unit matter

The unit pins a specific Jetson CUDA stack:

- **cuDSS 0.7.1** from NVIDIA's Jetson aarch64 redist — **not** the pip
  `nvidia-cudss-cu12` wheel, which is an SBSA build and **aborts** (not errors)
  inside `cudss.create()`.
- The same goes for the pip `cublas` / `nvrtc` wheels.
- `MemoryMax=5G` is a cgroup limit and must stay one. `ulimit -v` (an
  address-space limit) stops CUDA from initialising at all on this box, which is
  why the unit uses `MemoryMax` and the bench runs use
  `systemd-run --user --scope -p MemoryMax=5G`.

The full version matrix and the failures each wrong version produces are in
`scratch/gpu-reference/README.md`.

## Disk hygiene

A 0.2 mm job writes ~500 MB of intermediates. So:

- **On success** the job's work directory is deleted after its results are
  uploaded.
- **On failure** the work directory is kept, renamed `<job_id>.failed`, but only
  the **most recent** one — older failure directories are deleted as each new
  failure lands. `keep_failed` in the env file sets how many to keep (default
  1).

Work root is `~/mse-worker/work`, mode 700, overridable with `MSE_WORK_ROOT`.

## Terminal vs transient failures, and why there are two verbs

The worker tells the app *why* a job did not finish, and the app has two routes
for the two cases. Getting them the wrong way round is the bug this design
exists to prevent, so the choice is one tested function (`classify` in
`mse_worker.py`) rather than a judgement at each call site.

| the worker sends | when | what the app does |
|---|---|---|
| `POST /worker/jobs/<id>/failed` | the job itself is the problem: a mesh the mesher refuses, a box too detailed, an eigenproblem with no solution, a malformed upload | marks the row **FAILED** with the message, clears the lease, and **never requeues** — the owner reads the message on the job page |
| `POST /worker/jobs/<id>/release` | the *worker* could not finish right now: SIGTERM mid-job, a GPU that would not initialise, a dropped connection, a failed upload | requeues with no verdict, and **counts the release** |

Why not one route: a deterministic failure sent to a requeueing route comes
straight back to a worker that fails it the same way, forever, with the reason
visible only in a log the owner never reads. That loop is the defect.

**The bound.** A release is legitimate, but a bounded number of them. After
`MAX_RELEASES` (3, in `web/db.py`) the app fails the job itself with a plain
message, so even a worker that flaps forever cannot loop. The bound is enforced
as one atomic `UPDATE` with the count in the `WHERE` clause, because a
read-count-then-decide form lets two concurrent releases both requeue.

## Stopping the worker: what a stop can and cannot interrupt

`systemctl --user stop mse-worker` sets a flag the worker checks **after
`solve()` returns**. That is a real limitation and it is worth stating plainly
rather than implying a stop is always graceful:

**A stop that lands while the worker is between operations** (fetching input,
uploading, idle, waiting on the bench lock) takes effect immediately: the job is
released through the transient path, the lease is cleared, the bench lock is
dropped, and the job is claimable again at once.

**A stop that lands in the middle of a long C-level eigensolve cannot interrupt
it.** The spectral solve runs inside a compiled BLAS/cuDSS call; Python's signal
handler runs only between bytecodes, so the flag is set and then nothing happens
until the solve itself returns. In practice:

1. `systemctl stop` sends SIGTERM and waits.
2. The solve keeps running. After `TimeoutStopSec=120` the unit is **SIGKILLed**
   (`Result=timeout`, `status=9`) — this is what the owner observed.
3. No `release` and no `failed` is sent, so the lease is simply left to expire.
4. The app's tick reclaims the expired lease, which requeues the job and
   **counts the release**, bounded by `MAX_RELEASES` (3). Past the bound the app
   fails it with a plain message rather than looping.

So a stop mid-solve is **safe but slow**: the work in flight is lost, the job
comes back to the queue (or fails at the bound after repeated stops), and
nothing is stranded. The steps in 3 and 4 are the paths `tests/pg_race_check.py`
C1/C2 exercise on real Postgres, and the SIGKILL leaves no lock behind because
the bench lock is an `flock` the kernel drops with the process.

**What is NOT yet true.** A stop cannot make the worker report promptly: there
is no window where it can say "I am stopping" while a solve is in flight. The
design that would fix that — running each solve in a CHILD process the worker
can terminate on SIGTERM, then `/release` immediately — is proposed and NOT
built. It would also isolate solver crashes and GPU memory from the worker
process, which are separate wins.

**Every path back to claimable counts, not just `/release`.** A worker that dies
mid-job never calls anything — the row returns through lease **expiry**. That
path increments the same counter and hits the same bound, or a crash-looping
worker would requeue forever through the timeout.

**The message the owner sees is the one from the run that finally failed**, not
the first one — it names the cause that ended the job.

## Configuration

Everything is read from `~/.config/mse-worker/env` at startup. Beyond
`MSE_URL` and `MSE_WORKER_TOKEN`:

| key | default | what it does |
|---|---|---|
| `MSE_WORK_ROOT` | `~/mse-worker/work` | where job directories live |
| `MSE_LOCK_PATH` | `~/mse-bench/.gpu.lock` | the bench lock |
| `MSE_POLL_SECONDS` | `5` | how often to poll when the queue is empty |
| `MSE_BEAT_SECONDS` | `20` | how often the heartbeat renews the lease **while solving** |
| `MSE_KEEP_FAILED` | `1` | how many failed work directories to keep |

`MSE_BEAT_SECONDS` is the worker's own clock, deliberately not derived from the
app's lease: the lease is how long the row survives a worker that died and wants
to be generous, while the beat only has to be short enough that a blip still
leaves a renewal inside the window. If `beat_seconds` is not shorter than the
lease, the worker says so and falls back to `lease/3`.

There is no worker-side retry backoff any more. It existed only to stop the loop
re-claiming a job it had just failed; `/failed` is terminal now, so there is
nothing to back away from.

## What is proved here, and what is a deploy-time fact

**Proved on the VM** (`tests/test_worker.py` and `tests/test_routing.py`, all
tier-1, against a real HTTP server on a real socket through the real routes):

- the happy path: claim → input → solve → upload → done → work dir deleted
- a **terminal** failure (a solver exception) ends the job: the row is FAILED
  with the message, the lease is cleared, and the worker's next claim gets
  **204** — not the same job again
- a **transient** failure (a dropped connection) is handed back instead, the
  release is counted, and the retry succeeds
- **three releases plus one** end the job: the app fails it at the bound, so a
  flapping worker cannot loop
- SIGTERM mid-job, **arriving while the worker is between operations**,
  **releases** the job (not fails it) and leaves no dangling lease — a service
  restart must not destroy a good job. (What happens when it arrives DURING a
  solve is a different and weaker guarantee — see "Stopping the worker" below.)
- an **upload** failure reports `failed`, never `done`
- the worker makes no claim at all while the bench lock is held (asserted on the
  server's request log — "did not ask" and "asked and got nothing" look
  identical otherwise)
- the token never appears in stdout, stderr or the log across a full run
- disk hygiene both ways: success deletes; a failure keeps only the newest
- the heartbeat carries the **real** stage, not a constant, and a lost lease
  stops the worker before `done`
- the failure message survives the **job status API** projection, which is where
  the owner actually reads it
- the atomic bound holds under two interleaved releases, and the **expiry** path
  counts against the same bound

The last four are in `tests/test_routing.py` against the app, because the verbs
are the app's. Mutations that make `/failed` requeue, or remove the bound from
the release, each turn those red.

**NOT proved here — deploy-time facts:**

- that the systemd unit starts, and that `MemoryMax`/`Nice` take effect
- that a real `flock` serialises against the lead's benchmark runs
- that the cuDSS GPU path works at all on the Orin
- that the real Postgres `release_jetson`/`reclaim_expired_leases` SQL behaves as
  its double does — the doubles model the contract (the bound, the count on
  every path), and the real SQL is verified by reading, not by running

The unit and the real lock are written and reviewed here; they are proven on the
Jetson, not on the VM.
