# Tasks

## 0. What was actually missing

- [x] 0.1 Establish the starting state. The blame-the-object message for an OOM
      was ALREADY fixed on 2026-09-27 (`_names_an_allocation_failure`, used by
      `owner_message`). What was missing is simply that no retry existed: an
      OOM reached the owner as a failure with no attempt at the CPU.

## 1. Fallback

- [x] 1.1 Move the OOM word list into `mse_backend` so the retry decision and
      `owner_message` read ONE list. Its words already matched both Jetson
      failures recorded on 2026-11-02 (`ALLOC_FAILED (2)` and
      `cudaErrorMemoryAllocation: out of memory`); a few extra spellings are
      kept as defence and labelled as such. Test fixtures use those two real
      strings, in both shapes (a cuDSS status under the `EigenError` wrapper,
      and the driver message in the wrapper itself).
- [x] 1.2 On a GPU OOM, retry the same problem once on the CPU, guarded by the
      worker's memory limits; a refusal or a second OOM raises.
- [x] 1.3 Record the fallback in the backend description so the run's report
      says it happened.

## 2. The owner's message

- [x] 2.1 Keep the existing OOM branch of `owner_message` (it already names the
      machine, not the object) and point it at the shared list so the retry and
      the message cannot disagree. No new sentence was needed.
- [x] 2.2 Keep the existing advice for a problem that genuinely does not fit
      before any factorisation.

## 3. Tests

- [x] 3.1 A GPU OOM falls back once and succeeds, with the reason in the log.
- [x] 3.2 An absent CUDA stack still raises, and never reaches the CPU.
- [x] 3.3 A second OOM raises instead of looping.
- [x] 3.4 An over-limit problem raises without attempting the CPU solve.
- [x] 3.5 Pin the EXISTING owner-message behaviour (2.1 added no words): an OOM
      names the machine and not the object; the over-limit message still names
      the object.
- [x] 3.6 Both observed shapes retry: the cuDSS `ALLOC_FAILED (2)` status under
      its wrapper, and `cudaErrorMemoryAllocation: out of memory` in it.

## 4. The failure that started this (job e9423834)

- [x] 4.1 Reproduce the stored generic message from the deployed worker on the
      Jetson. The real chain is NOT an OOM status: the board's memory
      carve-out was exhausted (`NvMapMemAllocInternalTagged: ... error 12`),
      so the CUDA library could not be loaded and the owner was told to "try
      a simpler object". Reproduced in `~/mse-bench/repro_e94.py` with the
      worker's own venv; the traceback is the evidence.
- [x] 4.2 Recognise the library-load failure and give it its own sentence: our
      side, not the object's, and no promise of a fix by retrying the object.
- [x] 4.3 Recognise the driver's allocator ENOMEM as a memory failure. It is
      keyed on the allocator name AND `error 12` together: the same tag reports
      other errnos, and `error 22` must stay a device fault, not "try again".
- [x] 4.4 Test both through `_owner_message` and the stored job row, with the
      strings copied from the Jetson traceback.

## 5. Reviewer blockers (aa172bd)

- [x] 5.1 `FullOnBothProcessors` (the base type, not only its subclass) gets its
      own sentence: the GPU was busy and the job was too large for the main
      processor. It no longer falls through to the "contact support" apology.
      `TooBigForProcessor` keeps its existing text.
- [x] 5.2 The test asks `owner_message` for the sentence and the worker test
      asserts the STORED row message. Checked by removing the branch: the test
      fails on the generic apology.
- [x] 5.3 The source no longer claims the app re-queues a Jetson row. It states
      what happens: the app skips Jetson rows before reading any re-route token,
      so the worker fails the row with the busy-machine message. No re-queue is
      added in this change.

## 6. Close

- [x] 6.1 `openspec validate gpu-oom-cpu-fallback --strict`; report change and
      baseline counts.
- [x] 6.2 Hand back with the sha; the lead pushes.
