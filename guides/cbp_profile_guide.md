

# cbp_profile — CBP-NG Predictor Profiling Harness

This tool runs CBP-NG traces through your `branch_predictor` and prints:

* The **standard CBP CSV line** (exactly what `harcom_superuser` emits in `--format csv`)
* A **score breakdown** derived from that CSV (same equations as `predictor_metrics.py` for a *single trace*)
* Optional **per-function analysis**:

  * average energy per call for each predictor function
  * (model) latency for predict/reuse calls from HARCOM’s `val<1>` timing
  * optional wall-clock timing (“host time”) per call to help optimize C++ runtime

The profiler is designed to work without editing `cbp.hpp` or HARCOM files.

---

## Build

```bash
g++ -std=c++20 -O3 -o cbp_profile cbp_profile.cpp -lz
```

---

## Quick Start

### Accuracy-only (fast-ish, long runs)

Use this when iterating predictor logic. It prints **P1/P2 conditional accuracy** (measurement window only).

```bash
./cbp_profile --format csv --mode acc --no-score <trace.gz> <trace_name> <warmup_instr> <meas_instr>
```

Example:

```bash
./cbp_profile --format csv --mode acc --no-score ./gcc_test_trace.gz gcc 1000000 40000000 \
  1> run.csv 2> acc.txt
```

### Analyze (short runs)

Use this when you want to see **which functions burn energy**, and what latency is being reported.

```bash
./cbp_profile --format csv --mode analyze --profile <trace.gz> <trace_name> <warmup_instr> <meas_instr>
```

Example:

```bash
./cbp_profile --format csv --mode analyze --profile ./gcc_test_trace.gz gcc 50000 500000 \
  1> run.csv 2> analysis.txt
```

### Analyze + host timing (optional)

Adds wall-clock time per call (microseconds) for *all* functions, including `update_*`.

```bash
./cbp_profile --format csv --mode analyze --profile --host-timing ./gcc_test_trace.gz gcc 50000 500000 \
  1> run.csv 2> analysis.txt
```

---

## Command-Line Options

* `--format csv|human`

  * `csv`: prints CBP’s single CSV line to stdout (and lets us compute score breakdown)
  * `human`: doesn’t print the CSV line; mostly useful if you just want CBP’s native prints
* `--mode acc|analyze`

  * `acc`: prints P1/P2 **conditional accuracy**
  * `analyze`: enables per-function table if `--profile` is set
* `--profile`

  * enable per-function energy/latency accounting (only meaningful in `--mode analyze`)
* `--host-timing`

  * adds wall-clock time per function call (µs). Helpful for optimizing your C++ code’s runtime.
* `--no-score`

  * disables score breakdown (faster, avoids capturing stdout)

Positional args:

```
<trace.gz> <trace_name> <warmup_instr> <meas_instr>
```

**Warmup behavior:** measurement starts once `ninstr > warmup_instr`. Stats are reset at that boundary.

---

## Outputs

You will usually see outputs in **two streams**:

* **stdout**: the CBP CSV line (one line)
* **stderr**: human-readable breakdown + accuracy/profile tables

You can split them cleanly:

```bash
./cbp_profile ... 1> run.csv 2> report.txt
```

---

# 1) CBP CSV Line (stdout)

Example:

```
gcc,500000,105541,81142,78597,5044,2266,1347,2628,0.88,1.80667,1290
```

Field order:

```
name,
Ncp_instr,
Nbranch,
Ncondbr,
Nblock,
Textra,
Nshort,
Ncoincide,
Nlong,
p1_lat_max,
p2_lat_max,
EPIcbp
```

Meaning (high-level):

* **Ncp_instr**: number of correct-path instructions (measurement window)
* **Nblock**: number of prediction blocks (each block corresponds to one “predict cycle step” in the model; within a block, later instructions use reuse_predict*)
* **Textra**: extra prediction cycles (divergence / block boundary effects)
* **Nlong**: count of “long” events = **P2 mispredictions on conditional branches**
* **p1_lat_max / p2_lat_max**: maximum reported P1/P2 latency (cycles, can be fractional)
* **EPIcbp**: dynamic energy per instruction (fJ/instr) from HARCOM

---

# 2) Score Breakdown (stderr)

Example snippet:

```
=== Score breakdown (from CBP CSV counters) ===
Ncp(instr)=500000  Nblock=78597  AvgBlockLen=6.36  Textra=5044
CondBr=81142  Nshort=2266  Ncoincide=1347  Nlong=2628  LongRate=3.239%
Latency(max): p1=0.880cy  p2=1.807cy  =>  L1=1  L2=2
IPCcbp=5.758644  CPIcbp=0.052560  EPIcbp=1290.0 fJ/instr
Derived: MPI=0.005256  WPI=0.302674  IPC=4.420632
=== end ===
```

These match the *single-trace* computation used in `predictor_metrics.py`:

* **L1/L2** are `ceil(p1_lat_max)` and `ceil(p2_lat_max)`
* **IPCcbp**: correct-path instructions per “prediction cycle” model cycle
* **MPI**: mispredictions per instruction = `Nlong / Ncp_instr`
* **CPIcbp**: `MPI * (misprediction_penalty + L2)` where penalty is 8
* **IPC**: combined throughput once wrong-path penalty is accounted for
* **WPI**: wrong-path instructions per correct-path instruction (proxy for wasted work)

### How to use this

* Improving **P2 accuracy** reduces `Nlong` ⇒ reduces MPI and CPIcbp
* Reducing **p2_lat_max** might reduce `L2` (after ceil), which reduces CPIcbp *and* can improve IPCcbp
* Reducing **EPIcbp** reduces energy term in the final VFS scoring

---

# 3) ACC Mode Accuracy Output (stderr)

Example:

```
=== Accuracy (measurement window, conditional branches) ===
P1_cond_acc=0.951234 (12345/12980)
P2_cond_acc=0.967890 (12560/12980)
=== end ===
```

* This is **direction accuracy on conditional branches** within the measurement window.
* Computed by comparing last P1/P2 prediction bits for each instruction against the ground-truth `taken` in `update_condbr`.

**Important:**

* `harcom_superuser`’s CSV can tell you **P2** mispred rate (`Nlong/CondBr`) but it cannot tell you **P1** accuracy directly.
* ACC mode exists mainly to get **P1 accuracy**.

---

# 4) ANALYZE Mode Per-Function Table (stderr)

Example header:

```
Function         Calls      E_avg(fJ)     E_sum(fJ)   LatAvg(cy)  LatMax(cy)  CondAcc(ok/total)
```

Functions:

* `predict1`
* `reuse_predict1`
* `predict2`
* `reuse_predict2`
* `update_condbr`
* `update_cycle`

### Energy columns (all functions)

* **E_sum(fJ)**: total HARCOM dynamic energy attributed to that function
* **E_avg(fJ)**: average energy per call = `E_sum / Calls`

Use this to answer:

* “Which function burns the most energy?”
* “Is my reuse path really cheap?”
* “Is update doing something unexpectedly expensive?”

### Latency columns (predict/reuse only)

* **LatAvg(cy)**: average latency in cycles derived from HARCOM timing embedded in returned `val<1>`
* **LatMax(cy)**: maximum latency in cycles (should match or be close to CSV `p1_lat_max` / `p2_lat_max`)

For `update_*` these print `-` because there’s no timing to read (returns `void`).

### CondAcc(ok/total) (predict/reuse only)

* Conditional accuracy for each *function flavor* (predict vs reuse) as attributed at `update_condbr`.

This helps answer:

* “Does the reuse path degrade accuracy?”
* “Is P2 adding value over P1 in my design?”

---

## Host Timing Columns (optional)

If you pass `--host-timing`, the table also includes:

* **HostAvg(us)**: average wall-clock microseconds per call
* **HostMax(us)**: worst-case wall-clock microseconds per call

These are NOT part of CBP’s timing model. They’re for:

* speeding up your simulation
* identifying expensive branches in your C++ code (allocations, big loops, cache misses, etc.)

---

## What matters for the official score?

From the CBP CSV equations:

* **P2 mispredictions (`Nlong`)** matter directly (CPIcbp).
* **Maximum P1/P2 latency** matters after ceil (`L1`, `L2`).
* **EPIcbp** matters directly (energy term).

`update_condbr` and `update_cycle` do not contribute a direct “latency” term in the harness model since they return void, but:

* they can still affect **energy**
* and they can affect correctness/learning, which changes mispred count

---

## Recommended workflow

1. **Iterate correctness/accuracy fast**

```bash
./cbp_profile --format csv --mode acc --no-score ./gcc_test_trace.gz gcc 1000000 40000000 \
  1> run.csv 2> acc.txt
```

2. When accuracy looks good, run **short analyze**

```bash
./cbp_profile --format csv --mode analyze --profile ./gcc_test_trace.gz gcc 50000 500000 \
  1> run.csv 2> analysis.txt
```

3. If you’re optimizing runtime, add host timing:

```bash
./cbp_profile --format csv --mode analyze --profile --host-timing ./gcc_test_trace.gz gcc 50000 500000 \
  1> run.csv 2> analysis.txt
```

---

## Common interpretation tips

* If `predict2 E_avg` dominates and reuse is tiny:

  * optimize P2’s core access pattern (history folding, table reads, tag compares)
* If `reuse_predict2` is not much cheaper than `predict2`:

  * your reuse path might be doing too much work; cache computed indices/tags or skip recomputation
* If `update_cycle` energy is large:

  * you might be doing heavy work once per block; consider amortizing or lazily updating
* If `LatMax(cy)` is just barely >1.0 and ceil makes it 2:

  * shaving a small amount of timing can reduce `L2` by 1 full cycle in the score

---

If you want, paste one full analyze table + the CSV line and I’ll tell you “top 3 most actionable edits” based on energy share, latency ceiling effects, and accuracy deltas.
