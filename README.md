# CBP-NG Championship Documentation

Welcome to the repository for the Next-Generation Championship in Branch
Prediction! This championship aims to foster innovation in branch prediction by
encouraging the development of branch prediction algorithms achieving high
prediction accuracy while minimizing power consumption.

Below, you will find information designed to help you get started developing
your winning predictor design. Please see [the competition
website](https://cbp-ng.bpchamp.com) for additional information about the
championship, including how to join the mailing list for announcements and
conversation with the organizers and other participants.

## Writing a Predictor

### HARCOM Language

Predictors in the CBP-NG simulator are written using the HARCOM (HARdware
COMplexity) C++ library. Using this library allows CBP-NG to model the energy,
latency, and area costs of predictor designs.

HARCOM tracks these costs using special C++ types representing `reg`isters,
SRAM `arr`ays, and transient intermediate `val`ues. These types use operator
overloading and additional methods to allow you to implement array accesses and
needed hardware logic. These types are opaque, meaning your predictor cannot
(directly) read or write their values. For example, instead of being able to
use typical C++ 'if-else' statements, your predictor will instead need to
`select` (i.e. mux) between several opaque `val`'s based on another opaque
`val`.

Please review the full HARCOM manual, [available in the CBP-NG
repository](https://github.com/AmpereComputing/cbp-ng/raw/refs/heads/main/docs/harcom.pdf),
for a more details about how this library works and how to use it.

### Simulator Interface

The simulator interface is designed to balance flexibility of predictor design
with faithfulness to the typical external design constraints placed on branch
predictors.

The `predictor` interface your predictor must implement is specified in
[`cbp.hpp`](https://github.com/AmpereComputing/cbp-ng/tree/main/cbp.hpp). There
are code comments alongside the interface describing it, and a brief
explanation of it also follows:

There are two main 'levels' of branch prediction your predictor may implement -
levels 1 and 2. This interface structure is intended to support prediction
pipelining (for example, with a fast prediction providing throughput, but with
a slower predictor correcting the prediction if desired). The simulator calls
different methods on your predictor for each level, as described in the next
paragraph. Note that you may provide simple (zero-latency) implementations of
either level if you desire a single prediction level.

Your predictor is also responsible for choosing its own prediction block length
(the number of instructions predicted at one time). At the beginning of a
block, the simulator will call your predictor's `predict1()` (for level 1) and
`predict2()` (level 2) methods. These methods are responsible for returning the
prediction for the first instruction at each corresponding prediction level,
but they may additionally call a `reuse_prediction()` callback with a value of
`1` to specify that the next instruction be predicted by the simulator calling
`reuse_predict1()` and `reuse_predict2()` instead of `predict1()` and
`predict2()`. This prediction block will continue (the "reuse_" methods will be
called) until either you call `reuse_prediction()` with a `0`, a level-2
misprediction occurs, or a taken branch is encountered, at which point
`predict1()` and `predict2()` will be called once more.

The separation of these two sets of methods allows your predictor to both use
different logic for the first instruction in a given block (i.e. an initial
array lookup) than it does for subsequent instructions, and to directly control
the length of a block.

### Example Predictors

Sometimes an example is worth much more than an explanation. To that end,
example predictors are available in [the `./predictors` subdirectory of the
CBP-NG
repository](https://github.com/AmpereComputing/cbp-ng/tree/main/predictors).
These include bimodal, gshare, and TAGE reference predictors.

### Common Utilities 

Different predictors may re-use many of the same components. To help provide
some basic building blocks, we've gathered several potentially-common
components together into
[`predictors/common.hpp`](https://github.com/AmpereComputing/cbp-ng/blob/main/predictors/common.hpp)
in the CBP-NG repository. These common bits include helper functions for
updating saturating counters, tracking and "folding" branch history, and
reading/writing a banked RAM. Many of the example predictors in the same
directory provide examples of how to use them.

## Predictor Scoring

The competition score will take into account prediction accuracy,
energy-efficiency, and implementation complexity as part of a
"Voltage-Frequency-Scaled Speedup" (a.k.a. VFS) score. Pierre Michaud has
written [a paper available on the competition's github
repository](https://github.com/AmpereComputing/cbp-ng/blob/main/docs/vfs.pdf)
explaining this scoring to help contestants gain intuition for it.

Though we have attempted to ensure this scoring mechanism is aligned with
real-world design constraints and you are heavily encouraged to use your
creativity in maximizing it, submissions which optimize it in an unrealistic
way or which undermine the intention of the scoring will not be considered. All
submissions will be judged in the spirit of the competition: pushing the
boundaries of energy-efficient and high-performance branch prediction.

## Building and Running Predictors

### Requirements

The CBP-NG Simulator and HARCOM require either GCC version 12 or later or Clang
version 19 or later.

It is tested on Linux (Ubuntu), MacOS, and Windows (Ubuntu-based WSL).

### Compiling

Though you are welcome to compile your predictor however you wish, the
simulator repository contains a `./compile` helper script. To compile the
default predictor (i.e. that which `branch_predictor.hpp` sets the `PREDICTOR`
preprocessor macro to - `tage<>` in the main repository):

```console
./compile cbp
```

To compile a specific predictor, in this case gshare:

```console
./compile cbp -DPREDICTOR="gshare<>"
```

To add your own predictor named "my_predictor", it is suggested to add a new
file (for example, `./predictors/my_predictor.hpp`) to the repository
containing a class named `my_predictor` which inherits from `predictor`,
include your newly-created header file at the top of `branch_predictor.hpp`,
and then compile as:

```console
./compile cbp -DPREDICTOR="my_predictor<>"
```

### Simulating

Assuming your compiled binary is named `./cbp`, you can simulate one trace with
1M instructions of warmup and up to 40M instructions of measurement like:

```console
./cbp ./gcc_test_trace.gz test 1000000 40000000
```

Or use the run script like:

```console
./run ./cbp ./gcc_test_trace.gz
```

Note: the simulation output per traces is one row of CSV, with fields in the
order: (1) trace name, (2) instructions, (3) branches, (4) conditional
branches, (5) blocks, (6) extra cycles, (7) P1/P2 divergences, (8) P1/P2
divergences coinciding with block end, (9) P2 mispredictions, (10) P1 latency
(cycles), (11) P2 latency (cycles), (12) dynamic energy per instruction (fJ)

There is also a `run_all` script which may help simulate many traces at once
using the same binary. For example, to simulate all traces in the 'traces'
directory:

```console
mkdir OUTDIR
./run_all ./cbp ./traces OUTDIR
```

### Score Calculations

There are several scripts intended to help calculate scores and make sense of
the provided simulator metrics.

Once you have run a set of traces, you can compute ratios between statistics
using the `ratio` script. For example, the number of mispredictions per
instruction:

```console
./ratio 2 9 OUTDIR
```

The `predictor_metrics.py` script will compute the average IPC<sub>cbp</sub>,
CPI<sub>cbp</sub>, EPI<sub>cbp</sub> (as defined for VFS scoring, not their
usual meanings), as well as other metrics such as average counts of conditional
branches and mispredictions, across a full set of traces:

```console
./predictor_metrics.py OUTDIR
```

The output of `predictor_metrics.py` can be further used to compute the VFS
score for a set of runs via:

```console
./predictor_metrics.py OUTDIR | ./vfs.py
```

Or, if you would like to play around with the VFS score for arbitrary
IPC<sub>cbp</sub>, CPI<sub>cbp</sub>, and EPI<sub>cbp</sub> values (as defined
for VFS scoring), you may pass those (in that order) like:

```console
./vfs.py 7,0.03,1500
```

### Reference Predictor

The "reference" TAGE-SC-L predictor (from CBP 2025) was used to help us set the
prediction accuracy portion of the VFS score's reference predictor. It is
contained in `reference.cpp` and `seznec_cbp2025.h`. Note: this predictor is
not written using the CBP-NG simulator and therefore provides only prediction
accuracy (no prediction latency or energy usage). You may use it for your
research, but please be aware that all submitted predictors must be implemented
using the predictor interface detailed above, unlike the reference predictor!

To compile the TAGE-SC-L reference predictor:

```console
g++ -std=c++20 -o reference -O3 reference.cpp -lz
```

To run one trace:

```console
./reference ./gcc_test_trace.gz test 1000000 40000000
```

or

```console
./run ./reference ./gcc_test_trace.gz
```

To simulate all traces in the 'traces' directory:

```console
./run_all ./reference ./traces OUTDIR
```

## 'Training' Traces

You are encouraged to use the official set of 168 CBP-NG training traces,
available for download at
[https://drive.google.com/file/d/1kLKn_iKVBP-YxRpC4WiCy-ca-agU0BFG/view](https://drive.google.com/file/d/1kLKn_iKVBP-YxRpC4WiCy-ca-agU0BFG/view),
to develop your predictor designs. Once downloaded, you may un-compress and
extract the traces using the `tar xf cbp-ng_training_traces.tar.gz` command.
You can then point the CBP-NG simulator run script at this resulting directory
when running your simulations (see `./run_all` above for how to run multiple
traces). Though you are welcome to use other traces, we strongly recommend you
primarily use our provided training traces because we have gone to great
lengths to ensure they are representative of the set we will use for final
scoring.
