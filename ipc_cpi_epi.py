#!/usr/bin/env python

misprediction_penalty = 8

import sys
import os
import math

directory = sys.argv[1]

# find P1 and P2 latency
p1_latency = 0
p2_latency = 0
for filename in os.listdir(directory):
    if not filename.endswith(".out"):
        continue

    filepath = os.path.join(directory, filename)
    if not os.path.isfile(filepath):
        continue

    with open(filepath, 'r') as f:
        line = f.readline().strip()
        if not line:
            continue

        name, instr, branch, condbr, npred, diverge, misp, p1_lat, p2_lat, epi = line.split(',')

        # round latency to an integer number of clock cycles
        p1_latency = max(p1_latency,math.ceil(float(p1_lat)))
        p2_latency = max(p2_latency,math.ceil(float(p2_lat)))

# print("P1 latency = {}".format(p1_latency))
# print("P2 latency = {}".format(p2_latency))

# calculate average IPC, CPI, EPI
count = 0
avg_IPC = 0
avg_CPI = 0
avg_EPI = 0
for filename in os.listdir(directory):
    if not filename.endswith(".out"):
        continue

    filepath = os.path.join(directory, filename)
    if not os.path.isfile(filepath):
        continue

    with open(filepath, 'r') as f:
        line = f.readline().strip()
        if not line:
            continue

        name, instr, branch, condbr, npred, diverge, misp, p1_lat, p2_lat, epi = line.split(',')

        instructions = float(instr)
        pred_cycles = float(npred)
        divergences = float(diverge)
        p2_mispredictions = float(misp)

        # dynamic energy per instruction
        EPI = float(epi)

        # mispredictions per instruction
        MPI = p2_mispredictions / instructions

        # total cycles when the misprediction penalty is null
        # (predictor latency might be null with ahead pipelining)
        if p2_latency <= p1_latency:
            # ignore P1, P2 is sufficient
            cycles = pred_cycles * max(1,p2_latency)
        else:
            cycles = pred_cycles * max(1,p1_latency) + divergences * p2_latency

        # throughput in instructions predicted (P2) per cycle
        IPC = instructions / cycles

        # cycles lost per correct-path instruction because of mispredictions
        CPI = MPI * (misprediction_penalty + p2_latency)

        #print(f"{name},{IPC:.6f},{CPI:.6f},{EPI}")

        count += 1
        avg_IPC += 1 / IPC # harmonic mean
        avg_CPI += CPI # arithmetic mean
        avg_EPI += EPI # arithmetic mean

avg_IPC = count / avg_IPC
avg_CPI = avg_CPI / count
avg_EPI = avg_EPI / count

print(f"{avg_IPC:.6f},{avg_CPI:.6f},{avg_EPI:.6f}")
