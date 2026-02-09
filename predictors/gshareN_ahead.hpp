#include "../cbp.hpp"
#include "../harcom.hpp"

using namespace hcm;

// This predictor uses ahead indexing:
//
//   [1] Seznec et al., "Multiple-block ahead branch predictors", ASPLOS 1996.
//   [2] Michaud et al., "Exploring instruction-fetch bandwidth requirement in
//       wide-issue superscalar processors", PACT 1999.
//   [3] Seznec & Fraboulet, "Effective ahead pipelining of instruction block
//       address generation", ISCA 2003.
//
// Instead of indexing block B1's prediction with the address of block B1, we index it
// with the address of the previous block, B0, and the path followed out of block B0.
// N is the maximum number of branches predicted per cycle (at most one taken branch).
// Unless block B0 ends on an indirect jump, there are at most N+1 paths out of B0.
// The N+1 possible block predictions are read simultaneously from multiple banks.
// Once the path is known in the next cycle, it is used to select one of the N+1 block predictions.
// As the N+1 paths out of B0 are not equally likely, and in order to use storage evenly,
// the bank associated with a given path depends on some bits XB of B0's address via XORing.
// Each of the N predictions for B1 is associated with a lane.
// To use lanes evenly, the lane depends on some bits XL of B1's address via XORing.
//
// Indirect jumps, in particular function returns, degrade prediction accuracy.
// This problem can be mitigated by treating function calls as if they were conditional branches.
// When a block B0 ends on a call, the block prediction associated with the call
// being not-taken is saved in a stack. When it is time to get predictions for the block
// after the function return, the prediction at the top of the stack is substituted for the
// prediction obtained with ahead-indexing.


template<u64 LOGG=19, u64 GHIST=8, u64 N=4>
struct gshareN_ahead : predictor {
    // gshare with 2^LOGG entries, single prediction level (no overriding)
    // global history of GHIST bits
    // predicts up to N branches per cycle
    static constexpr u64 LOGLANES = std::bit_width(N-1);
    static constexpr u64 LANES = u64(1) << LOGLANES;
    static constexpr u64 PATHS = N+1;
    static constexpr u64 LOGBANKS = std::bit_width(PATHS-1);
    static constexpr u64 BANKS = u64(1) << LOGBANKS;
    static_assert(LOGG>(LOGLANES+LOGBANKS));
    static constexpr u64 index_bits = LOGG-(LOGLANES+LOGBANKS);
    // a block does not continue past a line boundary
    static constexpr u64 LINEINST = 64; // line size in instructions
    static_assert(std::has_single_bit(LINEINST)); // power of 2
    static constexpr u64 LOGLINEINST = std::bit_width(LINEINST-1);

    reg<GHIST> global_history;

    // pipelined over 2 cycles ([0]=current block, [1]=previous block)
    reg<index_bits> index[2];
    reg<LOGBANKS> XB[2]; // address bits for selecting the bank
    reg<LOGLANES> XL; // address bits for selecting the lane (current block)
    reg<LOGBANKS> path; // path out of previous block

    arr<reg<LANES>,BANKS> block_pred[2]; // read block predictions
    arr<reg<1>,LANES> unordered_pred; // read prediction bits for the current block, unordered
    arr<reg<1>,LANES> pred; // read prediction bits for the current block, ordered

    ram<arr<val<LANES>,BANKS>,(1<<index_bits)> ctr_hi; // prediction bits

    // simulation artifacts, hardware cost not modeled accurately
    reg<1> true_block = 1;
    reg<LINEINST> block_entry; // one-hot vector
    u64 num_branch = 0; // number of conditional branches in block so far
    u64 block_size = 0; // instructions in current block so far
    arr<reg<1>,N> branch_dir; // actual branch direction

    zone UPDATE_ONLY;
    ram<val<1>,(BANKS<<index_bits)> ctr_lo[LANES]; // hysteresis bit (1=weak, 0=strong)

    val<1> predict1([[maybe_unused]] val<64> inst_pc)
    {
        // new block
        assert(num_branch==0);
        block_size = 1;
        index[1] = index[0];
        val<index_bits> pc_bits = inst_pc >> (LOGBANKS+2);
        if constexpr (GHIST <= index_bits) {
            index[0] = pc_bits.fo1() ^ (val<index_bits>{global_history}<<(index_bits-GHIST));
        } else {
            index[0] = global_history.make_array(val<index_bits>{}).append(pc_bits.fo1()).fold_xor();
        }
        index[0].fanout(hard<2>{});
        XL = inst_pc >> 2;
        XL.fanout(hard<LANES>{});
        XB[1] = XB[0].fo1();
        XB[0] = inst_pc >> 2;
        XB[1].fanout(hard<2>{});
        block_pred[1] = block_pred[0].fo1();
        block_pred[0] = ctr_hi.read(index[0]);
        block_pred[1].fanout(hard<2>{});
        unordered_pred = block_pred[1].select(path^XB[1]).make_array(val<1>{});
        unordered_pred.fanout(hard<LANES>{});
        for (u64 i=0; i<LANES; i++) {
            pred[i] = unordered_pred.select(i^XL);
        }
        pred.fanout(hard<LINEINST*2>{});
        block_entry = val<LOGLINEINST>{inst_pc>>2}.decode().concat();
        block_entry.fanout(hard<LINEINST>{});
        reuse_prediction(~val<1>{block_entry>>(LINEINST-1)});
        return pred[0];
    };

    val<1> reuse_predict1([[maybe_unused]] val<64> inst_pc)
    {
        reuse_prediction(~val<1>{block_entry>>(LINEINST-1-block_size)});
        block_size++;
        return pred[num_branch];
    };

    val<1> predict2([[maybe_unused]] val<64> inst_pc)
    {
        return pred[num_branch];
    }

    val<1> reuse_predict2([[maybe_unused]] val<64> inst_pc)
    {
        return pred[num_branch];
    }

    void update_condbr([[maybe_unused]] val<64> branch_pc, val<1> taken, [[maybe_unused]] val<64> next_pc)
    {
        assert(num_branch<N);
        branch_dir[num_branch] = taken.fo1();
        num_branch++;
        if (num_branch == N) {
            // this was the last branch for which we have a prediction: end the block
            reuse_prediction(0);
        }
    }

    void update_cycle([[maybe_unused]] instruction_info &block_end_info)
    {
        val<1> &mispredict = block_end_info.is_mispredict;
        val<64> &next_pc = block_end_info.next_pc;
        if (num_branch == 0) {
            // no conditional branch in this block
            // update global history if previous block ended on a mispredicted not-taken branch
            // (we are still in the same line, this is the last chunk)
            // or if the block ends on an unconditional jump
            val<1> uncond_jump = block_end_info.is_taken & ~block_end_info.is_conditional;
            execute_if(~true_block | uncond_jump.fo1(), [&](){
                global_history = (global_history << 1) ^ val<GHIST>{next_pc.fo1()>>2};
                true_block = 1;
            });
            path = 0;
            return; // stop here
        }
        static_assert(LANES<=64);
        XL.fanout(hard<2*LANES>{});
        index[1].fanout(hard<2*LANES+1>{});
        branch_dir[num_branch-1].fanout(hard<LANES+LOGBANKS+1>{});
        mispredict.fanout(hard<4>{});

        // determine the lanes that are accessed
        arr<val<LANES>,LANES> lane = [&](u64 i) -> val<LANES> {
            return (XL^i).decode().concat();
        };
        lane.fanout(hard<2>{});
        arr<val<1>,LANES> access = [&](u64 i) -> val<1> {
            return (lane[i]>>num_branch) == hard<0>{};
        };

        // determine the lane corresponding to the mispredicted branch
        val<LANES> misp_lane = execute_if(mispredict, [&]() -> val<LANES> {
            return lane[num_branch-1];
        });
        misp_lane.fanout(hard<2>{});
        arr<val<1>,LANES> mispredicted = misp_lane.make_array(val<1>{});

        // determine the bank to update
        val<LOGBANKS> bank = path ^ XB[1];
        bank.fanout(hard<2*LANES+BANKS>{});

        // read hysteresis bit if lane corresponds to mispredicted branch
        arr<val<1>,LANES> weak = execute_if(misp_lane, [&](u64 i){
            // return 1 iff mispredict and hysteresis is weak
            return ctr_lo[i].read(concat(index[1],bank));
        });

        // we need an extra cycle if there is a mispredict
        need_extra_cycle(mispredict);
        // update prediction if mispredict and the hysteresis bit is weak
        execute_if(mispredict, [&](){
            val<LANES> block_bundle = arr<val<1>,LANES>{
                [&](u64 i){
                    return select(weak[i].fo1(),branch_dir[num_branch-1],unordered_pred[i]);
                }
            }.concat();
            block_bundle.fanout(hard<BANKS>{});
            arr<val<LANES>,BANKS> bundle = [&](u64 i){
                return select(bank==i, block_bundle, block_pred[1][i]);
            };
            ctr_hi.write(index[1],bundle.fo1());
        });

        // update hysteresis
        execute_if(access.fo1().concat(), [&](u64 i){
            ctr_lo[i].write(concat(index[1],bank),mispredicted[i].fo1());
        });

        // update the global history
        val<1> line_end = block_entry >> (LINEINST-block_size);
        true_block = (~mispredict & (num_branch<N)) | branch_dir[num_branch-1] | line_end.fo1();
        execute_if(true_block, [&](){
            global_history = (global_history << 1) ^ val<GHIST>{next_pc.fo1()>>2};
        });

        // record the path: if last cond branch not-taken, then path=0, else path=num_branch
        path = num_branch & branch_dir[num_branch-1].replicate(hard<LOGBANKS>{}).concat();
        path.fanout(hard<2>{});

        num_branch = 0;
    }
};
