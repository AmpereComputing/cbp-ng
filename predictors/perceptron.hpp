#include "../cbp.hpp"
#include "../harcom.hpp"
#include "common.hpp"

using namespace hcm;

/*
 * An implementation of a basic perceptron predictor, adapted from
 * https://www.cs.utexas.edu/~lin/papers/hpca01.pdf. This predictor maintains
 * global branch history and a table of weights, where each table entry
 * contains history_bits+1 weights. A conditional branch is predicted taken if
 * the dot product of the weights and the history is non-negative. This
 * predictor uses several "banks" (in the style of the other gshareN/bimodalN
 * example predictors), where each bank contains its own table of weights and
 * predicts one conditional branch in a given prediction block.
 *
 * The level-1 predictor always predicts not taken.
 */

template<u64 HIST_LEN=32, u64 LOGN=2, u64 LOG_ENTRIES=8, u64 LOGBBMAX=6>
struct perceptron : predictor {
    static constexpr u64 N = 1<<LOGN; // number of banks

    // Training threshold (a function of history length)
    static constexpr u64 THRESHOLD = static_cast<u64>(1.93 * HIST_LEN + 14);

    // The number of bits each weight uses (function of training threshold)
    static constexpr u64 WEIGHT_BITS = static_cast<u64>(std::bit_width(THRESHOLD));
    static constexpr u64 SUM_BITS = WEIGHT_BITS + std::bit_width(HIST_LEN + 1);
    static constexpr u64 BBMAX = 1<<LOGBBMAX; // maximum basic block size (in instructions)

    // Try to use all banks evenly. Define X as the LOGN rightmost bits of the
    // block address (inst_pc>>2), and assign the 1st pred to bank X, 2nd to
    // bank X^1, 3rd to bank X^2,...
    reg<LOGN> X;

    reg<LOG_ENTRIES> index;
    arr<reg<1>,HIST_LEN> global_history;
    arr<reg<1>, N> below_threshold;
    arr<reg<1>, N> pred_taken;

    struct {
        // Array containing all weights
        ram<arr<val<WEIGHT_BITS, i64>, HIST_LEN+1>, (1 << LOG_ENTRIES)> weights;
        // Registers holding the weights used to generate this block's
        // prediction and potentially perform an array update
        arr<reg<WEIGHT_BITS, i64>, HIST_LEN+1> branch_weights;
    } site[N]; // grouped together to reduce wire energy

    // State saved in update_condbr/predict2 for update_cycle's use
    u64 num_condbr = 0; // number of conditional branches in block so far
    u64 bb_inst = 0; // instructions in current basic block
    arr<reg<1>,N> branch_exec_taken; // executed branch direction

    void block_predict(val<64> inst_pc)
    {
        global_history.fanout(hard<N+1>{});
        inst_pc.fanout(hard<3>{});
        assert(num_condbr == 0);
        assert(bb_inst == 0);

        // Hash the instruction PC by first chunking it into an array of
        // values, then folding that array onto itself using XOR.
        index = (inst_pc >> 2).make_array(val<LOG_ENTRIES>{}).fold_xor();
        index.fanout(hard<2*N>{});

        // Determine which bank maps to which conditional branch
        X = val<LOGN*4>{inst_pc >> 2}.make_array(val<LOGN>{}).fold_xor();
        X.fanout(hard<3*N>{});

        // Approximate the dot products of the weights and global history
        arr<val<SUM_BITS, i64>, N> sums = [&](u64 i) -> val<SUM_BITS, i64>{
            site[i].branch_weights = site[i].weights.read(index);
            site[i].branch_weights.fanout(hard<2>{});

            arr<val<WEIGHT_BITS, i64>, HIST_LEN+1> flipped_weights = [&](u64 j) -> val<WEIGHT_BITS,i64> {
                if (j == 0) {
                    return site[i].branch_weights[j];
                } else {
                    auto ghbit = global_history[j-1].connect(site[i].weights);
                    ghbit.fanout(hard<WEIGHT_BITS>{});
                    return site[i].branch_weights[j] ^ ghbit.replicate(hard<WEIGHT_BITS>{}).concat();
                }
            };
            return flipped_weights.fo1().fold_add();
        };
        sums.fanout(hard<3>{});

        // Record whether the sums met the threshold to train (needed at
        // training-time).
        for (u64 i=0; i < N; i++) {
            below_threshold[i] = (sums[i] < static_cast<i64>(THRESHOLD)) & (sums[i] > -static_cast<i64>(THRESHOLD));
        }

        arr<val<1>, N> unordered_pred_taken = [&](u64 i) {
            // 1 if positive, 0 if negative
            return ~std::get<0>(split<1,SUM_BITS-1>(sums[i]));
        };
        unordered_pred_taken.fanout(hard<N>{});

        for (u64 i=0; i < N; i++) {
            pred_taken[i] = unordered_pred_taken.select(X^i);
        }
        pred_taken.fanout(hard<BBMAX*4>{}); // FIXME reduce to 2xBBMAX if/when making perceptron only second-level predictor
    };

    val<1> predict1([[maybe_unused]] val<64> inst_pc) {
        reuse_prediction(hard<1>{});
        // Always predict not taken
        return hard<0>{};
    }

    val<1> reuse_predict1([[maybe_unused]] val<64> inst_pc)
    {
        reuse_prediction(hard<1>{});
        // Always predict not taken
        return hard<0>{};
    };

    val<1> predict2([[maybe_unused]] val<64> inst_pc)
    {
        block_predict(inst_pc.fo1());
        bb_inst++;
        reuse_prediction(bb_inst < BBMAX);
        return pred_taken[num_condbr];
    }

    val<1> reuse_predict2([[maybe_unused]] val<64> inst_pc)
    {
        bb_inst++;
        reuse_prediction(bb_inst < BBMAX);
        return pred_taken[num_condbr];
    }

    void update_condbr([[maybe_unused]] val<64> branch_pc, [[maybe_unused]] val<1> taken, [[maybe_unused]] val<64> next_pc) {
        assert(num_condbr < N);

        branch_exec_taken[num_condbr] = taken.fo1();
        branch_exec_taken.fanout(hard<N>{});
        num_condbr++;
        bb_inst = 0;

        if (num_condbr == N) {
            // this was the last branch for which we have a prediction: end the block
            reuse_prediction(0);
        }
    }

    void update_cycle([[maybe_unused]] instruction_info &block_end_info) {
        block_end_info.is_taken.fanout(hard<HIST_LEN+1>{});
        if (num_condbr == 0) {
            // Update the global history upon a taken branch
            execute_if(block_end_info.is_taken, [&](){
                global_history = global_history.shift_left(val<2>{block_end_info.next_pc.fo1()>>2});
            });
            bb_inst = 0;
            return;
        }

        val<1> &mispredict = block_end_info.is_mispredict;
        mispredict.fanout(hard<N*N>{});

        // Map 3 pieces of information back to 'physical' banks:
        // 1) whether an executed branch from this prediction was mapped to
        //    that bank, and if so...
        // 2) whether the bank's branch was taken
        // 3) whether the bank's branch was mispredicted

        // Which banks had branches mapped to them?
        arr<val<N>, N> banks_with_branches_onehot = [&](u64 i) {
            // `valid_mask` will have all bits set if this was a executed in
            // this prediction block
            val<N> valid_mask = val<1>{i < num_condbr}.replicate(hard<N>{}).concat();
            // `decode` sets a bit corresponding to the value it is called on.
            // So, if `branch_offset[1]` holds a value of 5, this statement
            // will return 0b10000, assuming `i < num_branches`.
            return valid_mask.fo1() & (X^i).decode().concat();
        };
        banks_with_branches_onehot.fanout(hard<2>{});
        val<N> branches_mask = banks_with_branches_onehot.fold_or();

        // Which bank, if any, had a taken branch mapped to it?
        arr<val<N>, N> taken_onehot = [&](u64 i) {
            return banks_with_branches_onehot[i] & branch_exec_taken[i].replicate(hard<N>{}).concat();
        };
        arr<val<1>, N> taken = taken_onehot.fo1().fold_or().make_array(val<1>{});
        taken.fanout(hard<HIST_LEN+1>{});

        // Which bank, if any, had a mispredicted branch mapped to it?
        arr<val<N>, N> mispredicted_onehot = [&](u64 i) {
            // `valid_mask` will have all bits set if this was a executed in
            // this prediction block
            val<N> valid_mask = val<1>{i == (num_condbr - 1)}.replicate(hard<N>{}).concat();
            val<N> mispredicted_mask = mispredict.replicate(hard<N>{}).concat();
            return valid_mask.fo1() & mispredicted_mask.fo1() & (X^i).decode().concat();
        };
        val<N> mispredicted_mask = mispredicted_onehot.fo1().fold_or();

        // Which banks are performing an update?
        val<N> update_mask = mispredicted_mask.fo1() | (below_threshold.fo1().concat() & branches_mask.fo1());
        update_mask.fanout(hard<3>{});

        // Is *any* bank performing an update?
        val<1> performing_update = update_mask.make_array(val<1>{}).fold_or();

        // If we are doing an update, inform the simulator we need an extra
        // cycle to write the array (note this must be called *before* the
        // array write below, or it will fail at runtime with a message like
        // "single RAM access per cycle")
        need_extra_cycle(performing_update.fo1());

        global_history.fanout(hard<N+1>{});
        execute_if(update_mask, [&](u64 i){
            arr<val<WEIGHT_BITS, i64>, HIST_LEN+1> updated_vector = [&](u64 j) {
                if (j == 0)
                    return update_ctr(site[i].branch_weights[j], taken[i]);
                else
                    return update_ctr(site[i].branch_weights[j], taken[i] ^ global_history[j-1]);
            };
            site[i].weights.write(index, updated_vector.fo1());
        });

        // Update the global history upon a taken branch
        execute_if(block_end_info.is_taken, [&]{
            global_history = global_history.shift_left(val<2>{block_end_info.next_pc.fo1()>>2});
        });

        // Reset counters of instructions and conditional branches for the next
        // prediction block
        num_condbr = 0;
        bb_inst = 0;
    }
};
