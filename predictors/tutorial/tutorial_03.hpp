#include "../../cbp.hpp"
#include "../../harcom.hpp"

using namespace hcm;

template<u64 LINE_BITS = 4>
struct tutorial_03 : predictor {
    /*
     * Predict up to 16 instructions per cycle using an SRAM array of 16
     * two-bit counters.
     */

    static constexpr u64 LINE_INSTRUCTIONS = 1 << LINE_BITS;

    ram<arr<val<2>,LINE_INSTRUCTIONS>, 64> counters;

    // Record values from prediction-time to be used at update-time
    reg<6> index;
    arr<reg<2>, LINE_INSTRUCTIONS> counter;
    arr<reg<1>, LINE_INSTRUCTIONS> saturated;
    arr<reg<1>, LINE_INSTRUCTIONS> pred_taken;

    // Record values from per-branch updates to be used in per-cacheline
    // update logic
    u64 num_branches = 0;
    arr<reg<LINE_BITS>, LINE_INSTRUCTIONS> branch_offset;
    arr<reg<1>, LINE_INSTRUCTIONS> branch_taken;

    val<1> predict1([[maybe_unused]] val<64> inst_pc)
    {
        // Reset the count of the number of branches executed from this
        // prediction block
        num_branches = 0;

        // Hash the instruction PC to a 6-bit index by first chunking it into
        // an array of 6-bit values, then folding that array onto itself using
        // XOR.
        index = (inst_pc >> 6).make_array(val<6>{}).fold_xor();

        // Index into the array of 2-bit counters, saving the counter value to
        // a register
        counter = counters.read(index);

        // Save information about this prediction (its direction and whether
        // the counter was saturated) to be used for other predictions this
        // cycle and update-time.
        for (u64 i=0; i<LINE_INSTRUCTIONS; i++) {
            saturated[i] = (counter[i] == hard<0>{}) | (counter[i] == hard<3>{});
            pred_taken[i] = counter[i] >> (counter[i].size - 1);
        }

        return predict(inst_pc);
    };

    val<1> predict(val<64> inst_pc)
    {
        // Use the top bit of the counter to predict the branch's direction
        val<LINE_BITS> offset = inst_pc >> 2;
        reuse_prediction(offset != hard<LINE_INSTRUCTIONS-1>{});
        return pred_taken.select(offset);
    };

    val<1> reuse_predict1([[maybe_unused]] val<64> inst_pc)
    {
        return predict(inst_pc);
    };

    val<1> reuse_predict2([[maybe_unused]] val<64> inst_pc)
    {
        return predict(inst_pc);
    }

    val<1> predict2([[maybe_unused]] val<64> inst_pc)
    {
        return predict(inst_pc);
    }

    // Note: common.hpp contains a more generic version of a saturating counter
    // update, this is reproduced here for learning purposes.
    inline val<2> update_counter(val<2> counter, val<1> taken) {
        val<2> increased = select(counter == 3, counter, val<2>{counter + 1});
        val<2> decreased = select(counter == 0, counter, val<2>{counter - 1});
        return select(taken, increased, decreased);
    }

    void update_condbr([[maybe_unused]] val<64> branch_pc, [[maybe_unused]] val<1> taken, [[maybe_unused]] val<64> next_pc)
    {
        // For each conditional branch executed, record its offset and whether
        // it was taken. This is used by `update_cycle` to perform updates at a
        // cacheline granularity. Note that `num_branches` is a regular C++
        // integer. Although most values are represented by HARCOM types, it is
        // allowed to count branches and instructions using C++ integers and to
        // use those counts as you see demonstrated here.
        branch_offset[num_branches] = branch_pc >> 2;
        branch_taken[num_branches] = taken;
        num_branches++;
    }

    void update_cycle([[maybe_unused]] instruction_info &block_end_info)
    {
        // In order to perform updates for the current cacheline, we want a
        // mask with one bit corresponding to one instruction offset in the
        // line, but to start we have an array indexed by branch number.
        arr<val<LINE_INSTRUCTIONS>, LINE_INSTRUCTIONS> branch_onehot = [&](u64 i) {
            // `valid_mask` will have all bits set if this was a executed in
            // this prediction block
            val<LINE_INSTRUCTIONS> valid_mask = val<1>{i < num_branches}.replicate(hard<LINE_INSTRUCTIONS>{}).concat();
            // `decode` sets a bit corresponding to the value it is called on.
            // So, if `branch_offset[1]` holds a value of 5, this statement
            // will return 0b10000, assuming `i < num_branches`.
            return valid_mask & branch_offset[i].decode().concat();
        };
        // Fold that array of 16-bit vals into a single 16-bit val with
        // bitwise-OR
        val<LINE_INSTRUCTIONS> branch_mask = branch_onehot.fold_or();

        // Similar to `branch_mask` above, but this time we only want the bits
        // in the mask set if this were a *taken* branch
        arr<val<LINE_INSTRUCTIONS>, LINE_INSTRUCTIONS> taken_onehot = [&](u64 i) {
            return branch_onehot[i] & branch_taken[i].replicate(hard<LINE_INSTRUCTIONS>{}).concat();
        };
        val<LINE_INSTRUCTIONS> taken_mask = taken_onehot.fold_or();

        // Determine which offsets we want to update. We update any the counter
        // of any offset which was a branch and that branch's counter was
        // either incorrect or already-saturated.
        val<LINE_INSTRUCTIONS> incorrect = taken_mask ^ pred_taken.concat();
        val<LINE_INSTRUCTIONS> update_mask = branch_mask & (~saturated.concat() | incorrect);

        // Determine whether to perform an update - when the updated counter is
        // different than the read counter
        val<1> performing_update = (update_mask != hard<0>{});

        arr<val<2>, LINE_INSTRUCTIONS> new_counters = [&](u64 i){
            return select(val<1>{update_mask>>i},
                          update_counter(counter[i], val<1>{taken_mask>>i}),
                          counter[i]);
        };

        // If we are doing an update, inform the simulator we need an extra
        // cycle to write the array (note this must be called *before* the
        // array write below, or it will fail at runtime with a message like
        // "single RAM access per cycle")
        need_extra_cycle(performing_update);

        // Finally, write back to the array (only if needed)
        execute_if(performing_update, [&](){
            counters.write(index, new_counters);
        });
    }
};
