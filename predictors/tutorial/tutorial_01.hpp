#include "../../cbp.hpp"
#include "../../harcom.hpp"

using namespace hcm;

struct tutorial_01 : predictor {
    /*
     * Remember the PC of the last taken branch. Predict any branches matching
     * that PC as taken.
     */

    // Store the last taken branch
    reg<64> last_taken;

    // Save the branch PC and direction for use in `update_cycle()`
    reg<64> this_branch_pc;
    reg<1> this_branch_taken;

    val<1> predict1([[maybe_unused]] val<64> inst_pc)
    {
        // Predict taken if the current instruction's address matches that of
        // the last taken branch saved in `last_taken` register
        return last_taken == inst_pc;
    };

    val<1> predict2([[maybe_unused]] val<64> inst_pc)
    {
        // Re-use the same prediction for the second-level predictor
        return last_taken == inst_pc;
    }

    void update_condbr([[maybe_unused]] val<64> branch_pc, [[maybe_unused]] val<1> taken, [[maybe_unused]] val<64> next_pc)
    {
        // Temporarily save the last-seen branch address and direction to
        // registers. They will be used in `update_cycle` for the actual
        // update.
        this_branch_pc = branch_pc;
        this_branch_taken = taken;
    }

    void update_cycle([[maybe_unused]] instruction_info &block_end_info)
    {
        // Update the `last_taken` register if this branch was mispredicted. If
        // the mispredicted branch was executed taken, save its PC to
        // `last_taken`. If the mispredicted branch was executed not-taken, set
        // the "last_taken" register to 0 instead.
        val<64> branch_pc_to_update = select(this_branch_taken, this_branch_pc, val<64>{0});
        val<64> updated_last_taken = select(block_end_info.is_mispredict, branch_pc_to_update, last_taken);
        last_taken = updated_last_taken;
    }

    // reuse_predict1 and reuse_predict2 will never be called because this
    // predictor never calls reuse_prediction()
    val<1> reuse_predict1([[maybe_unused]] val<64> inst_pc)
    {
        return hard<0>{};
    };
    val<1> reuse_predict2([[maybe_unused]] val<64> inst_pc)
    {
        return hard<0>{};
    }
};
