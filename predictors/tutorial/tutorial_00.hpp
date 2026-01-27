#include "../../cbp.hpp"
#include "../../harcom.hpp"

using namespace hcm;

struct tutorial_00 : predictor {
    /*
     * Always predict not-taken (i.e. 0) for both first- and second-level
     * predictors. Predict a single instruction per cycle. 
     */

    val<1> predict1([[maybe_unused]] val<64> inst_pc)
    {
        // Always return 0 representing not-taken (`hard` is a hardware
        // parameter known at design-time)
        return hard<0>{};
    };

    val<1> predict2([[maybe_unused]] val<64> inst_pc)
    {
        // Do the same thing for the second-level predictor
        return hard<0>{};
    }

    void update_condbr([[maybe_unused]] val<64> branch_pc, [[maybe_unused]] val<1> taken, [[maybe_unused]] val<64> next_pc)
    {
    }

    void update_cycle([[maybe_unused]] instruction_info &block_end_info)
    {
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
