#include "../cbp.hpp"
#include "../harcom.hpp"

using namespace hcm;

struct never_taken : predictor {
    val<1> predict1([[maybe_unused]] val<64> inst_pc)
    {
        reuse_prediction(hard<1>{});
        return hard<0>{};
    };

    val<1> reuse_predict1([[maybe_unused]] val<64> inst_pc)
    {
        reuse_prediction(hard<1>{});
        return hard<0>{};
    };

    val<1> predict2([[maybe_unused]] val<64> inst_pc)
    {
        reuse_prediction(hard<1>{});
        return hard<0>{};
    }

    val<1> reuse_predict2([[maybe_unused]] val<64> inst_pc)
    {
        reuse_prediction(hard<1>{});
        return hard<0>{};
    }

    void update_condbr([[maybe_unused]] val<64> branch_pc, [[maybe_unused]] val<1> taken, [[maybe_unused]] val<64> next_pc) { }

    void update_cycle([[maybe_unused]] instruction_info &block_end_info) { }
};
