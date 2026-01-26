#include "../harcom.hpp"
#include "common.hpp"

using namespace hcm;


template<u64 LOGB=16, u64 LOGN=2>
struct bimodalN : predictor {
    // bimodal with 2^LOGB entries, single prediction level (no overriding)
    // provides 2^LOGN predictions per cycle
    // 1st pred is for first cond branch,
    // 2nd pred is for next cond branch (if first not taken), etc.
    static constexpr u64 N = 1<<LOGN; // number of banks
    static_assert(LOGB>LOGN);
    static constexpr u64 index_bits = LOGB-LOGN;
    static constexpr u64 BBMAX = 64; // maximum basic block size (instructions)

    // try to use all banks evenly
    // define X as the LOGN rightmost bits of the block address (inst_pc>>2)
    // 1st pred in bank X, 2nd pred in bank X^1, 3rd pred in bank X^2,...
    reg<LOGN> X;
    reg<index_bits> index; // same index for all banks
    arr<reg<1>,N> pred; // read prediction bits, ordered
    ram<val<1>,(1<<index_bits)> ctr_hi[N]; // prediction bit

    u64 num_branch = 0; // number of conditional branches in block so far
    u64 bb_inst = 0; // instructions in current basic block
    arr<reg<1>,N> branch_dir; // actual branch direction

    zone UPDATE_ONLY;
    ram<val<1>,(1<<index_bits)> ctr_lo[N]; // hysteresis bit (1=weak, 0=strong)

    val<1> predict1([[maybe_unused]] val<64> inst_pc)
    {
        // new block
        assert(num_branch==0);
        assert(bb_inst==0);
        inst_pc.fanout(hard<2>{});
        index = inst_pc >> (LOGN+2);
        index.fanout(hard<N>{});
        arr<val<1>,N> unordered_pred = [&](u64 i){
            return ctr_hi[i].read(index);
        };
        unordered_pred.fanout(hard<N>{});
        X = inst_pc >> 2;
        X.fanout(hard<N>{});
        for (u64 i=0; i<N; i++) {
            pred[i] = unordered_pred.select(X^i);
        }
        pred.fanout(hard<BBMAX*4>{});
        bb_inst++;
        reuse_prediction(bb_inst < BBMAX);
        return pred[num_branch];
    };

    val<1> reuse_predict1([[maybe_unused]] val<64> inst_pc)
    {
        bb_inst++;
        reuse_prediction(bb_inst < BBMAX);
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
        bb_inst = 0;
        if (num_branch==N) {
            // this was the last branch for which we have a prediction: end the block
            reuse_prediction(0);
        }
    }

    void update_cycle([[maybe_unused]] val<1> mispredict, [[maybe_unused]] val<64> next_pc)
    {
        if (num_branch==0) {
            bb_inst = 0;
            return;
        }
        static_assert(N<=64);
        X.fanout(hard<N*2>{});
        index.fanout(hard<N*3>{});
        branch_dir.fanout(hard<N>{});
        mispredict.fanout(hard<2>{});
        // determine banks that are accessed
        arr<val<N>,N> bank = [&](u64 i) -> val<N> {
            return (X^i).decode().concat();
        };
        bank.fanout(hard<2>{});
        arr<val<1>,N> access = [&](u64 i) -> val<1> {
            return (bank[i]>>num_branch) == hard<0>{};
        };
        // align branch directions with corresponding banks
        arr<val<1>,N> true_dir = [&](u64 i) -> val<1> {
            return branch_dir.select(X^i);
        };
        // determine bank corresponding to mispredicted branch
        val<N> misp_bank = execute_if(mispredict, [&]() -> val<N> {
            return bank[num_branch-1];
        });
        misp_bank.fanout(hard<2>{});
        arr<val<1>,N> mispredicted = misp_bank.make_array(val<1>{});
        // read hysteresis bit if bank corresponds to mispredicted branch
        arr<val<1>,N> weak = execute_if(misp_bank, [&](u64 i){
            // return 1 iff mispredict and hysteresis is weak
            return ctr_lo[i].read(index);
        });
        // we need an extra cycle if there is a mispredict
        need_extra_cycle(mispredict);
        // update prediction if mispredict and the hysteresis bit is weak
        execute_if(weak.fo1().concat(), [&](u64 i){
            ctr_hi[i].write(index,true_dir[i].fo1());
        });
        // update hysteresis
        execute_if(access.fo1().concat(), [&](u64 i){
            ctr_lo[i].write(index,mispredicted[i].fo1());
        });
        num_branch = 0;
        bb_inst = 0;
    }
};
