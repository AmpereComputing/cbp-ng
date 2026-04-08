#include "../cbp.hpp"
#include "../harcom.hpp"
#include "common.hpp"

using namespace hcm;


// Two-level local branch predictor
// PHT: indexed by branch PC, stores per-branch local history (HIST_LEN bits)
// BHT: indexed by local history, stores 2-bit saturating counters
// P1 reads PHT and returns history MSB; P2 reads BHT[history] for the accurate prediction.
//
// LOG_PHT  : log2(# PHT entries). Fewer → more PHT aliasing across branches.
// HIST_LEN : history bits per branch. More → finer patterns; BHT needs 2^HIST_LEN entries for no aliasing.
// LOG_BHT  : log2(# BHT entries). Equal to HIST_LEN = no BHT aliasing; less → BHT aliasing.
// LOGLB    : log2(cache-line bytes). Default 6 = 64-byte lines.
template<u64 LOGLB=6, u64 LOG_PHT=10, u64 HIST_LEN=8, u64 LOG_BHT=HIST_LEN>
struct local2_tpl : predictor {
    static_assert(HIST_LEN >= 1 && HIST_LEN <= 32);
    static_assert(LOG_PHT  >= 1 && LOG_PHT  <= 20);
    static_assert(LOG_BHT  >= 1 && LOG_BHT  <= 20);

    ram<val<HIST_LEN>, (1<<LOG_PHT)> pht;
    ram<val<2>, (1<<LOG_BHT)> bht;

    reg<LOG_PHT>  p_pht_idx;
    reg<HIST_LEN> p_history;
    reg<LOG_BHT>  p_bht_idx;
    reg<2>        p_bht_ctr;

    reg<LOG_PHT>  u_pht_idx;
    reg<HIST_LEN> u_history;
    reg<LOG_BHT>  u_bht_idx;
    reg<2>        u_bht_ctr;
    reg<1>        u_taken;

    u64 num_branches = 0;

    val<LOG_PHT> pht_index_of(val<64> pc) {
        return val<LOG_PHT>{pc >> 2};
    }

    val<LOG_BHT> bht_index_of(val<HIST_LEN> history) {
        return history.make_array(val<LOG_BHT>{}).fold_xor();
    }

    val<1> predict1(val<64> inst_pc) {
        p_pht_idx = pht_index_of(inst_pc);
        p_history = pht.read(p_pht_idx);
        return p_history >> (HIST_LEN - 1);
    }

    val<1> reuse_predict1([[maybe_unused]] val<64> inst_pc) {
        return p_history >> (HIST_LEN - 1);
    }

    val<1> predict2([[maybe_unused]] val<64> inst_pc) {
        p_bht_idx = bht_index_of(p_history);
        p_bht_ctr = bht.read(p_bht_idx);
        reuse_prediction(hard<0>{});
        return p_bht_ctr >> 1;
    }

    val<1> reuse_predict2([[maybe_unused]] val<64> inst_pc) {
        return p_bht_ctr >> 1;
    }

    void update_condbr(val<64> branch_pc, val<1> taken,
                       [[maybe_unused]] val<64> next_pc) {
        u_pht_idx = pht_index_of(branch_pc);
        u_history = p_history;
        u_bht_idx = p_bht_idx;
        u_bht_ctr = p_bht_ctr;
        u_taken   = taken;
        num_branches++;
    }

    void update_cycle([[maybe_unused]] instruction_info &block_end_info) {
        if (num_branches == 0) return;
        num_branches = 0;

        need_extra_cycle(hard<1>{});

        val<2> new_ctr = update_ctr(u_bht_ctr, u_taken);
        val<1> bht_changed = val<1>{new_ctr != u_bht_ctr};
        execute_if(bht_changed, [&](){
            bht.write(u_bht_idx, new_ctr);
        });

        val<HIST_LEN> new_history =
            val<HIST_LEN>{(val<HIST_LEN>{u_history} << 1) | val<HIST_LEN>{u_taken}};
        pht.write(u_pht_idx, new_history);
    }
};

// Default instantiation — usable as -DPREDICTOR=local2 (no angle brackets needed)
struct local2 : local2_tpl<> {};
