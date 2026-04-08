#include "../cbp.hpp"
#include "../harcom.hpp"
#include "common.hpp"

using namespace hcm;

//
// Two-level local branch predictor
//
// Architecture:
//   PHT (Pattern History Table): Indexed by branch PC, stores a per-branch
//        local history shift register of HIST_LEN bits.
//   BHT (Branch History Table):  Indexed by the local history from PHT, stores
//        2-bit saturating counters (00=SN, 01=WN, 10=WT, 11=ST).
//
// On each prediction:
//   P1 (predict1) reads the PHT to fetch the local history, and returns the
//        history MSB as a fast bimodal-style guess.
//   P2 (predict2) uses that history to index the BHT and returns the accurate
//        two-level prediction (counter MSB).
//
// -----------------------------------------------------------------------
// Configurable template parameters — edit these to explore trade-offs:
//
//   LOG_PHT   log2(# PHT entries). Default 10 → 1024 history registers.
//             Fewer entries → more PHT aliasing: distinct branches collide
//             and corrupt each other's history registers.
//
//   HIST_LEN  History length in bits per PHT entry. Default 8.
//             Longer → predictor can distinguish more repeating patterns,
//             but the BHT must be 2^HIST_LEN large to avoid BHT aliasing.
//
//   LOG_BHT   log2(# BHT entries). Default == HIST_LEN (zero BHT aliasing).
//             Setting LOG_BHT < HIST_LEN causes BHT aliasing: different
//             history patterns share a 2-bit counter, hurting accuracy.
//             Setting LOG_BHT > HIST_LEN wastes BHT capacity (some entries
//             are unreachable since there are only 2^HIST_LEN distinct
//             history values).
//
//   LOGLB     log2(cache-line size in bytes). Default 6 = 64-byte lines.
//             Usually leave this unchanged.
//
// Example configurations to compare:
//   local2<>                        // baseline: LOG_PHT=10, HIST_LEN=8, LOG_BHT=8
//   local2<6,8,8,8>                 // fewer PHT entries → more PHT aliasing
//   local2<6,10,8,4>                // small BHT → heavy BHT aliasing
//   local2<6,10,4,4>                // short history → less pattern depth
//   local2<6,12,12,12>              // large tables, long history
// -----------------------------------------------------------------------
template<u64 LOGLB=6, u64 LOG_PHT=10, u64 HIST_LEN=8, u64 LOG_BHT=HIST_LEN>
struct local2 : predictor {
    static_assert(HIST_LEN >= 1 && HIST_LEN <= 32, "HIST_LEN out of range");
    static_assert(LOG_PHT  >= 1 && LOG_PHT  <= 20, "LOG_PHT out of range");
    static_assert(LOG_BHT  >= 1 && LOG_BHT  <= 20, "LOG_BHT out of range");

    // PHT: 2^LOG_PHT history registers, each HIST_LEN bits wide.
    // Indexed by the lower LOG_PHT bits of (branch_pc >> 2).
    ram<val<HIST_LEN>, (1<<LOG_PHT)> pht;

    // BHT: 2^LOG_BHT two-bit saturating counters.
    // Indexed by the lower LOG_BHT bits of the local history from PHT.
    // (When LOG_BHT == HIST_LEN all history bits are used as the index;
    //  when LOG_BHT < HIST_LEN the upper bits are folded in via XOR.)
    ram<val<2>, (1<<LOG_BHT)> bht;

    // ---- Registers holding prediction state across P1 → P2 ----
    reg<LOG_PHT>  p_pht_idx;   // PHT index computed from inst_pc
    reg<HIST_LEN> p_history;   // local history read from PHT
    reg<LOG_BHT>  p_bht_idx;   // BHT index (folded history)
    reg<2>        p_bht_ctr;   // 2-bit counter read from BHT

    // ---- Registers holding state latched during update_condbr ----
    reg<LOG_PHT>  u_pht_idx;   // PHT index for the branch being updated
    reg<HIST_LEN> u_history;   // history at prediction time
    reg<LOG_BHT>  u_bht_idx;   // BHT index for the branch being updated
    reg<2>        u_bht_ctr;   // 2-bit counter value at prediction time
    reg<1>        u_taken;     // actual branch direction

    // Non-hardware: tracks whether a conditional branch occurred this block
    u64 num_branches = 0;

    // Compute the PHT index from a 64-bit PC
    val<LOG_PHT> pht_index_of(val<64> pc) {
        return val<LOG_PHT>{pc >> 2};
    }

    // Fold HIST_LEN-bit history down to LOG_BHT bits via XOR chunking.
    // When LOG_BHT >= HIST_LEN this is a simple truncation (zero-extension).
    val<LOG_BHT> bht_index_of(val<HIST_LEN> history) {
        return history.make_array(val<LOG_BHT>{}).fold_xor();
    }

    // ---------------------------------------------------------------- predict

    val<1> predict1(val<64> inst_pc) {
        // Stage 1: read the PHT to obtain this branch's local history.
        p_pht_idx = pht_index_of(inst_pc);
        p_history = pht.read(p_pht_idx);
        // P1 fast prediction: use the MSB of the local history.
        // This is equivalent to a 1-bit bimodal keyed on the PC.
        return p_history >> (HIST_LEN - 1);
    }

    val<1> reuse_predict1([[maybe_unused]] val<64> inst_pc) {
        // Re-use the history read from the most recent predict1.
        return p_history >> (HIST_LEN - 1);
    }

    val<1> predict2([[maybe_unused]] val<64> inst_pc) {
        // Stage 2: use the local history to index the BHT.
        p_bht_idx = bht_index_of(p_history);
        p_bht_ctr = bht.read(p_bht_idx);
        // Do not reuse: each branch in the block may have a different history.
        reuse_prediction(hard<0>{});
        return p_bht_ctr >> 1;   // MSB of 2-bit counter
    }

    val<1> reuse_predict2([[maybe_unused]] val<64> inst_pc) {
        return p_bht_ctr >> 1;
    }

    // ---------------------------------------------------------------- update

    void update_condbr(val<64> branch_pc, val<1> taken,
                       [[maybe_unused]] val<64> next_pc) {
        // Latch the prediction context so update_cycle can modify both tables.
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

        // We always write to the PHT (history shift), so we always need an
        // extra cycle for the update.
        need_extra_cycle(hard<1>{});

        // --- BHT update: move the 2-bit saturating counter toward 'taken' ---
        val<2> new_ctr = update_ctr(u_bht_ctr, u_taken);
        val<1> bht_changed = val<1>{new_ctr != u_bht_ctr};
        execute_if(bht_changed, [&](){
            bht.write(u_bht_idx, new_ctr);
        });

        // --- PHT update: shift the local history and insert the actual direction ---
        // new_history = { old_history[HIST_LEN-2 : 0], taken }
        val<HIST_LEN> new_history =
            val<HIST_LEN>{(val<HIST_LEN>{u_history} << 1) | val<HIST_LEN>{u_taken}};
        pht.write(u_pht_idx, new_history);
    }
};
