#include "../cbp.hpp"
#include "../harcom.hpp"
#include "common.hpp"

#include <array>

using namespace hcm;


template<
    u64 LOGLB    = 6,   // 64B fetch block
    u64 NTABLES  = 8,   // number of tables
    u64 MAXHIST  = 100, // maximum global history length
    u64 MINHIST  = 2,   // minimum global history length
    u64 WBITS    = 4,   // signed 4-bit weight (-8 to 7)
    u64 LOGTABLE = 13,  // 32KB hashed perceptron for P2
    u64 LOGP1    = 14,  // 4KB gshare for P1
    u64 GHIST1   = 6    // P1 gshare history length
>
struct hashed_perceptron : predictor {
    static_assert(LOGLB >= 2);
    static constexpr u64 LOGLINEINST = LOGLB - 2; // 4B instruction

    static constexpr u64 LINEINST = 1ull << LOGLINEINST;
    static constexpr u64 NUMHIST = NTABLES - 1; // number of tables using history

    static constexpr u64 YBITS  = WBITS + std::bit_width(NTABLES-1);
    static constexpr u64 TCBITS = 4; // corresponds to SPEED = 16
    static constexpr u64 THETABITS = YBITS + TCBITS;
    static constexpr u64 PATHBITS = 6;

    static_assert(LOGP1 >= LOGLINEINST);
    static constexpr u64 index1_bits = LOGP1 - LOGLINEINST;
    static_assert(LOGTABLE >= LOGLINEINST);
    static constexpr u64 index2_bits = LOGTABLE - LOGLINEINST;

    geometric_folds<NUMHIST, MINHIST, MAXHIST, index2_bits> gfolds;
    reg<1> true_block = 1;

    // ---- for P1 (gshare) ----
    reg<GHIST1> global_history1;
    reg<index1_bits> index1;
    arr<reg<1>, LINEINST> readp1;
    reg<LINEINST> p1;

    // ---- for P2 (hashed perceptron) ----
    // based on ChampSim's implementation
    // https://github.com/ChampSim/ChampSim/blob/29b7b368f761fce1309103c17280c847ada44c1e/branch/hashed_perceptron.bpred
    arr<reg<index2_bits>, NTABLES> index2;
    arr<reg<WBITS, i64>, NTABLES> readw[LINEINST];
    arr<reg<YBITS, i64>, LINEINST> yout;
    reg<LINEINST> p2;
    // dynamic update threshold
    // packed counter representing (theta << TCBITS) + tc
    reg<THETABITS, i64> theta_and_tc = 10 << TCBITS;

    // ---- simulation artifacts ----
    u64 num_branch = 0;
    u64 block_size = 0;
    arr<reg<LOGLINEINST>, LINEINST> branch_offset;
    arr<reg<1>, LINEINST> branch_dir;
    reg<LINEINST> block_entry;

    // ---- RAMs ----
    // P2 weight tables
    ram<val<WBITS, i64>, (1 << index2_bits)> wtable[NTABLES][LINEINST] {{"P2 weight"}};

    // P1 prediction bits
    ram<val<1>, (1 << index1_bits)> table1_pred[LINEINST] {"P1 pred"};
    // P1 hysteresis bits
    zone UPDATE_ONLY;
    ram<val<1>, (1 << index1_bits)> table1_hyst[LINEINST] {"P1 hyst"};

    void new_block(val<64> inst_pc)
    {
        val<LOGLINEINST> offset = inst_pc.fo1() >> 2;
        block_entry = offset.fo1().decode().concat();
        block_entry.fanout(hard<4*LINEINST>{});
        block_size = 1;
    }

    // P1 gshare
    val<1> predict1(val<64> inst_pc)
    {
        inst_pc.fanout(hard<2>{});
        new_block(inst_pc);

        // index1 = PC ^ ghist1
        val<std::max(index1_bits, GHIST1)> lineaddr = inst_pc >> LOGLB;
        global_history1.fanout(hard<2>{});
        if constexpr (GHIST1 <= index1_bits) {
            index1 = lineaddr.fo1() ^ (val<index1_bits>{global_history1} << (index1_bits - GHIST1));
        } else {
            index1 = global_history1.make_array(val<index1_bits>{}).append(lineaddr.fo1()).fold_xor();
        }
        index1.fanout(hard<LINEINST+1>{});

        // read prediction bits
        for (u64 offset=0; offset<LINEINST; offset++) {
            readp1[offset] = table1_pred[offset].read(index1);
        }
        readp1.fanout(hard<2>{});
        p1 = readp1.concat();
        p1.fanout(hard<LINEINST>{});

        // prediction for the first instruction
        return (block_entry & p1) != hard<0>{};
    }

    val<1> reuse_predict1([[maybe_unused]] val<64> inst_pc)
    {
        // prediction for subsequent instructions
        return ((block_entry << block_size) & p1) != hard<0>{};
    }

    // P2 hashed perceptron
    val<1> predict2(val<64> inst_pc)
    {
        val<index2_bits> lineaddr = inst_pc.fo1() >> LOGLB;
        lineaddr.fanout(hard<NTABLES>{});
        gfolds.fanout(hard<2>{});

        // index2 = PC ^ folded_history
        for (u64 i=0; i<NTABLES; i++) {
            if (i == 0) {
                index2[i] = lineaddr;
            } else {
                index2[i] = lineaddr ^ gfolds.template get<0>(i-1);
            }
        }
        index2.fanout(hard<2>{});

        // read weights, then summed up
        for (u64 i=0; i<NTABLES; i++) {
            auto dindex2 = index2[i].distribute(wtable[i]);
            for (u64 offset=0; offset<LINEINST; offset++) {
                readw[offset][i] = wtable[i][offset].read(dindex2[offset].fo1());
            }
        }
        for (u64 offset=0; offset<LINEINST; offset++) {
            readw[offset].fanout(hard<2>{});
            yout[offset] = readw[offset].fold_add();
        }
        yout.fanout(hard<2>{});

        // per-offset predictions
        arr<val<1>, LINEINST> p2bits = [&](u64 offset) {
            auto [sign_bit, rest] = split<1, YBITS-1>(yout[offset]);
            return sign_bit.fo1();
        };
        p2 = p2bits.fo1().concat();
        p2.fanout(hard<LINEINST>{});

        // prediction for the first instruction
        val<1> taken = (block_entry & p2) != hard<0>{};

        // determine block termination
        reuse_prediction(~val<1>{block_entry >> (LINEINST - 1)});
        return taken.fo1();
    }

    val<1> reuse_predict2([[maybe_unused]] val<64> inst_pc)
    {
        // prediction for subsequent instructions
        val<1> taken = ((block_entry << block_size) & p2) != hard<0>{};

        // determine block termination
        reuse_prediction(~val<1>{block_entry >> (LINEINST - 1 - block_size)});
        block_size++;
        return taken.fo1();
    }

    // update interface
    void update_condbr(val<64> branch_pc, val<1> taken, [[maybe_unused]] val<64> next_pc)
    {
        assert(num_branch < LINEINST);
        branch_offset[num_branch] = branch_pc.fo1() >> 2;
        branch_dir[num_branch] = taken.fo1();
        num_branch++;
    }

    void update_cycle(instruction_info &block_end_info)
    {
        val<1> &mispredict = block_end_info.is_mispredict;
        val<64> &next_pc = block_end_info.next_pc;

        // updates for all conditional branches in the predicted block
        if (num_branch == 0) {
            // no conditional branch in this block
            val<1> line_end = block_entry >> (LINEINST - block_size);
            // update global history if previous block ended on a mispredicted not-taken branch
            // (we are still in the same line, this is the last chunk)
            // or if the block ends before the line boundary (unconditional jump)
            true_block.fanout(hard<2>{});
            val<1> actual_block = ~(true_block & line_end.fo1());
            actual_block.fanout(hard<MAXHIST+NUMHIST+3>{});
            execute_if(actual_block, [&](){
                next_pc.fanout(hard<2>{});
                global_history1 = (global_history1 << 1) ^ val<GHIST1>{next_pc >> 2};
                gfolds.update(val<PATHBITS>{next_pc >> 2});
                true_block = 1;
            });
            return; // stop here
        }

        branch_dir.fanout(hard<2>{});
        branch_offset.fanout(hard<LINEINST>{});
        index1.fanout(hard<LINEINST*3>{});
        index2.fanout(hard<LINEINST>{});
        yout.fanout(hard<3>{});
        theta_and_tc.fanout(hard<2>{});
        p2.fanout(hard<2>{});

        // masks mapping executed branches to offset
        u64 update_valid = (u64(1) << num_branch) - 1;
        arr<val<LINEINST>, LINEINST> update_mask = [&](u64 offset){
            arr<val<1>, LINEINST> match_offset = [&](u64 i){ return branch_offset[i] == offset; };
            return match_offset.fo1().concat() & update_valid;
        };
        update_mask.fanout(hard<2>{});

        // Is there a branch instruction at the offset?
        arr<val<1>, LINEINST> is_branch = [&](u64 offset){
            return update_mask[offset] != hard<0>{};
        };
        is_branch.fanout(hard<2>{});
        val<LINEINST> branch_mask = is_branch.concat();
        branch_mask.fanout(hard<6>{});

        // is the outcome of the branch at the offset taken?
        val<LINEINST> actualdirs = branch_dir.concat();
        actualdirs.fanout(hard<LINEINST>{});
        arr<val<1>, LINEINST> branch_taken = [&](u64 offset){
            return (actualdirs & update_mask[offset]) != hard<0>{};
        };
        branch_taken.fanout(hard<NTABLES+1>{});

        // is the P2 (=final) prediction correct?
        auto p2_split = p2.make_array(val<1>{});
        p2_split.fanout(hard<3>{});
        arr<val<1>, LINEINST> correct = [&](u64 offset){
            return p2_split[offset] == branch_taken[offset];
        };
        val<LINEINST> correct_mask = correct.fo1().concat();
        correct_mask.fanout(hard<3>{});

        // is the prediction weak?
        auto [theta, tc] = split<YBITS, TCBITS>(theta_and_tc);
        theta.fanout(hard<LINEINST>{});
        arr<val<1>, LINEINST> weak = [&](u64 offset){
            auto absy = select(yout[offset] < hard<0>{}, -yout[offset], yout[offset]);
            return absy.fo1() < theta;
        };
        val<LINEINST> weak_mask = weak.fo1().concat();
        weak_mask.fanout(hard<2>{});

        // perceptrons are updated if the prediction is wrong or weak
        val<LINEINST> train_mask = branch_mask & (~correct_mask | weak_mask);
        train_mask.fanout(hard<NTABLES+2>{});

        // ---- P1 (gshare) update ----
        // did P1 and P2 disagree?
        val<LINEINST> disagree_mask = (p1 ^ p2) & branch_mask;
        disagree_mask.fanout(hard<2>{});
        arr<val<1>,LINEINST> disagree = disagree_mask.make_array(val<1>{});
        disagree.fanout(hard<2>{});

        // read P1 hysteresis if P1 and P2 disagree
        arr<val<1>, LINEINST> p1_weak = [&](u64 offset) -> val<1> {
            return execute_if(disagree[offset], [&](){
                return ~table1_hyst[offset].read(index1); // hyst=0 means weak
            });
        };

        val<1> extra_cycle = (train_mask != hard<0>{}) | (disagree_mask != hard<0>{});
        need_extra_cycle(extra_cycle.fo1());

        // update P1 prediction if P1 and P2 disagree and hysteresis bit is weak
        execute_if(p1_weak.fo1().concat(), [&](u64 offset){
            table1_pred[offset].write(index1, p2_split[offset]);
        });
        // update P1 hysteresis for executed branches
        execute_if(branch_mask, [&](u64 offset){
            table1_hyst[offset].write(index1, ~disagree[offset]);
        });

        // ---- P2 (hashed perceptron) update ----
        // weight update
        execute_if(train_mask, [&](u64 offset){
            for (u64 i=0; i<NTABLES; i++) {
                wtable[i][offset].write(index2[i], update_ctr(readw[offset][i], ~branch_taken[offset]));
            }
        });

        // update packed counter
        // decrement on weak & correct
        // increment on mispredict
        theta_and_tc = theta_and_tc - (branch_mask & weak_mask & correct_mask).ones() + (branch_mask & ~correct_mask).ones();

        // ---- update global history ----
        val<1> line_end = block_entry >> (LINEINST - block_size);
        true_block = ~mispredict.fo1() | branch_dir[num_branch-1] | line_end.fo1();
        true_block.fanout(hard<MAXHIST+NUMHIST+2>{});
        execute_if(true_block, [&](){
            next_pc.fanout(hard<2>{});
            global_history1 = (global_history1 << 1) ^ val<GHIST1>{next_pc >> 2};
            gfolds.update(val<PATHBITS>{next_pc >> 2});
        });

        num_branch = 0;
    }
};
