#include "../harcom.hpp"
#include "common.hpp"

using namespace hcm;


template<u64 LOGLB=6, u64 LOGP2=17, u64 GHIST2=12, u64 LOGP1=14>
struct gshare : predictor {
    // P1: gshare with 2^LOGP1 entries
    // P2: gshare with 2^LOGP2 entries; global history of GHIST2 bits
    // provides 2^(LOGLB-2) predictions per cycle
    static_assert(LOGLB>2);
    static constexpr u64 LOGLINEINST = LOGLB-2;
    static constexpr u64 LINEINST = 1<<LOGLINEINST;
    static_assert(LOGP1 > LOGLINEINST);
    static_assert(LOGP2 > LOGLINEINST);
    static constexpr u64 index1_bits = LOGP1-LOGLINEINST;
    static constexpr u64 index2_bits = LOGP2-LOGLINEINST;

    reg<GHIST2> global_history;
    reg<1> true_block = 1;
    reg<index1_bits> index1;
    arr<reg<1>,LINEINST> ctr1_hi; // prediction bits read from P1 table
    reg<LINEINST> p1; // predictions
    reg<index2_bits> index2;
    arr<reg<1>,LINEINST> ctr2_hi; // prediction bits read from P2 table
    reg<LINEINST> p2; // predictions

    // simulation artifacts, hardware cost may not be real
    u64 num_branch = 0;
    u64 block_size = 0;
    arr<reg<LOGLINEINST>,LINEINST> branch_offset;
    arr<reg<1>,LINEINST> branch_dir;
    reg<LINEINST> block_entry; // one-hot vector

    ram<val<1>,(1<<index1_bits)> table1_hi[LINEINST]; // P1 prediction bits
    ram<val<1>,(1<<index2_bits)> table2_hi[LINEINST]; // P2 prediction bits

    zone UPDATE_ONLY;
    // hysteresis bits (0=weak, 1=strong)
    ram<val<1>,(1<<index1_bits)> table1_lo[LINEINST];
    ram<val<1>,(1<<index2_bits)> table2_lo[LINEINST];

    void new_block(val<64> inst_pc)
    {
        val<LOGLINEINST> offset = inst_pc.fo1() >> 2;
        block_entry = offset.fo1().decode().concat();
        block_entry.fanout(hard<LINEINST*2>{});
        block_size = 1;
    }

    val<1> predict1([[maybe_unused]] val<64> inst_pc)
    {
        // P1 does not predict the branch, it predicts P2
        // so we index P1 like P2, to the extent possible
        inst_pc.fanout(hard<2>{});
        new_block(inst_pc);
        val<std::max(index2_bits,GHIST2)> lineaddr = inst_pc >> LOGLB;
        lineaddr.fanout(hard<2>{});
        if constexpr (GHIST2 <= index2_bits) {
            index2 = lineaddr ^ (val<index2_bits>{global_history}<<(index2_bits-GHIST2));
        } else {
            index2 = global_history.make_array(val<index2_bits>{}).append(lineaddr).fold_xor();
        }
        index2.fanout(hard<LINEINST+1>{});
        index1 = index2;
        index1.fanout(hard<LINEINST>{});
        for (u64 i=0; i<LINEINST; i++) {
            ctr1_hi[i] = table1_hi[i].read(index1);
        }
        ctr1_hi.fanout(hard<2>{});
        p1 = ctr1_hi.concat();
        p1.fanout(hard<LINEINST>{});
        return (block_entry & p1) != hard<0>{};
    };

    val<1> reuse_predict1([[maybe_unused]] val<64> inst_pc)
    {
        return ((block_entry<<block_size) & p1) != hard<0>{};
    };

    val<1> predict2([[maybe_unused]] val<64> inst_pc)
    {
        for (u64 i=0; i<LINEINST; i++) {
            ctr2_hi[i] = table2_hi[i].read(index2);
        }
        ctr2_hi.fanout(hard<2>{});
        p2 = ctr2_hi.concat();
        p2.fanout(hard<LINEINST>{});
        val<1> taken = (block_entry & p2) != hard<0>{};
        reuse_prediction(~val<1>{block_entry>>(LINEINST-1)});
        return taken.fo1();
    }

    val<1> reuse_predict2([[maybe_unused]] val<64> inst_pc)
    {
        val<1> taken = ((block_entry<<block_size) & p2) != hard<0>{};
        reuse_prediction(~val<1>{block_entry>>(LINEINST-1-block_size)});
        block_size++;
        return taken.fo1();
    }

    void update_condbr(val<64> branch_pc, val<1> taken, [[maybe_unused]] val<64> next_pc)
    {
        assert(num_branch < LINEINST);
        branch_offset[num_branch] = branch_pc.fo1() >> 2;
        branch_dir[num_branch] = taken.fo1();
        num_branch++;
    }

    void update_cycle(val<1> mispredict, val<64> next_pc)
    {
        // updates for all conditional branches in the predicted block
        if (num_branch == 0) {
            // no conditional branch in this block
            execute_if(~true_block, [&](){
                // previous block ended on a mispredicted not-taken branch
                // we are still in the same line, this is the last chunk
                global_history = (global_history << 1) ^ val<GHIST2>{next_pc.fo1()>>2};
                true_block = 1;
            });
            return; // stop here
        }
        mispredict.fanout(hard<LINEINST+2>{});
        branch_offset.fanout(hard<LINEINST>{});
        branch_dir.fanout(hard<3>{});
        index1.fanout(hard<LINEINST*3>{});
        index2.fanout(hard<LINEINST*3>{});
        u64 update_valid = (u64(1)<<num_branch)-1;
        arr<val<LINEINST>,LINEINST> update_mask = [&](u64 offset){
            arr<val<1>,LINEINST> match_offset = [&](u64 i){return branch_offset[i] == offset;};
            return match_offset.fo1().concat() & update_valid;
        };
        update_mask.fanout(hard<2>{});
        arr<val<1>,LINEINST> is_branch = [&](u64 offset){
            return update_mask[offset] != hard<0>{};
        };
        val<LINEINST> branch_mask = is_branch.fo1().concat();
        branch_mask.fanout(hard<3>{});
        val<LINEINST> actualdirs = branch_dir.concat();
        actualdirs.fanout(hard<LINEINST>{});
        arr<val<1>,LINEINST> branch_taken = [&](u64 offset){
            return (actualdirs & update_mask[offset]) != hard<0>{};
        };

        arr<val<1>,LINEINST> last_branch = branch_offset[num_branch-1].decode();
        arr<val<1>,LINEINST> mispredicted = [&](u64 i){
            return last_branch[i].fo1() & mispredict;
        };
        mispredicted.fanout(hard<2>{});

        // do P1 and P2 agree?
        val<LINEINST> disagree_mask = (p1 ^ p2) & branch_mask;
        disagree_mask.fanout(hard<2>{});
        arr<val<1>,LINEINST> disagree = disagree_mask.make_array(val<1>{});
        disagree.fanout(hard<2>{});

        // read the P1 hysteresis if P1 and P2 disagree
        arr<val<1>,LINEINST> p1_weak = [&] (u64 i) -> val<1> {
            // returns 1 iff disagreement and hysteresis bit is weak
            return execute_if(disagree[i], [&](){
                return ~table1_lo[i].read(index1); // 0=weak
            });
        };

        // read the P2 hysteresis if misprediction
        arr<val<1>,LINEINST> p2_weak = [&] (u64 i) -> val<1> {
            // returns 1 iff mispredicted and hysteresis bit is weak
            return execute_if(mispredicted[i], [&](){
                return ~table2_lo[i].read(index2); // 0=weak
            });
        };

        // we need an extra cycle if there is a mispredict or a disagreement
        need_extra_cycle(mispredict | (disagree_mask != hard<0>{}));

        // update P1 prediction if P1 and P2 disagree and the hysteresis bit is weak
        execute_if(p1_weak.fo1().concat(), [&](u64 i){
            // update with the P2 prediction, not with the actual branch direction
            table1_hi[i].write(index1,ctr2_hi[i]);
        });
        // update P1 hysteresis
        execute_if(branch_mask,[&](u64 i){
            table1_lo[i].write(index1,~disagree[i]);
        });

        // update P2 prediction if misprediction and the hysteresis bit is weak
        execute_if(p2_weak.fo1().concat(), [&](u64 i){
            table2_hi[i].write(index2,branch_taken[i].fo1());
        });
        // update P2 hysteresis
        execute_if(branch_mask,[&](u64 i){
            table2_lo[i].write(index2,~mispredicted[i]);
        });

        // update the global history
        val<1> line_end = block_entry >> (LINEINST-block_size);
        true_block = ~mispredict.fo1() | branch_dir[num_branch-1] | line_end.fo1();
        execute_if(true_block, [&](){
            global_history = (global_history << 1) ^ val<GHIST2>{next_pc.fo1()>>2};
        });

        num_branch = 0; // done
    }

    ~gshare()
    {
#ifdef VERBOSE
        panel.print(std::cerr);
#endif
    }
};
