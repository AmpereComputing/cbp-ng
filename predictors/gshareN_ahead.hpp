#include "../cbp.hpp"
#include "../harcom.hpp"

using namespace hcm;

// This predictor uses ahead indexing:
//
//   [1] Seznec et al., "Multiple-block ahead branch predictors", ASPLOS 1996.
//   [2] Michaud et al., "Exploring instruction-fetch bandwidth requirement in
//       wide-issue superscalar processors", PACT 1999.
//   [3] Seznec & Fraboulet, "Effective ahead pipelining of instruction block
//       address generation", ISCA 2003.
//
// Instead of indexing block B1's prediction with the address of block B1, we index it
// with the address of the previous block, B0, and the path followed out of block B0.
// N is the maximum number of branches predicted per cycle (at most one taken branch).
// Unless block B0 ends on an indirect jump, there are at most N+1 paths out of B0.
// The N+1 possible block predictions are read simultaneously from multiple banks.
// Once the path is known in the next cycle, it is used to select one of the N+1 block predictions.
// As the N+1 paths out of B0 are not equally likely, and in order to use storage evenly,
// the bank associated with a given path depends on some bits XB of B0's address via XORing.
// Each of the N predictions for B1 is associated with a lane.
// To use lanes evenly, the lane depends on some bits XL of B1's address via XORing.


template<u64 LOGG=19, u64 GHIST=12, u64 N=4>
struct gshareN_ahead : predictor {
    // gshare with 2^LOGG entries, single prediction level (no overriding)
    // global history of GHIST bits
    // predicts up to N branches per cycle
    static constexpr u64 LOGLANES = std::bit_width(N-1);
    static constexpr u64 LANES = u64(1) << LOGLANES;
    static constexpr u64 PATHS = N+1;
    static constexpr u64 LOGBANKS = std::bit_width(PATHS-1);
    static constexpr u64 BANKS = u64(1) << LOGBANKS;
    static_assert(LOGG>(LOGLANES+LOGBANKS));
    static constexpr u64 index_bits = LOGG-(LOGLANES+LOGBANKS);
    // a block does not continue past a line boundary
    static constexpr u64 LINEINST = 64; // line size in instructions
    static_assert(std::has_single_bit(LINEINST)); // power of 2
    static constexpr u64 LOGLINEINST = std::bit_width(LINEINST-1);

    reg<GHIST> global_history;

    // pipelined over 2 cycles ([0]=current block, [1]=previous block)
    reg<index_bits> index[2];
    reg<LOGBANKS> XB[2]; // for using banks evenly
    reg<LANES> XL; // for using lanes evenly
    reg<LOGBANKS> path[2]; // path out of previous block

    arr<reg<LANES>,BANKS> block_pred[2]; // read block predictions
    reg<LANES> unordered_pred; // read prediction bits for the current block, unordered
    arr<reg<1>,LANES> pred; // read prediction bits for the current block, ordered

    // a true block is a block whose length is the same whether or not there is a mispredict
    reg<1> true_block = 1;
    reg<1> condbr_taken = 1;

    // for detecting the line boundary and the last available prediction
    reg<LINEINST> block_entry; // one-hot vector pointing to entry point in the line
    reg<N> rank; // one-hot bit vector telling the rank of the current branch in the block

    // simulation artifacts, hardware cost not modeled accurately
    u64 num_branch = 0; // number of conditional branches in block so far
    u64 block_size = 0; // instructions in current block so far
    arr<reg<1>,N> branch_dir; // actual branch direction

    // RAMs
    ram<arr<val<LANES>,BANKS>,(1<<index_bits)> ctr_hi; // prediction bits
    ram<val<1>,(BANKS<<index_bits)> ctr_lo[LANES]; // hysteresis bit (1=weak, 0=strong)

    val<1> line_end()
    {
        return block_entry >> (LINEINST-block_size);
    }

    val<1> last_pred()
    {
        return rank.rotate_left(num_branch);
    }

    val<1> predict1([[maybe_unused]] val<64> inst_pc)
    {
        inst_pc.fanout(hard<3>{});
        global_history.fanout(hard<2>{});
        true_block.fanout(hard<6>{});

        // if the previous block was not a true block, we continue using the previous block predictions
        // (golden rule: never make a predictor's inputs depend on its outputs)

        block_entry = select(true_block,
                             val<LOGLINEINST>{inst_pc>>2}.decode().concat(),
                             block_entry<<block_size);
        block_entry.fanout(hard<LINEINST+N+1>{});

        rank = select(true_block, val<N>{1}, rank.rotate_left(num_branch));

        XL = select(true_block,
                    val<LOGLANES>{inst_pc>>2}.decode().concat(),
                    XL.rotate_left(num_branch));
        XL.fanout(hard<LANES>{});

        execute_if(true_block, [&](){
            index[1] = index[0];
            val<index_bits> pc_bits = inst_pc >> (LOGBANKS+2);
            if constexpr (GHIST <= index_bits) {
                index[0] = pc_bits.fo1() ^ (val<index_bits>{global_history}<<(index_bits-GHIST));
            } else {
                index[0] = global_history.make_array(val<index_bits>{}).append(pc_bits.fo1()).fold_xor();
            }
            block_pred[1] = block_pred[0].fo1();
            block_pred[0] = ctr_hi.read(index[0]);
            XB[1] = XB[0].fo1();
            XB[0] = inst_pc >> 2;
            path[1] = (path[0] + num_branch) & condbr_taken.replicate(hard<LOGBANKS>{}).concat();
            path[1].fanout(hard<3>{});
            unordered_pred = block_pred[1].select(path[1]^XB[1]);
            unordered_pred.fanout(hard<LANES>{});
        });

        path[0] = select(true_block, val<LOGBANKS>{0}, val<LOGBANKS>{path[0]+num_branch});
        path[0].fanout(hard<2>{});

        for (u64 i=0; i<LANES; i++) {
            pred[i] = (unordered_pred & XL.rotate_left(i)) != hard<0>{};
        }
        pred.fanout(hard<LINEINST*2>{});
        block_size = 1;
        num_branch = 0;
        reuse_prediction(~line_end());
        return pred[num_branch];
    };

    val<1> reuse_predict1([[maybe_unused]] val<64> inst_pc)
    {
        block_size++;
        reuse_prediction(~line_end());
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
        reuse_prediction(~(line_end() | last_pred()));
    }

    void update_cycle([[maybe_unused]] instruction_info &block_end_info)
    {
        val<1> &mispredict = block_end_info.is_mispredict;
        val<64> &next_pc = block_end_info.next_pc;
        if (num_branch == 0) {
            // no conditional branch in this block
            global_history = (global_history << 1) ^ val<GHIST>{next_pc.fo1()>>2};
            condbr_taken = 0;
            true_block = 1;
            return; // stop here
        }
        static_assert(LANES<=64);
        XL.fanout(hard<2*LANES>{});
        index[1].fanout(hard<2*LANES+1>{});
        branch_dir[num_branch-1].fanout(hard<LANES+LOGBANKS+1>{});
        mispredict.fanout(hard<4>{});

        // access = mask telling which lanes are accessed by branches in the block
        arr<val<1>,LANES> access = arr<val<LANES>,LANES> { [&](u64 i){
                return XL.rotate_left(i) & val<LANES>{-(i<num_branch)};
        }}.fold_or().make_array(val<1>{});

        // misp bank = bit vector pointing to the lane accessed by the mispredicted branch
        // (all zero if no mispredict)
        val<LANES> misp_bank = XL.rotate_left(num_branch-1) & mispredict.replicate(hard<LANES>{}).concat();
        arr<val<1>,LANES> mispredicted = misp_bank.fo1().make_array(val<1>{});
        mispredicted.fanout(hard<2>{});

        // determine the bank to update
        val<LOGBANKS> bank = path[1] ^ XB[1];
        bank.fanout(hard<2*LANES+BANKS>{});

        // read hysteresis bit iff mispredict
        // weak[i] = 1 iff bank #i corresponds to mispredicted branch and hysteresis is weak
        arr<val<1>,LANES> weak = [&](u64 i){
            return execute_if(mispredicted[i], [&](){
                return ctr_lo[i].read(concat(index[1],bank));
            });
        };

        // we need an extra cycle if there is a mispredict
        need_extra_cycle(mispredict);

        // update prediction if mispredict and the hysteresis bit is weak
        execute_if(mispredict, [&](){
            arr<val<1>,LANES> stored_pred = unordered_pred.make_array(val<1>{});
            val<LANES> block_bundle = arr<val<1>,LANES>{
                [&](u64 i){
                    return select(weak[i].fo1(),branch_dir[num_branch-1],stored_pred[i].fo1());
                }
            }.concat();
            block_bundle.fanout(hard<BANKS>{});
            arr<val<LANES>,BANKS> bundle = [&](u64 i){
                return select(bank==i, block_bundle, block_pred[1][i]);
            };
            ctr_hi.write(index[1],bundle.fo1());
        });

        // update hysteresis
        for (u64 i=0; i<LANES; i++) {
            execute_if(access[i].fo1(), [&](){
                ctr_lo[i].write(concat(index[1],bank),mispredicted[i].fo1());
            });
        }

        condbr_taken = branch_dir[num_branch-1];

        // update the global history if this is a true block
        true_block = arr<val<1>,4> {
            ~mispredict, condbr_taken, last_pred(), line_end()
        }.fold_or();

        execute_if(true_block, [&](){
            global_history = (global_history << 1) ^ val<GHIST>{next_pc.fo1()>>2};
        });

    }
};
