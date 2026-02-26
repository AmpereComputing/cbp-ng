#include "../cbp.hpp"
#include "../harcom.hpp"

using namespace hcm;


template<u64 LOGG=15, u64 GHIST=8, u64 LOGN=2>
struct gshareN : predictor {
    // gshare with 2^LOGG entries, single prediction level (no overriding)
    // global history of GHIST bits
    // provides 2^LOGN predictions per cycle
    // 1st pred is for first cond branch,
    // 2nd pred is for next cond branch (if first not taken), etc.
    static constexpr u64 N = 1<<LOGN; // number of banks = maximum number of branches per block
    static_assert(LOGG>LOGN);
    static constexpr u64 index_bits = LOGG-LOGN;
    // a block does not continue past a line boundary
    static constexpr u64 LINEINST = 64; // line size in instructions
    static_assert(std::has_single_bit(LINEINST)); // power of 2
    static constexpr u64 LOGLINEINST = std::bit_width(LINEINST-1);

    reg<GHIST> global_history;

    reg<N> X; // for using banks evenly
    reg<index_bits> index; // same index for all banks
    reg<N> unordered_pred; // read prediction bits, unordered
    arr<reg<1>,N> pred; // read prediction bits, ordered

    // a true block is a block whose length is the same whether or not there is a mispredict
    reg<1> true_block = 1;

    // for detecting the line boundary and the last available prediction
    reg<LINEINST> block_entry; // one-hot vector pointing to entry point in the line
    reg<N> rank; // one-hot bit vector telling the rank of the current branch in the block

    // simulation artifacts
    u64 num_branch = 0; // number of conditional branches in block so far
    u64 block_size = 0; // instructions in current block so far
    arr<reg<1>,N> branch_dir; // actual branch direction

    // RAMs
    ram<val<N>,(1<<index_bits)> ctr_hi; // prediction bits
    ram<val<1>,(1<<index_bits)> ctr_lo[N]; // hysteresis bits (1=weak, 0=strong)

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
        true_block.fanout(hard<4>{});
        // if the previous block was not a true block, we continue using the previous block predictions
        // (golden rule: never make a predictor's inputs depend on its outputs)
        block_entry = select(true_block,
                             val<LOGLINEINST>{inst_pc>>2}.decode().concat(),
                             block_entry<<block_size);
        block_entry.fanout(hard<LINEINST+N+1>{});
        rank = select(true_block,
                      val<N>{1},
                      rank.rotate_left(num_branch));
        rank.fanout(hard<N+1>{});
        X = select(true_block,
                   val<LOGN>{inst_pc>>2}.decode().concat(),
                   X.rotate_left(num_branch));
        X.fanout(hard<N>{});
        execute_if(true_block, [&](){
            val<index_bits> pc_bits = inst_pc >> (LOGN+2);
            if constexpr (GHIST <= index_bits) {
                index = pc_bits.fo1() ^ (val<index_bits>{global_history}<<(index_bits-GHIST));
            } else {
                index = global_history.make_array(val<index_bits>{}).append(pc_bits.fo1()).fold_xor();
            }
            unordered_pred = ctr_hi.read(index);
        });
        unordered_pred.fanout(hard<N>{});
        for (u64 i=0; i<N; i++) {
            pred[i] = (unordered_pred & X.rotate_left(i)) != hard<0>{};
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
            true_block = 1;
            return; // stop here
        }
        static_assert(N<=64);
        index.fanout(hard<N*3>{});
        branch_dir.fanout(hard<N>{});
        mispredict.fanout(hard<N+3>{});
        X.fanout(hard<N+1>{});

        // access = mask telling which banks are accessed by branches in the block
        val<N> access = arr<val<N>,N> { [&](u64 i)->val<N> {
                return X.rotate_left(i) & val<N>{-(i<num_branch)};
        }}.fold_or();

        // misp bank = bit vector pointing to the bank accessed by the mispredicted branch
        // (all zero if no mispredict)
        val<N> misp_bank = X.rotate_left(num_branch-1) & mispredict.replicate(hard<N>{}).concat();
        misp_bank.fanout(hard<2>{});
        arr<val<1>,N> mispredicted = misp_bank.make_array(val<1>{});

        // read hysteresis bit iff mispredict
        // weak[i] = 1 iff bank #i corresponds to mispredicted branch and hysteresis is weak
        arr<val<1>,N> weak = execute_if(misp_bank, [&](u64 i){
            return ctr_lo[i].read(index);
        });

        // we need an extra cycle if there is a mispredict
        need_extra_cycle(mispredict);

        // update prediction if mispredict and the hysteresis bit is weak
        execute_if(mispredict, [&](){
            arr<val<1>,N> stored_pred = unordered_pred.make_array(val<1>{});
            arr<val<1>,N> bundle = [&](u64 i){
                return select(weak[i].fo1(),branch_dir[num_branch-1],stored_pred[i].fo1());
            };
            ctr_hi.write(index,bundle.fo1().concat());
        });

        // update hysteresis
        execute_if(access.fo1(), [&](u64 i){
            ctr_lo[i].write(index,mispredicted[i].fo1());
        });

        // update the global history if this is a true block
        true_block = arr<val<1>,4> {
            ~mispredict, branch_dir[num_branch-1], last_pred(), line_end()
        }.fold_or();

        execute_if(true_block, [&](){
            global_history = (global_history << 1) ^ val<GHIST>{next_pc.fo1()>>2};
        });
    }
};
