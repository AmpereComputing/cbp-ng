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
    static constexpr u64 N = 1<<LOGN; // number of banks
    static_assert(LOGG>LOGN);
    static constexpr u64 index_bits = LOGG-LOGN;
    // a block does not continue past a line boundary
    static constexpr u64 LINEINST = 64; // line size in instructions
    static_assert(std::has_single_bit(LINEINST)); // power of 2
    static constexpr u64 LOGLINEINST = std::bit_width(LINEINST-1);

    reg<GHIST> global_history;

    // try to use all banks evenly
    // define X as the LOGN rightmost bits of the block address (inst_pc>>2)
    // 1st pred in bank X, 2nd pred in bank X^1, 3rd pred in bank X^2,...
    reg<LOGN> X;
    reg<index_bits> index; // same index for all banks
    arr<reg<1>,N> unordered_pred; // read prediction bits, unordered
    arr<reg<1>,N> pred; // read prediction bits, ordered
    ram<arr<val<1>,N>,(1<<index_bits)> ctr_hi; // prediction bits

    // simulation artifacts, hardware cost not modeled accurately
    reg<1> true_block = 1;
    reg<LINEINST> block_entry; // one-hot vector
    u64 num_branch = 0; // number of conditional branches in block so far
    u64 block_size = 0; // instructions in current block so far
    arr<reg<1>,N> branch_dir; // actual branch direction

    zone UPDATE_ONLY;
    ram<val<1>,(1<<index_bits)> ctr_lo[N]; // hysteresis bit (1=weak, 0=strong)

    val<1> predict1([[maybe_unused]] val<64> inst_pc)
    {
        // new block
        assert(num_branch==0);
        block_size = 1;
        val<index_bits> pc_bits = inst_pc >> (LOGN+2);
        if constexpr (GHIST <= index_bits) {
            index = pc_bits.fo1() ^ (val<index_bits>{global_history}<<(index_bits-GHIST));
        } else {
            index = global_history.make_array(val<index_bits>{}).append(pc_bits.fo1()).fold_xor();
        }
        unordered_pred = ctr_hi.read(index);
        unordered_pred.fanout(hard<N>{});
        X = inst_pc >> 2;
        X.fanout(hard<N>{});
        for (u64 i=0; i<N; i++) {
            pred[i] = unordered_pred.select(X^i);
        }
        pred.fanout(hard<LINEINST*2>{});
        block_entry = val<LOGLINEINST>{inst_pc>>2}.decode().concat();
        block_entry.fanout(hard<LINEINST>{});
        reuse_prediction(~val<1>{block_entry>>(LINEINST-1)});
        return pred[num_branch];
    };

    val<1> reuse_predict1([[maybe_unused]] val<64> inst_pc)
    {
        reuse_prediction(~val<1>{block_entry>>(LINEINST-1-block_size)});
        block_size++;
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
        if (num_branch == N) {
            // this was the last branch for which we have a prediction: end the block
            reuse_prediction(0);
        }
    }

    void update_cycle([[maybe_unused]] val<1> mispredict, [[maybe_unused]] val<64> next_pc)
    {
        if (num_branch == 0) {
            // no conditional branch in this block
            val<1> line_end = block_entry >> (LINEINST-block_size);
            // update global history if previous block ended on a mispredicted not-taken branch
            // (we are still in the same line, this is the last chunk)
            // or if the block ends before the line boundary (unconditional jump)
            execute_if(~true_block | ~line_end.fo1(), [&](){
                global_history = (global_history << 1) ^ val<GHIST>{next_pc.fo1()>>2};
                true_block = 1;
            });
            return; // stop here
        }
        static_assert(N<=64);
        X.fanout(hard<N*2>{});
        index.fanout(hard<N*3>{});
        branch_dir.fanout(hard<N>{});
        mispredict.fanout(hard<3>{});
        // determine banks that are accessed
        arr<val<N>,N> bank = [&](u64 i) -> val<N> {
            return (X^i).decode().concat();
        };
        bank.fanout(hard<2>{});
        arr<val<1>,N> access = [&](u64 i) -> val<1> {
            return (bank[i]>>num_branch) == hard<0>{};
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
        execute_if(mispredict, [&](){
            arr<val<1>,N> bundle = [&](u64 i){
                return select(weak[i].fo1(),branch_dir[num_branch-1],unordered_pred[i]);
            };
            ctr_hi.write(index,bundle);
        });
        // update hysteresis
        execute_if(access.fo1().concat(), [&](u64 i){
            ctr_lo[i].write(index,mispredicted[i].fo1());
        });
        // update the global history
        val<1> line_end = block_entry >> (LINEINST-block_size);
        true_block = (~mispredict & (num_branch<N)) | branch_dir[num_branch-1] | line_end.fo1();
        execute_if(true_block, [&](){
            global_history = (global_history << 1) ^ val<GHIST>{next_pc.fo1()>>2};
        });
        num_branch = 0;
    }
};
