#include "../harcom.hpp"
#include "common.hpp"

using namespace hcm;


template<u64 LOGLB=6, u64 LOGP2=17, u64 LOGP1=12>
struct bimodal : predictor {
  // P1: bimodal with 2^LOGP1 entries
  // P2: bimodal with 2^LOGP2 entries
  // provides 2^(LOGLB-2) predictions per cycle
  static_assert(LOGLB>2);
  static constexpr u64 LOGLINEINST = LOGLB-2;
  static constexpr u64 LINEINST = 1<<LOGLINEINST;
  static_assert(LOGP1 > LOGLINEINST);
  static_assert(LOGP2 > LOGLINEINST);
  static constexpr u64 index1_bits = LOGP1-LOGLINEINST;
  static constexpr u64 index2_bits = LOGP2-LOGLINEINST;

  reg<index1_bits> index1;
  arr<reg<2>,LINEINST> ctr1; // counters read from the P1 table
  reg<LINEINST> p1; // predictions
  reg<index2_bits> index2;
  arr<reg<2>,LINEINST> ctr2; // counters read from the P2 table
  reg<LINEINST> p2; // predictions

  // simulation artifacts, hardware cost may not be real
  u64 num_branch = 0;
  u64 block_size = 0;
  arr<reg<LOGLINEINST>,LINEINST> branch_offset;
  arr<reg<1>,LINEINST> branch_dir;
  reg<LINEINST> block_entry; // one-hot vector

  ram<val<2>,(1<<index1_bits)> table1[LINEINST]; // P1
  ram<val<2>,(1<<index2_bits)> table2[LINEINST]; // P2

  void new_block(val<64> inst_pc)
  {
    val<LOGLINEINST> offset = inst_pc.fo1() >> 2;
    block_entry = offset.fo1().decode().concat();
    block_entry.fanout(hard<LINEINST*2>{});
    block_size = 1;
  }
  
  val<1> predict1([[maybe_unused]] val<64> inst_pc)
  {
    inst_pc.fanout(hard<2>{});
    new_block(inst_pc);
    index1 = inst_pc >> LOGLB;
    index1.fanout(hard<LINEINST>{});
    for (u64 i=0; i<LINEINST; i++) {
      ctr1[i] = table1[i].read(index1);
    }
    ctr1.fanout(hard<2>{});
    p1 = arr<val<1>,LINEINST>{[&](u64 i){
      return ctr1[i] >> 1;
    }}.concat();
    p1.fanout(hard<LINEINST>{});
    return (block_entry & p1) != hard<0>{};
  };

  val<1> reuse_predict1([[maybe_unused]] val<64> inst_pc)
  {
    return ((block_entry<<block_size) & p1) != hard<0>{};
  };

  val<1> predict2(val<64> inst_pc)
  {
    inst_pc.fanout(hard<2>{});
    index2 = inst_pc >> LOGLB;
    index2.fanout(hard<LINEINST>{});
    for (u64 i=0; i<LINEINST; i++) {
      ctr2[i] = table2[i].read(index2);
    }
    ctr2.fanout(hard<2>{});
    p2 = arr<val<1>,LINEINST>{[&](u64 i){return ctr2[i]>>1;}}.concat();
    p2.fanout(hard<LINEINST>{});
    val<1> taken = (block_entry & p2) != hard<0>{};
    taken.fanout(hard<2>{});
    reuse_prediction(~val<1>{block_entry>>(LINEINST-1)});
    return taken;
  }

  val<1> reuse_predict2([[maybe_unused]] val<64> inst_pc)
  {
    val<1> taken = ((block_entry<<block_size) & p2) != hard<0>{};
    taken.fanout(hard<2>{});
    reuse_prediction(~val<1>{block_entry>>(LINEINST-1-block_size)});
    block_size++;
    return taken;
  }

  void update_condbr(val<64> branch_pc, val<1> taken, [[maybe_unused]] val<64> next_pc)
  {
    assert(num_branch < LINEINST);
    branch_offset[num_branch] = branch_pc.fo1() >> 2;
    branch_dir[num_branch] = taken.fo1();
    num_branch++;
  }

  void update_cycle([[maybe_unused]] val<1> mispredict, [[maybe_unused]] val<64> next_pc)
  {
    // updates for all conditional branches in the predicted block
    if (num_branch == 0) {
      return;
    }
    branch_offset.fanout(hard<LINEINST>{});
    branch_dir.fanout(hard<2>{});
    index1.fanout(hard<LINEINST>{});
    index2.fanout(hard<LINEINST>{});
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
    branch_mask.fanout(hard<2>{});
    val<LINEINST> actualdirs = branch_dir.concat();
    actualdirs.fanout(hard<LINEINST>{});
    arr<val<1>,LINEINST> branch_taken = [&](u64 offset){
      return (actualdirs & update_mask[offset]) != hard<0>{};
    };
    // Update P1 with the P2 prediction, not with the actual branch direction
    auto p2_split = p2.make_array(val<1>{});
    execute_if(branch_mask, [&](u64 i){
      table1[i].write(index1,update_ctr(ctr1[i],p2_split[i].fo1()));
    });
    // update P2 with the actual branch direction
    execute_if(branch_mask, [&](u64 i){
      table2[i].write(index2,update_ctr(ctr2[i],branch_taken[i].fo1()));
    });
    num_branch = 0; // done
  }

  ~bimodal()
  {
#ifdef VERBOSE
    panel.print(std::cerr);
#endif
  }
};
