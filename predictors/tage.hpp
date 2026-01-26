// this is a basic TAGE, not necessarily well optimized

#define USE_META
#define RESET_UBITS

#include "../harcom.hpp"
#include "common.hpp"

using namespace hcm;

template<u64 LOGLB=6, u64 NUMG=8, u64 LOGG=11, u64 LOGB=12, u64 TAGW=11, u64 GHIST=100, u64 LOGP1=14, u64 GHIST1=6>
struct tage : predictor {
    // provides 2^(LOGLB-2) predictions per cycle
    // P2 is a TAGE, P1 is a gshare
    static_assert(LOGLB>2);
    static_assert(NUMG>0);
    static constexpr u64 MINHIST = 2;
    static constexpr u64 METABITS = 4;
    static constexpr u64 UCTRBITS = 8;
    static constexpr u64 PATHBITS = 6;
#ifdef USE_META
    static constexpr u64 METAPIPE = 2;
#endif
    static constexpr u64 LOGLINEINST = LOGLB-2;
    static constexpr u64 LINEINST = 1<<LOGLINEINST;
    static_assert(LOGP1 > LOGLINEINST);
    static_assert(LOGB > LOGLINEINST);
    static constexpr u64 index1_bits = LOGP1-LOGLINEINST;
    static constexpr u64 bindex_bits = LOGB-LOGLINEINST;
    static_assert(TAGW > LOGLINEINST); // the unhashed line offset is part of the tag
    static constexpr u64 HTAGBITS = TAGW-LOGLINEINST; // hashed tag bits

    geometric_folds<NUMG,MINHIST,GHIST,LOGG,HTAGBITS> gfolds;
    reg<1> true_block = 1;

    // for P1
    reg<GHIST1> global_history1;
    reg<index1_bits> index1;
    arr<reg<1>,LINEINST> readp1; // prediction bits read from P1 table for each offset
    reg<LINEINST> p1; // P1 predictions

    // for P2
    reg<bindex_bits> bindex; // bimodal table index
    arr<reg<LOGG>,NUMG> gindex; // global tables indexes
    arr<reg<HTAGBITS>,NUMG> htag; // computed hashed tags

    arr<reg<1>,LINEINST> readb; // read bimodal prediction bit for each offset
    arr<reg<TAGW>,NUMG> readt; // read tags
    arr<reg<1>,NUMG> readc; // read predictions
    arr<reg<2>,NUMG> readh; // read hysteresis
    arr<reg<1>,NUMG> readu; // read u bits
    reg<NUMG> notumask; // read u bits, inverted

    arr<reg<NUMG+1>,LINEINST> match; // all matches for each offset
    arr<reg<NUMG+1>,LINEINST> match1; // longest match for each offset
    arr<reg<NUMG+1>,LINEINST> match2; // second longest match for each offset

    arr<reg<1>,LINEINST> pred1; // primary P2 prediction for each offset
    arr<reg<1>,LINEINST> pred2; // alternate P2 prediction for each offset
    reg<LINEINST> p2; // final P2 predictions

#ifdef USE_META
    arr<reg<METABITS,i64>,METAPIPE> meta; // select between pred1 and pred2
    arr<reg<1>,LINEINST> newly_alloc;
#endif

#ifdef RESET_UBITS
    reg<UCTRBITS> uctr; // u bits counter (reset u bits when counter saturates)
#endif

    // simulation artifacts, hardware cost may not be real
    u64 num_branch = 0;
    u64 block_size = 0;
    arr<reg<LOGLINEINST>,LINEINST> branch_offset;
    arr<reg<1>,LINEINST> branch_dir;
    reg<LINEINST> block_entry; // one-hot vector

    // P1 (gshare)
    ram<val<1>,(1<<index1_bits)> table1_pred[LINEINST] {"P1 pred"}; // P1 prediction bit

    // P2 (TAGE)
    ram<val<TAGW>,(1<<LOGG)> gtag[NUMG] {"tags"}; // tags
    ram<val<1>,(1<<LOGG)> gpred[NUMG] {"gpred"}; // predictions
    rwram<2,(1<<LOGG),4> ghyst[NUMG] {"ghyst"}; // hysteresis
    rwram<1,(1<<LOGG),4> ubit[NUMG] {"u"}; // "useful" bits
    ram<val<1>,(1<<bindex_bits)> bim[LINEINST] {"bpred"}; // bimodal prediction bits

    zone UPDATE_ONLY;
    ram<val<1>,(1<<index1_bits)> table1_hyst[LINEINST] {"P1 hyst"}; // P1 hysteresis
    ram<val<1>,(1<<bindex_bits)> bhyst[LINEINST] {"bhyst"}; // bimodal hysteresis

    tage()
    {
#ifdef TAGE_VERBOSE
        std::cerr << "TAGE history lengths: ";
        for (u64 i=0; i<NUMG; i++) std::cerr << gfolds.HLEN[i] << " ";
        std::cerr << std::endl;
        if (LOGG == HTAGBITS) {
            std::cerr << "WARNING: the tag function and index function are not different enough\n";
        }
#endif
    }

    void new_block(val<64> inst_pc)
    {
        val<LOGLINEINST> offset = inst_pc.fo1() >> 2;
        block_entry = offset.fo1().decode().concat();
        block_entry.fanout(hard<6*LINEINST>{});
        block_size = 1;
    }

    val<1> predict1([[maybe_unused]] val<64> inst_pc)
    {
        inst_pc.fanout(hard<2>{});
        new_block(inst_pc);
        val<std::max(index1_bits,GHIST1)> lineaddr = inst_pc >> LOGLB;
        lineaddr.fanout(hard<2>{});
        if constexpr (GHIST1 <= index1_bits) {
            index1 = lineaddr ^ (val<index1_bits>{global_history1}<<(index1_bits-GHIST1));
        } else {
            index1 = global_history1.make_array(val<index1_bits>{}).append(lineaddr).fold_xor();
        }
        index1.fanout(hard<LINEINST>{});
        for (u64 offset=0; offset<LINEINST; offset++) {
            readp1[offset] = table1_pred[offset].read(index1);
        }
        readp1.fanout(hard<2>{});
        p1 = readp1.concat();
        p1.fanout(hard<LINEINST>{});
        return (block_entry & p1) != hard<0>{};
    };

    val<1> reuse_predict1([[maybe_unused]] val<64> inst_pc)
    {
        return ((block_entry<<block_size) & p1) != hard<0>{};
    };

    val<1> predict2(val<64> inst_pc)
    {
        val<std::max(bindex_bits,LOGG)> lineaddr = inst_pc >> LOGLB;
        lineaddr.fanout(hard<1+NUMG*2>{});
        gfolds.fanout(hard<2>{});

        // compute indexes
        bindex = lineaddr;
        bindex.fanout(hard<LINEINST>{});
        for (u64 i=0; i<NUMG; i++) {
            gindex[i] = lineaddr ^ gfolds.template get<0>(i);
        }
        gindex.fanout(hard<4>{});

        // compute hashed tags
        for (u64 i=0; i<NUMG; i++) {
            htag[i] = val<HTAGBITS>{lineaddr}.reverse() ^ gfolds.template get<1>(i);
        }
        htag.fanout(hard<2>{});

        // read tables
        for (u64 offset=0; offset<LINEINST; offset++) {
            readb[offset] = bim[offset].read(bindex);
        }
        readb.fanout(hard<2>{});
        for (u64 i=0; i<NUMG; i++) {
            readt[i] = gtag[i].read(gindex[i]);
            readc[i] = gpred[i].read(gindex[i]);
            readh[i] = ghyst[i].read(gindex[i]);
            readu[i] = ubit[i].read(gindex[i]);
        }
        readt.fanout(hard<LINEINST+1>{});
        readc.fanout(hard<3>{});
        readh.fanout(hard<2>{});
        readu.fanout(hard<2>{});
        notumask = ~readu.concat();
        notumask.fanout(hard<2>{});

        // gather prediction bits for each offset
        val<NUMG> gpreds = readc.concat();
        gpreds.fanout(hard<LINEINST>{});
        arr<val<NUMG+1>,LINEINST> preds = [&](u64 offset){return concat(readb[offset],gpreds);};
        preds.fanout(hard<2*LINEINST>{});

        // hashed tags comparisons
        arr<val<1>,NUMG> htagcmp_split = [&](int i){return val<HTAGBITS>{readt[i]} == htag[i];};
        val<NUMG> htagcmp = htagcmp_split.fo1().concat();
        htagcmp.fanout(hard<LINEINST>{});

        // generate match mask for each offset
        static_loop<LINEINST>([&]<u64 offset>(){
            arr<val<1>,NUMG> tagcmp = [&](int i){return val<LOGLINEINST>{readt[i]>>HTAGBITS} == hard<offset>{};};
            match[offset] = concat(val<1>{1}, tagcmp.fo1().concat() & htagcmp); // bimodal is default when no match
        });
        match.fanout(hard<2>{});

        // for each offset, find longest match and select primary prediction
        for (u64 offset=0; offset<LINEINST; offset++) {
            match1[offset] = match[offset].one_hot();
        }
        match1.fanout(hard<3>{});
        for (u64 offset=0; offset<LINEINST; offset++) {
            pred1[offset] = (match1[offset] & preds[offset]) != hard<0>{};
        }
        pred1.fanout(hard<2>{});

        // for each offset, find second longest match and select secondary prediction
        for (u64 offset=0; offset<LINEINST; offset++) {
            match2[offset] = (match[offset]^match1[offset]).one_hot();
        }
        match2.fanout(hard<2>{});
        for (u64 offset=0; offset<LINEINST; offset++) {
            pred2[offset] = (match2[offset] & preds[offset]) != hard<0>{};
        }
        pred2.fanout(hard<2>{});

#ifdef USE_META
        meta.fanout(hard<2>{});
        arr<val<1>,NUMG> weakctr = [&](int i) {return readh[i]==hard<0>{};};
        val<NUMG> coldctr = notumask & weakctr.fo1().concat();
        coldctr.fanout(hard<LINEINST>{});
        val<1> metasign = (meta[METAPIPE-1] >= hard<0>{});
        metasign.fanout(hard<LINEINST>{});
        for (u64 offset=0; offset<LINEINST; offset++) {
            newly_alloc[offset] = (match1[offset] & coldctr) != hard<0>{};
        }
        newly_alloc.fanout(hard<2>{});
        arr<val<1>,LINEINST> altsel = [&](u64 offset){
            arr<val<1>,3> inputs = {metasign, newly_alloc[offset], match2[offset]!=hard<0>{}};
            return inputs.fo1().fold_and();
        };
        p2 = arr<val<1>,LINEINST> {[&](u64 offset){
            return select(altsel[offset].fo1(),pred2[offset],pred1[offset]);
        }}.concat();
#else
        p2 = pred1.concat();
#endif
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

    void update_cycle(val<1> mispredict, val<64> next_pc)
    {
        // updates for all conditional branches in the predicted block
        if (num_branch == 0) {
            // no conditional branch in this block
            val<1> line_end = block_entry >> (LINEINST-block_size);
            // update global history if previous block ended on a mispredicted not-taken branch
            // (we are still in the same line, this is the last chunk)
            // or if the block ends before the line boundary (unconditional jump)
            execute_if(~true_block | ~line_end.fo1(), [&](){
                next_pc.fanout(hard<2>{});
                global_history1 = (global_history1 << 1) ^ val<GHIST1>{next_pc>>2};
                gfolds.update(val<PATHBITS>{next_pc>>2});
                true_block = 1;
            });
            return; // stop here
        }
        mispredict.fanout(hard<NUMG+2>{});
        val<1> correct_pred = ~mispredict;
        correct_pred.fanout(hard<NUMG+2>{});
        index1.fanout(hard<LINEINST*3>{});
        p2.fanout(hard<2>{});
        bindex.fanout(hard<LINEINST*3>{});
        gindex.fanout(hard<4>{});
        htag.fanout(hard<3>{});
        readb.fanout(hard<2>{});
        readt.fanout(hard<4>{});
        readc.fanout(hard<2>{});
        match1.fanout(hard<3>{});
        match2.fanout(hard<2>{});
        pred1.fanout(hard<2>{});
        pred2.fanout(hard<2+NUMG>{});
        branch_offset.fanout(hard<LINEINST+NUMG+1>{});
        branch_dir.fanout(hard<2>{});
        gfolds.fanout(hard<2>{});
#ifdef USE_META
        meta.fanout(hard<2>{});
#endif
        val<LOGLINEINST> last_offset = branch_offset[num_branch-1];
        last_offset.fanout(hard<4*NUMG+2>{});

        u64 update_valid = (u64(1)<<num_branch)-1;
        arr<val<LINEINST>,LINEINST> update_mask = [&](u64 offset){
            arr<val<1>,LINEINST> match_offset = [&](u64 i){return branch_offset[i] == offset;};
            return match_offset.fo1().concat() & update_valid;
        };
        update_mask.fanout(hard<2>{});

        arr<val<1>,LINEINST> is_branch = [&](u64 offset){
            return update_mask[offset] != hard<0>{};
        };
        is_branch.fanout(hard<5>{});

        val<LINEINST> branch_mask = is_branch.concat();
        branch_mask.fanout(hard<2>{});

        val<LINEINST> actualdirs = branch_dir.concat();
        actualdirs.fanout(hard<LINEINST>{});

        arr<val<1>,LINEINST> branch_taken = [&](u64 offset){
            return (actualdirs & update_mask[offset]) != hard<0>{};
        };
        branch_taken.fanout(hard<3>{});

        arr<val<NUMG+1>,LINEINST> actual_match1 = [&] (u64 offset) {
            return select(is_branch[offset],match1[offset],val<NUMG+1>{0});
        };
        actual_match1.fanout(hard<2>{});

        val<NUMG> primary_mask = actual_match1.fold_or();
        primary_mask.fanout(hard<3>{});

        arr<val<1>,NUMG> primary = primary_mask.make_array(val<1>{});
        primary.fanout(hard<2>{});

        arr<val<1>,LINEINST> primary_wrong = [&](u64 offset){
            return pred1[offset] != branch_taken[offset];
        };
        primary_wrong.fanout(hard<2>{});

        // select some candidate entries for allocation
        val<NUMG> mispmask = mispredict.replicate(hard<NUMG>{}).concat();
        arr<val<1>,NUMG> last_tagcmp = [&](int i){return readt[i] == concat(last_offset,htag[i]);};
        val<NUMG+1> last_match1 = last_tagcmp.fo1().append(1).concat().one_hot();
        last_match1.fanout(hard<2>{});
        val<NUMG> postmask = mispmask.fo1() & val<NUMG>(last_match1-1);
        postmask.fanout(hard<2>{});
        val<NUMG> candallocmask = postmask & notumask; // candidate post entries for allocation
        candallocmask.fanout(hard<2>{});
        // if multiple candidate entries, we select a single one, with some randomization
        val<NUMG> collamask = candallocmask.reverse();
        collamask.fanout(hard<2>{});
        val<NUMG> collamask1 = collamask.one_hot();
        collamask1.fanout(hard<3>{});
        val<NUMG> collamask2 = (collamask^collamask1).one_hot();
        val<NUMG> collamask12 = select(val<2>{std::rand()}==hard<0>{}, collamask2.fo1(), collamask1);
        val<NUMG> allocmask = collamask12.fo1().reverse();
        allocmask.fanout(hard<5>{});
        arr<val<1>,NUMG> allocate = allocmask.make_array(val<1>{});
        allocate.fanout(hard<5>{});

        // associate a branch direction to each global table
        arr<val<1>,NUMG> bdir = [&](u64 i) {
            val<LOGLINEINST> tag_offset = readt[i] >> HTAGBITS;
            val<LOGLINEINST> offset = select(allocate[i],last_offset,tag_offset.fo1());
            offset.fanout(hard<LINEINST>{});
            arr<val<1>,LINEINST> match_offset = [&](u64 j){return branch_offset[j] == offset;};
            return (match_offset.fo1().concat() & update_valid & actualdirs) != hard<0>{};
        };
        bdir.fanout(hard<2>{});

        // tell if global prediction is incorrect
        arr<val<1>,NUMG> badpred1 = [&](u64 i){
            return readc[i] != bdir[i];
        };
        badpred1.fanout(hard<3>{});

        // associate to each global table a bit telling if local prediction differs from secondary prediction
        arr<val<1>,NUMG> altdiffer = [&](u64 i){
            val<LOGLINEINST> tag_offset = readt[i] >> HTAGBITS;
            return readc[i] != pred2.select(tag_offset.fo1());
        };

        // associate to each global table a bit telling if prediction for owning branch is correct
        arr<val<1>,NUMG> goodpred = [&](u64 i){
            val<LOGLINEINST> tag_offset = readt[i] >> HTAGBITS;
            return (tag_offset.fo1() != last_offset) | correct_pred;
        };

        // do P1 and P2 agree?
        val<LINEINST> disagree_mask = (p1 ^ p2) & branch_mask;
        disagree_mask.fanout(hard<2>{});
        arr<val<1>,LINEINST> disagree = disagree_mask.make_array(val<1>{});
        disagree.fanout(hard<2>{});

        // read the P1 hysteresis if P1 and P2 disagree
        arr<val<1>,LINEINST> p1_weak = [&] (u64 offset) -> val<1> {
            // returns 1 iff disagreement and hysteresis is weak
            return execute_if(disagree[offset], [&](){
                return ~table1_hyst[offset].read(index1); // hyst=0 means weak
            });
        };

        // read the bimodal hysteresis if bimodal caused a misprediction
        arr<val<1>,LINEINST> b_weak = [&] (u64 offset) -> val<1> {
            // returns 1 iff cause of misprediction and hysteresis is weak
            val<1> bim_primary = actual_match1[offset] >> NUMG;
            return execute_if(bim_primary.fo1() & primary_wrong[offset], [&](){
                return ~bhyst[offset].read(bindex); // hyst=0 means weak
            });
        };

        // determine which primary global predictions are incorrect with a weak hysteresis
        arr<val<1>,NUMG> g_weak = [&] (u64 i) -> val<1> {
            // returns 1 iff incorrect primary prediction and hysteresis is weak
            return primary[i] & badpred1[i] & (readh[i]==hard<0>{});
        };

        // need extra cycle for modifying prediction bits and for TAGE allocation
        val<1> some_badpred1 = (primary_mask & badpred1.concat()) != hard<0>{};
        val<1> extra_cycle = some_badpred1.fo1() | mispredict | (disagree_mask != hard<0>{});
        extra_cycle.fanout(hard<NUMG*2+1>{});
        need_extra_cycle(extra_cycle);

#ifdef USE_META
        // update meta counter
        arr<val<1>,LINEINST> altdiff = [&](u64 offset){
            // for each offset, tell if primary and secondary predictions differ
            return (match2[offset] != hard<0>{}) & (pred2[offset] != pred1[offset]);
        };
        arr<val<2,i64>,LINEINST> meta_incr = [&](u64 offset) -> val<2,i64> {
            val<1> update_meta = is_branch[offset] & altdiff[offset].fo1() & newly_alloc[offset];
            val<1> bad_pred2 = (pred2[offset] != branch_taken[offset]);
            return select(update_meta.fo1(),concat(bad_pred2.fo1(),val<1>{1}),val<2>{0});
        };
        for (u64 i=METAPIPE-1; i!=0; i--) {
            meta[i] = meta[i-1];
        }
        auto newmeta = meta[0] + meta_incr.fo1().fold_add();
        newmeta.fanout(hard<3>{});
        using meta_t = valt<decltype(meta[0])>;
        meta[0] = select(newmeta>meta_t::maxval, meta_t{meta_t::maxval}, select(newmeta<meta_t::minval, meta_t{meta_t::minval}, meta_t{newmeta}));
#endif

        // overwrite the tag in the allocated entry (mispredict)
        execute_if(allocmask, [&](u64 i){gtag[i].write(gindex[i],concat(last_offset,htag[i]));});

        // update the u bits
        val<NUMG> umask = primary_mask & altdiffer.fo1().concat(); // u bits update mask
        // if all post entries have the u bit set, reset their u bits
        val<1> noalloc = (candallocmask == hard<0>{});
        val<NUMG> uclearmask = postmask & noalloc.fo1().replicate(hard<NUMG>{}).concat();
        uclearmask.fanout(hard<2>{});
        arr<val<1>,NUMG> uclear = uclearmask.make_array(val<1>{});
        execute_if(umask.fo1() | allocmask | uclearmask, [&](u64 i) {
            val<1> newu = goodpred[i].fo1() & ~allocate[i] & ~uclear[i].fo1();
            ubit[i].write(gindex[i],newu.fo1(),extra_cycle);
        });

        // update P1 prediction if P1 and P2 disagree and the hysteresis bit is weak
        auto p2_split = p2.make_array(val<1>{});
        execute_if(p1_weak.fo1().concat(), [&](u64 offset){
            // update with the P2 prediction, not with the actual branch direction
            table1_pred[offset].write(index1,p2_split[offset].fo1());
        });
        // update P1 hysteresis
        execute_if(branch_mask,[&](u64 offset){
            table1_hyst[offset].write(index1,~disagree[offset]);
        });

        // update incorrect bimodal prediction if primary provider and hysteresis is weak
        execute_if(b_weak.fo1().concat(), [&](u64 offset){
            bim[offset].write(bindex,branch_taken[offset]);
        });
        // update bimodal hysteresis if bimodal is primary provider
        for (u64 offset=0; offset<LINEINST; offset++) {
            val<1> bim_primary = match1[offset] >> NUMG;
            execute_if(is_branch[offset] & bim_primary.fo1(), [&](){
                bhyst[offset].write(bindex,~primary_wrong[offset]);
            });
        }

        // update incorrect global prediction if primary provider and the hysteresis is weak;
        // initialize global prediction in the allocated entry
        execute_if(g_weak.fo1().concat() | allocmask, [&](u64 i){
            gpred[i].write(gindex[i],bdir[i]);
        });
        // update global prediction hysteresis if primary provider or allocated entry
        execute_if(primary_mask | allocmask, [&](u64 i){
            // if allocated entry, set hysteresis to 0;
            // otherwise, increment hysteresis if correct pred, decrement if incorrect
            val<2> newhyst = select(allocate[i],val<2>{0},update_ctr(readh[i],~badpred1[i]));
            ghyst[i].write(gindex[i],newhyst.fo1(),extra_cycle);
        });

#ifdef RESET_UBITS
        uctr.fanout(hard<3>{});
        val<NUMG> allocmask1  = collamask1.reverse();
        allocmask1.fanout(hard<2>{});
        val<1> faralloc = (((last_match1>>3) | allocmask1).one_hot() ^ allocmask1) == hard<0>{};
        val<1> uctrsat = (uctr == hard<decltype(uctr)::maxval>{});
        uctrsat.fanout(hard<2>{});
        uctr = select(correct_pred,uctr,select(uctrsat,val<decltype(uctr)::size>{0},update_ctr(uctr,faralloc.fo1())));
        execute_if(uctrsat,[&](){for (auto &uram : ubit) uram.reset();});
#endif

        // update global history
        val<1> line_end = block_entry >> (LINEINST-block_size);
        true_block = correct_pred | branch_dir[num_branch-1] | line_end.fo1();
        execute_if(true_block, [&](){
            next_pc.fanout(hard<2>{});
            global_history1 = (global_history1 << 1) ^ val<GHIST1>{next_pc>>2};
            gfolds.update(val<PATHBITS>{next_pc>>2});
        });

        num_branch = 0; // done
    }
};
