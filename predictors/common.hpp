#ifndef COMMON_H
#define COMMON_H

#include "../harcom.hpp"

using namespace hcm;


// up-down saturating counter update function
template<u64 N, typename T>
[[nodiscard]] val<N,T> update_ctr(val<N,T> ctr, val<1> incr)
{
    ctr.fanout(hard<6>{});
    val<N,T> incsat = select(ctr==hard<ctr.maxval>{},ctr,val<N,T>{ctr+1});
    val<N,T> decsat = select(ctr==hard<ctr.minval>{},ctr,val<N,T>{ctr-1});
    return select(incr.fo1(),incsat.fo1(),decsat.fo1());
}


// global history updated by shifting left by one bit and then xoring with some branch bits
// (branch direction, PC bits, target bits, whatever...)
// N is the history length in bits
template<u64 N>
struct global_history {

    arr<reg<1>,N> h; // initial value = 0 (consistent with folded history)

    void update(valtype auto in)
    {
        auto input = in.fo1().make_array(val<1>{});
        static_assert(input.size<=N);
        for (u64 i=N-1; i>=input.size; i--) h[i] = h[i-1];
        for (u64 i=input.size-1; i>=1; i--) h[i] = h[i-1] ^ input[i].fo1();
        h[0] = input[0].fo1();
    }

    val<1>& operator[] (u64 i)
    {
        return h[i];
    }

    void fanout(hardval auto fo)
    {
        h.fanout(fo);
    }
};


// folding a global history means splitting it into equal size chunks and XORing all chunks together
// folded_gh does folding incrementally with a circular shift register
// see P. Michaud, "A PPM-like, tag-based branch predictor", Journal of ILP, vol. 7, 2005
template<u64 F>
struct folded_gh {
    static_assert(F!=0);

    reg<F> folded; // initial value = 0 (consistent with global history)

    val<F> get()
    {
        return folded;
    }

    void fanout(hardval auto fo)
    {
        folded.fanout(fo);
    }

    template<u64 MAXL>
    void update(global_history<MAXL> &gh, hardval auto ghlen, valtype auto in)
    {
        // left shift of global history corresponds to left rotate of folded history
        // the bit that is pushed out of the global history is XORed out of the folded history
        constexpr u64 inbits = std::min(F,std::min(in.size,ghlen.value));
        val<inbits> input = in.fo1(); // truncate input if longer than global history
        auto f = folded.make_array(val<1>{});
        static_assert(f.size==F);
        val<1> outbit = gh[ghlen-1];
        u64 outpos = ghlen % F;
        arr<val<1>,F> ff = [&](u64 i){
            if (i==0) {
                return (outpos==0)? f[F-1].fo1()^outbit.fo1() : f[F-1].fo1();
            } else {
                return (outpos==i)? f[i-1].fo1()^outbit.fo1() : f[i-1].fo1();
            }
        };
        auto x = input.fo1().make_array(val<1>{});
        arr<val<1>,F> y = [&](u64 i){return (i<x.size)? x[i].fo1()^ff[i].fo1() : ff[i].fo1();};
        folded = y.fo1().concat();
    }
};


// geometrically increasing global history lengths folds
// NH = number of history lengths, MINH = shortest, MAXH = longest
// FOLDS = fold sizes (in bits)
template<u64 NH, u64 MINH, u64 MAXH, u64... FOLDS>
struct geometric_folds {
    static_assert(NH>=2);
    static constexpr u64 NF = sizeof...(FOLDS); // number of folds per history length

    static constexpr auto HLEN = [] () {
        std::array<u64,NH> hlen;
        u64 prevhl = 0;
        for (u64 i=0; i<NH; i++) {
            u64 hl = MINH * mypow(f64(MAXH)/MINH,f64(i)/(NH-1));
            hl = std::max(prevhl+1,hl);
            hlen[NH-1-i] = hl;
            prevhl = hl;
        }
        return hlen;
    }();

    static_assert(HLEN[0]==MAXH); // HLEN[0] is the longest history

    global_history<MAXH> gh;
    std::array<std::tuple<folded_gh<FOLDS>...>,NH> folds;

    template<u64 J=0>
    auto get(u64 i)
    {
        if (i>=NH) {
            std::cerr << "geometric folds: out of bound access\n";
            std::terminate();
        }
        return std::get<J>(folds[i]).get();
    }

    void fanout(hardval auto fo)
    {
        for (u64 i=0; i<NH; i++) {
            static_loop<NF>([&]<u64 J>(){
                std::get<J>(folds[i]).fanout(fo);
            });
        }
    }

    void update(valtype auto branchbits)
    {
        // update folds before global history
        branchbits.fanout(hard<NH*NF+1>{});
        gh.fanout(hard<std::max(u64(2),NF+1)>{});
        static_loop<NH>([&]<u64 I>(){
            static_loop<NF>([&]<u64 J>(){
                std::get<J>(folds[I]).update(gh,hard<HLEN[I]>{},branchbits);
            });
        });
        gh.update(branchbits);
    }
};


#endif // COMMON_H
