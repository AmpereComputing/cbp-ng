// HARCOM: Hardware Complexity Model
// URL: https://gitlab.inria.fr/pmichaud/harcom
// Language: C++20

// Copyright (c) 2025 INRIA
// Copyright (c) 2025 Pierre Michaud

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software
// and associated documentation files (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge, publish, distribute,
// sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE
// AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef HARCOM_H
#define HARCOM_H

#include <cstdint>
#include <cassert>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <ios>
#include <type_traits>
#include <concepts>
#include <bitset>
#include <bit>
#include <string>
#include <utility>
#include <initializer_list>
#include <numeric>
#include <algorithm>
#include <tuple>
#include <array>
#include <vector>
#include <limits>
#include <exception>
#include <span>
#include <fstream>
#include <numbers>


class harcom_superuser; // has access to private members of class "val"


namespace hcm {

  using u8 = std::uint8_t;
  using u16 = std::uint16_t;
  using u32 = std::uint32_t;
  using u64 = std::uint64_t;
  using i8 = std::int8_t;
  using i16 = std::int16_t;
  using i32 = std::int32_t;
  using i64 = std::int64_t;
  using f32 = float;
  using f64 = double;

#ifdef ENABLE_FP
  // TODO
  template<typename T>
  concept arith = std::integral<T> || std::floating_point<T>;
#else
  template<typename T>
  concept arith = std::integral<T>;
#endif

  template<arith auto N>
  struct hard {
    using type = decltype(N);
    static constexpr type value = N;
    constexpr operator type() {return N;}
  };

  template<typename T>
  concept hardval  = requires(const T &x) {[]<arith auto N>(const hard<N>&){}(x);};

  template<typename T>
  concept intlike = std::integral<T> || (hardval<T> && std::integral<typename T::type>);

  template<typename T>
  concept arithlike = arith<T> || (hardval<T> && arith<typename T::type>);

  template<typename T>
  concept arraylike = requires (T &a) {
    {std::size(a)} -> std::unsigned_integral;
    {a[0]};
  };

  template<typename T>
  struct arraysize_impl {};

  template<typename T, u64 N>
  struct arraysize_impl<std::array<T,N>> {
    static constexpr u64 value = N;
  };

  template<typename T, u64 N>
  struct arraysize_impl<T[N]> {
    static constexpr u64 value = N;
  };

  template<typename T>
  constexpr u64 arraysize = arraysize_impl<std::remove_reference_t<T>>::value;

  template<arith T>
  constexpr u64 bitwidth = (std::same_as<T,bool>)? 1 : sizeof(T)*8;

  template<typename T, typename X, typename Y>
  concept unaryfunc = (std::same_as<X,void> && requires (T f) {{f()} -> std::convertible_to<Y>;}) || requires (T f, X i) {{f(i)} -> std::convertible_to<Y>;};

  template<typename T>
  concept action = requires (T f) {f();} || requires (T f, u64 i) {f(i);};

  template<u64 N, arith T> class val;
  template<u64 N, arith T> class reg;

  template<typename T>
  concept valtype = requires(const T &x) {[]<u64 N, arith U>(const val<N,U>&){}(x);};

  template<typename T>
  concept arrayofval = requires(const T &x) {[]<valtype U, u64 N>(const std::array<U,N>&){}(x);};

  template<valtype T, u64 N> class arr;

  template<typename T>
  concept arrtype = requires(const T &x) {[]<valtype U, u64 N>(const arr<U,N>&){}(x);};

  template<typename T>
  concept regtype = requires(const T &x) {[]<u64 N, arith U>(const reg<N,U>&){}(x);};

  template<typename T>
  concept memdatatype = (valtype<T> || arrtype<T>) && ! regtype<T> && ! regtype<typename T::type>;

  template<memdatatype T, u64 N> class ram;

  template<typename T>
  concept ramtype = requires(const T &x) {[]<memdatatype U, u64 N>(const ram<U,N>&){}(x);};


  // ######################################################

  template<typename T>
  struct toval_impl {};

  template<arith T>
  struct toval_impl<T> {
    using type = val<bitwidth<T>,T>;
  };

  template<valtype T>
  struct toval_impl<T> {
    using U = std::remove_reference_t<T>;
    using type = val<U::size,typename U::type>;
  };

  template<typename T>
  using toval = toval_impl<T>::type;

  template<typename T>
  struct base_impl {};

  template<arith T>
  struct base_impl<T> {using type = T;};

  template<valtype T>
  struct base_impl<T> {using type = toval<T>::type;};

  template<typename T>
  using base = base_impl<T>::type;

  template<typename T>
  concept ival = std::integral<std::remove_reference_t<T>> || std::integral<base<T>>;

  template<typename T>
  concept fval = std::floating_point<std::remove_reference<T>> || std::floating_point<base<T>>;

  template<typename T>
  inline constexpr u64 length = 0;

  template<valtype T>
  inline constexpr u64 length<T> = toval<T>::size;

  template<typename T1, typename ...T>
  struct valt_impl {};

  template<typename T>
  struct valt_impl<T> {
    using type = toval<T>;
  };

  template<ival T1, ival T2>
  struct valt_impl<T1,T2> {
    using type = std::conditional_t<(length<T2> > length<T1>),toval<T2>,toval<T1>>;
  };

  template<fval T1, fval T2>
  struct valt_impl<T1,T2> {
    using type = std::conditional_t<(length<T2> > length<T1>),toval<T2>,toval<T1>>;
  };

  template<ival T1, fval T2>
  struct valt_impl<T1,T2> {
    using type = toval<T2>;
  };

  template<fval T1, ival T2>
  struct valt_impl<T1,T2> {
    using type = toval<T1>;
  };

  template<typename T1, typename T2, typename ...T>
  struct valt_impl<T1,T2,T...> {
    using type = typename valt_impl<T1,typename valt_impl<T2,T...>::type>::type;
  };

  template<typename ...T>
  using valt = valt_impl<T...>::type;

  template<action A>
  struct action_return {};

  template<action A> requires (std::invocable<A>)
  struct action_return<A> {
    using type = std::invoke_result_t<A>;
  };

  template<action A> requires (std::invocable<A,u64>)
  struct action_return<A> {
    using type = std::invoke_result_t<A,u64>;
  };

  template<action A>
  using return_type = action_return<A>::type;

  // ######################################################
  // math functions

  // Unfortunately, in C++20, functions of the math library are not constexpr
  // Clang and GCC seem to behave differently (Clang follows the standard)

  constexpr f64 myfabs(f64 x)
  {
    return (x>=0)? x:-x;
  }

  constexpr f64 mysqrt(f64 x)
  {
    f64 y = 1;
    for (u64 i=0; i<100; i++) {
      f64 yy = y;
      y = (y+x/y)*0.5;
      y = (y+x/y)*0.5;
      y = (y+x/y)*0.5;
      y = (y+x/y)*0.5;
      if (myfabs(yy-y)<y*1e-6) break;
    }
    return y;
  }

  constexpr f64 ipow(f64 x, u64 n)
  {
    f64 y = 1;
    f64 p = x;
    while (n) {
      y *= (n&1)? p:1;
      p = p*p;
      n >>= 1;
    }
    return y;
  }

  constexpr f64 myexp(f64 x)
  {
    i64 n = x / std::numbers::ln2;
    f64 xx = x - n * std::numbers::ln2;
    f64 y = 1;
    for (u64 k=20; k>=1; k--) {
      y = 1+xx*y/k;
    }
    if (n>=0) {
      return y * ipow(2,n);
    } else {
      return y / ipow(2,-n);
    }
  }

  constexpr f64 mylog(f64 x)
  {
    assert(x>0);
    if (x==1) return 0;
    f64 y = 1;
    for (u64 i=0; i<100; i++) {
      f64 z = myexp(y);
      y += 2*(x-z)/(x+z);
      if (myfabs((z-x)/x)<1e-6) break;
    }
    return y;
  }

  constexpr f64 myceil(f64 x)
  {
    f64 y = 0;
    if (x>=0) {
      u64 n = x;
      if (f64(n)!=x) n++;
      y = n;
    } else {
      i64 n = x;
      y = n;
    }
    assert(myfabs(y-x)<1); // triggers when |x| huge
    return y;
  }

  constexpr f64 myfloor(f64 x)
  {
    f64 y = 0;
    if (x>=0) {
      u64 n = x;
      y = n;
    } else {
      i64 n = x;
      if (f64(n)!=x) n--;
      y = n;
    }
    assert(myfabs(y-x)<1); // triggers when |x| huge
    return y;
  }

  constexpr i64 myllround(f64 x)
  {
    return (x>=0)? x+0.5 : x-0.5;
  }

  constexpr f64 mypow(f64 x, f64 p)
  {
    if (p==0) return 1.;
    if (x==0) return 0;
    i64 n = p;
    if (f64(n)==p) {
      if (n>=0) {
        return ipow(x,n);
      } else {
        return 1. / ipow(x,-n);
      }
    }
    assert(x>0);
    return myexp(p*mylog(x));
  }


  // ######################################################
  // integer comparison functions

  template <typename T>
  concept standard_integral = std::integral<T> && ! std::same_as<T,char> && ! std::same_as<T,char8_t> && ! std::same_as<T,char16_t> && ! std::same_as<T,char32_t> && ! std::same_as<T,wchar_t> && ! std::same_as<T,bool>;

  template<typename T, typename U>
  bool is_equal(T x, U y)
  {
    if constexpr (standard_integral<T> && standard_integral<U>) {
      return std::cmp_equal(x,y);
    } else {
      return x == y;
    }
  }

  template<typename T, typename U>
  bool is_different(T x, U y)
  {
     if constexpr (standard_integral<T> && standard_integral<U>) {
      return std::cmp_not_equal(x,y);
    } else {
      return x != y;
    }
  }

  template<typename T, typename U>
  bool is_less(T x, U y)
  {
    if constexpr (standard_integral<T> && standard_integral<U>) {
      return std::cmp_less(x,y);
    } else {
      return x < y;
    }
  }

  template<typename T, typename U>
  bool is_greater(T x, U y)
  {
    if constexpr (standard_integral<T> && standard_integral<U>) {
      return std::cmp_greater(x,y);
    } else {
      return x > y;
    }
  }

  template<typename T, typename U>
  bool is_less_equal(T x, U y)
  {
    if constexpr (standard_integral<T> && standard_integral<U>) {
      return std::cmp_less_equal(x,y);
    } else {
      return x <= y;
    }
  }

  template<typename T, typename U>
  bool is_greater_equal(T x, U y)
  {
    if constexpr (standard_integral<T> && standard_integral<U>) {
      return std::cmp_greater_equal(x,y);
    } else {
      return x >= y;
    }
  }


  // ######################################################
  // miscellaneous functions

  constexpr auto to_unsigned(std::integral auto x) {return std::make_unsigned_t<decltype(x)>(x);}
  constexpr auto to_unsigned(f32 x) {return std::bit_cast<u32>(x);}
  constexpr auto to_unsigned(f64 x) {return std::bit_cast<u64>(x);}

  template<std::integral T>
  constexpr u64 minbits(T x)
  {
    // minimum number of bits to encode value x
    if constexpr (std::unsigned_integral<T>) {
      return std::bit_width(x);
    } else {
      static_assert(std::signed_integral<T>);
      if (x>=0) {
        return std::bit_width(u64(x))+1;
      } else {
        return std::bit_width(u64(-x-1))+1;
      }
    }
  }

  template<u64 N, std::integral T>
  constexpr auto truncate(T x)
  {
    if constexpr (N >= bitwidth<T>) {
      return x;
    } else {
      return x & ((std::make_unsigned_t<T>(1)<<N)-1);
    }
  }

  template<u64 N, arith T>
  constexpr auto ones(T x)
  {
    return std::popcount(truncate<N>(to_unsigned(x)));
  }

  constexpr u64 to_pow2(f64 x)
  {
    // returns the power of 2 closest to x
    u64 n0 = std::bit_floor(u64(myllround(myfloor(x))));
    u64 n1 = std::bit_ceil(u64(myllround(myceil(x))));
    return (x*x > n0*n1)? n1 : n0;
  }

  constexpr u8 bit_reversal(u8 x)
  {
    u8 y = x&1;
    for (int i=1; i<8; i++) {
      x >>= 1;
      y = (y<<1) | (x&1);
    }
    return y;
  }

  constexpr auto reversed_byte = [] () {
    std::array<u8,256> b;
    for (u64 i=0; i<256; i++) {
      b[i] = bit_reversal(i);
    }
    return b;
  }();

  auto constexpr reverse_bits(std::unsigned_integral auto x)
  {
    using T = decltype(x);
    if constexpr (std::same_as<T,bool>) {
      return x;
    } else {
      T y = reversed_byte[x & 0xFF];
      for (u64 i=1; i<sizeof(T); i++) {
        x >>= 8;
        y = (y<<8) | reversed_byte[x & 0xFF];
      }
      return y;
    }
  }

  template<u64 W, std::unsigned_integral T, u64 N>
  auto pack_bits(const std::array<T,N> &in)
  {
    // input array has W-bit elements
    // output array has 64-bit elements
    static_assert(N!=0);
    static constexpr u64 M = (N*W+63)/64;
    if constexpr (W==64) {
      static_assert(std::same_as<T,u64>);
      return in;
    } else {
      static_assert(W!=0 && W<64);
      constexpr u64 mask = (u64(1)<<W)-1;
      std::array<u64,M> out{};
      u64 nbits = 0;
      for (u64 i=0; i<N; i++) {
        u64 x = u64(in[i]) & mask;
        u64 j = nbits / 64;
        u64 pos = nbits % 64;
        assert(j<M);
        out[j] |= x << pos;
        if (pos+W > 64) {
          assert(pos!=0);
          assert(j+1<M);
          out[j+1] = x >> (64-pos);
        }
        nbits += W;
      }
      return out;
    }
  }

  template<u64 W, u64 N>
  auto unpack_bits(const std::array<u64,N> &in)
  {
    // input array has 64-bit elements
    // output array has W-bit elements
    static_assert(N!=0);
    constexpr u64 M = (N*64+W-1)/W;
    if constexpr (W==64) {
      return in;
    } else {
      static_assert(W!=0 && W<64);
      constexpr u64 mask = (u64(1)<<W)-1;
      std::array<u64,M> out{};
      u64 nbits = 0;
      for (u64 j=0; j<M; j++) {
        u64 i = nbits / 64;
        u64 pos = nbits % 64;
        assert(i<N);
        if (pos+W <= 64) {
          out[j] = (in[i] >> pos) & mask;
        } else {
          u64 right = in[i] >> pos;
          u64 left = (i+1<N)? in[i+1] : 0;
          out[j] = (right | (left << (64-pos))) & mask;
        }
        nbits += W;
      }
      return out;
    }
  }

  template<u64 N>
  constexpr void fractional(u64 a, u64 b, std::array<bool,N> &frac)
  {
    assert(a<b);
    // returns the first N bits of the (binary) fractional part of a/b
    // least significant bit (rightmost) is at position 0 of the array
    const u64 halfb = b/2 + (b&1);
    for (u64 i=0; i<N; i++) {
      if (a >= halfb) {
        // 2a >= b
        frac.at(N-1-i) = 1;
        a = 2*a-b;
      } else {
        // 2a < b
        frac.at(N-1-i) = 0;
        a = 2*a;
      }
      assert(a<b);
    }
  }

  constexpr u64 pow2_minus1(u64 m, u64 nmax)
  {
    // returns the smallest n <= nmax, n>0, such that 2^n-1 is a multiple of m
    if (!(m&1)) return 0; // m even
    if (m==1) return 1;
    assert(m>=3);
    u64 modm = 2;
    for (u64 n=1; n<=nmax; n++) {
      if (modm == 1) return n;
      modm = (modm*2) % m;
    }
    return 0; // n does not exist
  }

  constexpr u64 pow2_plus1(u64 m, u64 nmax)
  {
    // returns the smallest n <= nmax, n>0, such that 2^n+1 is a multiple of m
    if (!(m&1)) return 0; // m even
    if (m==1) return 1;
    assert(m>=3);
    u64 modm = 2;
    for (u64 n=1; n<=nmax; n++) {
      if (modm == m-1) return n;
      modm = (modm*2) % m;
    }
    return 0; // n does not exist
  }


  // ######################################################
  // utility for doing a static loop over an integer template argument (0,...,N-1)
  // the loop body is a C++ lambda with one integer template parameter

  template<typename T>
  concept static_loop_body = requires (T obj) {obj.template operator()<0>();};

  template <u64 ...I>
  constexpr void static_loop(static_loop_body auto F, std::integer_sequence<u64,I...>)
  {
    (F.template operator()<I>(),...);
  }

  template<u64 N>
  constexpr void static_loop(static_loop_body auto F)
  {
    static_loop(F,std::make_integer_sequence<u64,N>{});
  }


  // ######################################################
  // HARDWARE MODEL

  // wires: see Nikolic, FPGA 2021
  // assume unique linear capacitance (fF/um) for all metal layers
  inline constexpr f64 METALCAP_fF = 0.2; // fF/um
  inline constexpr f64 METALRES[] = {150/*Mx*/,25/*My*/}; // Ohm/um

  inline constexpr f64 PINV = 1; // ratio of drain capacitance to gate capacitance

  // TSMC 5nm, Chang, IEEE JSSC, jan 2021:
  // each transistor in 6T cell has single fin (for high density)
  // bitline capacitance per cell = 1 drain capacitance (Cd) + 1 Mx wire segment (Cw)
  // Flying bitline (Mx+2) reduces bitline capacitance: 50%(bottom)/65%(top)
  // bottom bitline = (N/2)x(Cd+Cw)
  // top bitline = (N/2)x(Cd+Cw) + (N/2)xCw
  // (Cd+2Cw)/(Cd+Cw) = 65/50 ==> Cd/Cw = 0.7/0.3 = 2.33
  inline constexpr struct {
    u64 transistors = 6; // single RW port
    u64 fins = transistors; // single-fin transistors (high density cell)
    f64 area = 0.02; // um^2
    f64 density = fins / area;
    f64 aspect_ratio = 2; // wordline longer than bitline
    f64 drain_blwire_cap_ratio = 2.33;
    f64 bitline_length = mysqrt(area/aspect_ratio); // um
    f64 wordline_length = bitline_length * aspect_ratio; // um
    f64 bitline_wirecap = bitline_length * METALCAP_fF; // fF
    f64 wordline_wirecap = wordline_length * METALCAP_fF; // fF
    f64 bitline_resistance = bitline_length * METALRES[0]; // Ohm
    f64 wordline_resistance = wordline_length * METALRES[0]; // Ohm
    f64 drain_capacitance = drain_blwire_cap_ratio * bitline_wirecap; // fF
    f64 gate_capacitance = drain_capacitance / PINV; // fF
    f64 bitline_capacitance = bitline_wirecap + drain_capacitance; // fF
    f64 wordline_capacitance = wordline_wirecap + 2 * gate_capacitance; // fF

    void print(std::ostream & os = std::cout) const
    {
      os << "gate capacitance (fF): " << gate_capacitance << std::endl;
      os << "drain capacitance (fF): " << drain_capacitance << std::endl;
      os << "wordline (per cell):" << std::endl;
      os << "  length (um): " << wordline_length << std::endl;
      os << "  wire resistance (Ohm): " << wordline_resistance << std::endl;
      os << "  wire capacitance (fF): " << wordline_wirecap << std::endl;
      os << "  capacitance (fF): " << wordline_capacitance << std::endl;
      os << "bitline (per cell):" << std::endl;
      os << "  length (um): " << bitline_length << std::endl;
      os << "  wire resistance (Ohm): " << bitline_resistance << std::endl;
      os << "  wire capacitance (fF): " << bitline_wirecap << std::endl;
      os << "  capacitance (fF): " << bitline_capacitance << std::endl;
    }
  } SRAM_CELL;

  // CGATE = gate capacitance of single-fin nFET
  inline constexpr f64 CGATE_fF = SRAM_CELL.gate_capacitance; // fF
  inline constexpr f64 CGATE_pF = CGATE_fF * 1e-3; // pF
  inline constexpr f64 METALCAP = METALCAP_fF / CGATE_fF; // linear metal capacitance in units of CGATE
  inline constexpr f64 METALCAP_pF = METALCAP * CGATE_pF; // pF
  inline constexpr f64 VDD0 = 0.75; // nominal supply voltage (V)
  inline constexpr f64 VDD = 0.75; // supply voltage (V)

  // Idsat and Ioff values are taken from (FIXME?)
  // "ASAP5: a predictive PDK for the 5 nm node", Vashishtha & Clark, Microelectronics Journal 126 (2022)
  // FIXME: these numbers are for 0.7 V and 25 C
  inline constexpr f64 GM = 130e-6; // transconductance dIds/dVgs (A/V)
  inline constexpr f64 IDSAT = 60e-6 + GM * (VDD-VDD0); // nFET Idsat (A) per fin
  inline constexpr f64 IDSAT_SRAM = 40e-6 + GM * (VDD-VDD0); // nFET Idsat (A) per fin in an SRAM cell
  inline constexpr f64 IOFF = 1e-9; // leakage current per fin (A)
  inline constexpr f64 IOFF_SRAM = 17e-12; // leakage current per fin (A) in an SRAM cell

  // IEFF, REFF ==> Razavieh, Device Research Conference, 2018
  inline constexpr f64 IEFF = IDSAT/2; // nFET effective current per fin (A)
  inline constexpr f64 REFF = VDD/(2*IEFF); // nFET effective resistance (Ohm)
  inline constexpr f64 GAMMA = 1; // ratio of pFET to nFET capacitance at same conductance
  inline constexpr f64 INVCAP = 1+GAMMA; // input capacitance of single-fin inverter relative to CGATE
  inline constexpr f64 TAU_ps = CGATE_pF * REFF; // intrinsic delay (ps)

  inline constexpr f64 DSE = 6; // default stage effort (delay vs energy tradeoff)
  inline constexpr f64 DSMAX = 1000000; // default maximum scale is unlimited (FIXME?)


  constexpr f64 energy_fJ(f64 cap_fF, f64 vdiff/*volt*/)
  {
    // energy dissipated for charging a capacitance by VDIFF>0, then discharging by -VDIFF
    assert(vdiff>0);
    return cap_fF * vdiff * VDD; // fJ
  }


  constexpr f64 leakage_power_mW(u64 total_fins, u64 sram_cells)
  {
    u64 sram_fins = sram_cells * SRAM_CELL.fins;
    assert(total_fins >= sram_fins);
    u64 logic_fins = total_fins - sram_fins;
    // FIXME: this is a very rough model (e.g., ignores stacking and temperature)
    return ((sram_fins/2) * IOFF_SRAM + (logic_fins/2) * IOFF) * VDD * 1e3; // mW
  }


  constexpr f64 wire_res_delay(f64 res/*Ohm*/, f64 wirecap_pF, f64 loadcap_pF=0)
  {
    // approximate distributed RC line as lumped PI1 circuit (Rao, DAC 1995)
    f64 ElmoreDelay = res * (wirecap_pF/2 + loadcap_pF); // ps
    // the Elmore delay is a good approximation when the input voltage varies slowly
    // see Gupta et al., IEEE TCAD jan 1997
    return ElmoreDelay;
  }


  constexpr f64 proba_switch(f64 bias)
  {
    assert(bias>=0 && bias<=1);
    return 2 * bias * (1-bias);
  }


  constexpr f64 proba_bias(f64 proba_switch)
  {
    assert(proba_switch <= 0.5); // random switching (Bernoulli process)
    return 0.5 * (1-mysqrt(1-2*proba_switch)); // does not exceed 1/2
  }


  struct circuit {
    u64 t = 0; // transistors
    f64 d = 0; // delay (ps)
    f64 ci = 0; // maximum input capacitance relative to CGATE
    f64 cg = 0; // total gate capacitance (all transistors) relative to CGATE
    f64 e = 0; // energy (fJ)
    f64 w = 0; // total wiring (um)
    u64 f = 0; // fins

    constexpr circuit() : t(0), d(0), ci(0), cg(0), e(0), w(0), f(0) {}

    constexpr circuit(u64 t, f64 d, f64 ci, f64 cg, u64 f, f64 bias) : t(t), d(d), ci(ci), cg(cg), e(gate_energy_fJ(bias)), w(0), f(f)
    {
      assert(t!=0);
      assert(d>0);
      assert(ci>0);
      assert(cg>0);
      assert(f!=0);
      assert(e>=0);
    }

    constexpr bool nogate() const
    {
      return t==0;
    }

    constexpr u64 delay() const {return myllround(myceil(d));}

    void print(std::string s = "", std::ostream & os = std::cout) const
    {
      os << s;
      os << std::setprecision(4);
      os << "xtors: " << t;
      os << " ; wires (um): " << w;
      os << " ; ps: " << d;
      os << " ; fJ: " << e;
      os << " ; input cap: " << ci;
      //os << " ; gate cap: " << cg;
      os << std::endl;
    }

    constexpr f64 gate_energy_fJ(f64 bias) const
    {
      // neglected: short-circuit currents, and glitches
      // rough approximation of switching capacitance (FIXME?)
      f64 c = (1+PINV) * CGATE_fF * cg; // switching capacitance (fF)
      return proba_switch(bias) * energy_fJ(c,VDD) * 0.5; // fJ
    }

    constexpr f64 cost() const
    {
      return e*d*d;
    }

    constexpr circuit operator+ (const circuit &x) const
    {
      // series (LHS output connected to RHS input)
      circuit chain;
      chain.t = t + x.t;
      chain.d = d + x.d;
      chain.ci = (ci==0)? x.ci : ci;
      chain.cg = cg + x.cg;
      chain.f = f + x.f;
      chain.e = e + x.e;
      chain.w = w + x.w;
      return chain;
    }

    template<typename T>
    constexpr circuit operator* (T n) const
    {
      // parallel, distinct inputs
      if (n==0) return {};
      circuit rep;
      rep.t = t * n;
      rep.d = d;
      rep.ci = ci;
      rep.cg = cg * n;
      rep.f = f * n;
      rep.e = e * n;
      rep.w = w * n;
      return rep;
    }

    constexpr circuit operator|| (const circuit &x) const
    {
      // parallel, distinct inputs
      circuit para;
      para.t = t + x.t;
      para.d = std::max(d,x.d);
      para.ci = std::max(ci,x.ci);
      para.cg = cg + x.cg;
      para.f = f + x.f;
      para.e = e + x.e;
      para.w = w + x.w;
      return para;
    }

    constexpr circuit operator| (const circuit &x) const
    {
      // parallel, single input
      circuit para;
      para.t = t + x.t;
      para.d = std::max(d,x.d);
      para.ci = ci + x.ci;
      para.cg = cg + x.cg;
      para.f = f + x.f;
      para.e = e + x.e;
      para.w = w + x.w;
      return para;
    }
  };


  struct basic_gate {
    using cilist = std::initializer_list<f64>;
    static constexpr u64 MAX_INPUT_TYPES = 3;
    u64 tr = 0; // number of transistors
    f64 ci[MAX_INPUT_TYPES] = {0}; // inputs capacitances, relative to CGATE
    f64 cp = 0; // parasitic output capacitance, relative to CGATE
    f64 cg = 0; // total gate capacitances, relative to CGATE
    u64 nci = 0; // number of input types
    f64 cimax = 0; // maximum input capacitance
    u64 fins = 0; // number of fins

    constexpr void init_ci(cilist l)
    {
      nci = l.size();
      assert(nci!=0);
      assert(nci<=MAX_INPUT_TYPES);
      cimax = std::max(l);
      int i = 0;
      for (auto e : l) ci[i++] = e;
    }

    constexpr basic_gate() {}

    constexpr basic_gate(u64 tr, cilist cl, f64 p, f64 cg, u64 f) : tr(tr), cp(p*PINV), cg(cg), fins(f)
    {
      init_ci(cl);
    }

    template<u64 I=0>
    constexpr f64 icap(f64 scale=1) const
    {
      assert(I<nci);
      return ci[I] * scale;
    }

    template<u64 I=0>
    constexpr f64 logical_effort() const
    {
      assert(I<nci);
      return ci[I] / INVCAP;
    }

    constexpr circuit make(f64 co, f64 scale=1, f64 bias=0.5) const
    {
      //  co = output (load) capacitance relative to CGATE
      assert(scale>=1);
      assert(bias>=0 && bias<=1);
      assert(co>=0);
      f64 delay_ps = (cp + co/scale) * TAU_ps;
      return {tr, delay_ps, cimax*scale, cg*scale, u64(fins*scale), bias};
    }
  };


  // We assume static CMOS implementation for all gates.
  // We do not consider pass-transistor logic (Zimmermann & Fichtner, IEEE JSSC, july 1997)
  // Standard cell libraries use PTL only in a few gates (XOR, MUX)
  // Chinazzo et al., ICECS 2022, "Investigation of pass transistor logic in a 12nm FinFET CMOS technology"

  struct inv : basic_gate { // inverter
    constexpr inv() : basic_gate(2,{INVCAP},INVCAP,INVCAP,2) {}
  };

  struct nand : basic_gate { // single-input NAND is inverter
    constexpr nand(u64 n) : basic_gate(2*n,{n+GAMMA},n+n*GAMMA,n*(n+GAMMA),n+n*n) {assert(n>=1);}
  };

  struct nor : basic_gate { // single-input NOR is inverter
    constexpr nor(u64 n) : basic_gate(2*n,{1+n*GAMMA},n+n*GAMMA,n*(1+n*GAMMA),n+n*n) {assert(n>=1);}
  };

  struct and_nor /*aka AOI21*/: basic_gate { // ~(a+bc)
    constexpr and_nor() : basic_gate(6,{1+2*GAMMA/*a*/,2+2*GAMMA/*b*/,2+2*GAMMA/*c*/},3+2*GAMMA,5+6*GAMMA,11) {}
  };

  struct or_nand /*aka OAI21*/: basic_gate { // ~(a(b+c))
    constexpr or_nand() : basic_gate(6,{2+GAMMA/*a*/,2+2*GAMMA/*b*/,2+2*GAMMA/*c*/},2+3*GAMMA,6+5*GAMMA,11) {}
  };

  struct and2_nor /*aka AOI22*/ : basic_gate { // ~(ab+cd)
    constexpr and2_nor() : basic_gate(8,{INVCAP},4+4*GAMMA,8+8*GAMMA,16) {}
  };

  struct mux_inv_tri : basic_gate { // inverting MUX (tristate inverters)
    constexpr mux_inv_tri(u64 n) : basic_gate(4*n,{2+2*GAMMA/*data*/,2/*sel*/,2*GAMMA/*csel*/},n*(2+2*GAMMA),n*(4+4*GAMMA),n*8) {assert(n>=1);}
  };

  struct inv_tri : mux_inv_tri { // tristate inverter
    constexpr inv_tri() : mux_inv_tri(1) {}
  };

  struct xor_cpl : basic_gate { // a,~a,b,~b ==> a^b
    constexpr xor_cpl() : basic_gate(8,{2+2*GAMMA},4+4*GAMMA,8+8*GAMMA,16) {}
  };

  using xnor_cpl = xor_cpl; // a,~a,b,~b ==> ~(a^b) = (~a)^b

  template<f64 STAGE_EFFORT=DSE>
  constexpr u64 num_stages_inv(f64 path_effort, bool odd=false)
  {
    // higher stage effort increases delay but reduces energy
    assert(path_effort>=1);
    auto delay = [=] (u64 ninv) {
      return (ninv+1) * (PINV+mypow(path_effort,1./(ninv+1)));
    };
    i64 n = std::max(1.,myceil(mylog(path_effort) / mylog(STAGE_EFFORT))) - 1;
    assert(n>=0);
    n += (n & 1) ^ odd;
    u64 nn = n;
    while (nn>=2 && delay(nn-2)<=delay(n)*1.05) {
      nn -= 2;
    }
    return nn;
  }

  template<f64 STAGE_EFFORT=DSE>
  constexpr u64 num_stages(f64 path_effort, bool odd=false)
  {
    // higher stage effort increases delay but reduces energy
    assert(path_effort>=1);
    i64 n = mylog(path_effort) / mylog(STAGE_EFFORT);
    assert(n>=0);
    n += (n & 1) ^ odd;
    return n;
  }

  template<f64 SE=DSE, f64 SMAX=DSMAX>
  constexpr circuit buffer(f64 co, bool cpl, f64 scale=1, f64 bias=0.5)
  {
    assert(scale==1);
    f64 fo = std::max(1.,co/inv{}.icap());
    u64 ninv = num_stages_inv<SE>(fo,cpl);
    f64 beta = mypow(fo,1./(ninv+1));
    assert(beta>0);
    if (fo/beta > SMAX) {
      circuit last = inv{}.make(co,SMAX,bias);
      return buffer<4.,SMAX>(last.ci,cpl^1,scale,bias) + last;
    }
    circuit buf;
    if (ninv!=0) {
      assert(beta>=1);
      for (u64 i=0; i<ninv; i++) {
        scale *= beta;
        f64 ci = inv{}.icap(scale) * beta;
        buf = buf + inv{}.make((i==(ninv-1))? co:ci,scale,bias);
      }
    }
    if (buf.nogate()) buf.ci = co;
    return buf;
  }


  template<f64 SE=DSE, f64 SMAX=DSMAX, u64 L=1/*layer*/>
  constexpr circuit wire(f64 length/*um*/, bool cpl=0, f64 cload=0, bool dload=true, f64 bias=0.5)
  {
    // Unidirectional segmented wire
    // The wire is divided into equal length segments
    // The first segment is driven by a buffer, other segments by large inverters
    // Load capacitance can be terminal (dload=0) or uniformly distributed (dload=1)
    // TODO: bidirectional wire is possible (with tristate inverters), but can it be pipelined?
    static_assert(L<std::size(METALRES));
    assert(length>=0);
    if (length==0) dload = false;
    f64 co = (dload)? 0 : cload; // terminal load capacitance, relative to CGATE
    // for distributed load cap, merge with wire capacitance (FIXME: is this good approx?)
    f64 linearcap = METALCAP + ((dload)? (cload/length) : 0); // CGATE/um
    f64 seglen = mysqrt(2*REFF*INVCAP*(1+PINV) / (linearcap*METALRES[L])); // um
    f64 optinvscale = mysqrt(REFF * linearcap / (INVCAP * METALRES[L]));
    f64 invscale = std::max(1.,std::min(f64(SMAX),optinvscale)); // inverter scale
    u64 nseg = std::max(u64(1), u64(myllround(length/seglen))); // number of segments
    seglen = length / nseg; // segment length (um)
    f64 cseg = linearcap * seglen; // wire segment capacitance relative to CGATE
    // transistors:
    f64 repcap = inv{}.icap() * invscale;
    f64 cfirst = cseg + ((nseg==1)? co : repcap);
    circuit c = buffer<SE,SMAX>(cfirst,(nseg-1+cpl)&1,1,bias);
    circuit rep = inv{}.make(cseg+repcap,invscale,bias); // repeater = inverter
    for (u64 i=1; i<(nseg-1); i++) c = c + rep;
    if (nseg>1) c = c + inv{}.make(cseg+co,invscale,bias); // last segment drives output cap
    // wire:
    c.w = length;
    c.d += wire_res_delay(METALRES[L]*seglen, cseg*CGATE_pF, rep.ci*CGATE_pF) * (nseg-1);
    c.d += wire_res_delay(METALRES[L]*seglen, cseg*CGATE_pF, co*CGATE_pF); // last segment
    c.e += proba_switch(bias) * energy_fJ(METALCAP_fF*length,VDD) * 0.5;
    if (c.nogate()) c.ci = cfirst; // just a wire
    return c;
  }


  template<u64 B=4/*branching*/>
  constexpr circuit broadcast_tree(u64 n, f64 co=INVCAP, bool cpl=false)
  {
    // broadcast one bit to n outputs with a tree of inverters
    // wires not modeled (TODO)
    static_assert(B>=2);
    if (n<=B) {
      if (cpl) {
        return inv{}.make(co*n);
      } else {
        circuit nogate;
        nogate.ci = co*n;
        return nogate;
      }
    }
    u64 m = n / B;
    u64 r = n % B;
    u64 mm = (r==0)? m : m+1;
    circuit stage = inv{}.make(co*B) * m;
    if (r!=0) stage = stage || inv{}.make(co*r);
    return broadcast_tree<B>(mm,stage.ci,cpl^1) + stage;
  }


  constexpr circuit majority(f64 co)
  {
    // a,b,c ==> ab+ac+bc = ~(~(b+c)+(~a~(bc)))
    and_nor aoi;
    f64 scale_aoi = std::max(1.,mysqrt(co/aoi.icap<1>()));
    f64 c1 = aoi.icap<0>(scale_aoi);
    f64 c2 = aoi.icap<1>(scale_aoi);
    circuit i = inv{}.make(c2);
    circuit na = nand{2}.make(c2);
    circuit no = nor{2}.make(c1);
    return ((na | no) || i) + aoi.make(co,scale_aoi); // FIXME: and_nor inputs bias
  }


  template<f64 SE=DSE, f64 SMAX=DSMAX, u64 ARITY=4>
  constexpr circuit nand_nor_tree(u64 n, bool nandfirst, bool cpl, f64 co, f64 scale=1, f64 bias=0.5)
  {
    // alternate NANDs and NORs
    assert(scale==1);
    static_assert(ARITY >= 2);
    if (n==0) return {};
    if (n==1) return (cpl)? inv{}.make(co,scale,bias) : circuit{};
    basic_gate gate[2][ARITY+1];
    for (u64 i=1; i<=ARITY; i++) {
      gate[0][i] = nor(i);
      gate[1][i] = nand(i);
    }
    u64 ngates[2][ARITY+1];

    auto partition = [] (u64 n) {
      u64 w = std::min(n,ARITY);
      u64 r = n%w;
      u64 nw = n/w;
      if (r!=0) {
        r += w;
        nw--;
      }
      return std::tuple{w,nw,r-r/2,r/2};
    };

    auto populate_gates = [partition,&ngates] (bool nand_stage, u64 n) {
      assert(n>0);
      if (n==1) return u64(0);
      auto [w,nw,w2,w3] = partition(n);
      assert(w<=ARITY && w2<=ARITY && w3<=ARITY);
      for (u64 &e : ngates[nand_stage]) e = 0;
      ngates[nand_stage][w] += nw;
      ngates[nand_stage][w2]++;
      ngates[nand_stage][w3]++;
      return (nw==0)? w2 : w; // widest
    };

    auto path_logical_effort = [nandfirst,gate,partition] (u64 n) {
      f64 path_le = 1;
      bool nand_stage = nandfirst;
      while (n>1) {
        auto [w,nw,w2,w3] = partition(n);
        u64 width = (nw==0)? w2 : w;
        assert(width!=0);
        path_le *= gate[nand_stage][width].logical_effort();
        n = nw;
        if (w2) n++;
        if (w3) n++;
        nand_stage ^= 1;
      }
      return path_le;
    };

    bool nand_stage = nandfirst;
    u64 width = populate_gates(nand_stage,n);
    f64 ci = gate[nand_stage][width].icap(scale);
    f64 fanout = co/ci;
    u64 path_effort = std::max(1.,path_logical_effort(n)*fanout);
    u64 depth_target = num_stages<SE>(path_effort);
    u64 depth = myllround(myceil(mylog(n)/mylog(ARITY)));
    bool extra_inv = (depth & 1) ^ cpl; // odd number of stages if cpl=true, even otherwise
    depth += extra_inv;
    u64 ninv = 0;
    if (depth < depth_target) {
      // add inverters
      ninv = depth_target - depth;
      ninv += ninv & 1; // even
      depth += ninv;
    }
    ninv += extra_inv;
    assert(depth!=0);
    f64 stage_effort = mypow(path_effort,1./depth);

    if (path_effort/stage_effort > SMAX) {
      circuit last = inv{}.make(co,SMAX,bias);
      return nand_nor_tree<4.,SMAX>(n,nandfirst,cpl^1,last.ci,scale,bias) + last;
    }

    // build the tree
    circuit tree;
    n = std::accumulate(ngates[nand_stage]+1,ngates[nand_stage]+1+ARITY,0);
    do {
      assert(width>=2);
      u64 prev_width = width;
      width = populate_gates(nand_stage^1,n);
      f64 next_le = (width!=0)? gate[nand_stage^1][width].logical_effort() : 1;
      f64 next_scale = std::max(1.,scale * stage_effort / next_le);
      f64 cload = (width!=0)? gate[nand_stage^1][width].icap(next_scale) : (depth>1)? inv{}.icap(next_scale) : co;
      circuit stage;
      for (u64 w=1; w<=ARITY; w++) {
        stage = stage || (gate[nand_stage][w].make(cload,scale,bias) * ngates[nand_stage][w]);
      }
      tree = tree + stage;
      bias = mypow(bias,prev_width); // each NAND/NOR stage reduces switching probability (neglect glitching)
      depth--;
      scale = next_scale;
      nand_stage ^= 1;
      if (width!=0) n = std::accumulate(ngates[nand_stage]+1,ngates[nand_stage]+1+ARITY,0);
    } while (width!=0);

    if (ninv!=0) {
      // add inverters
      for (u64 i=0; i<ninv; i++) {
        f64 next_scale = scale * stage_effort;
        f64 cload = (depth>1)? inv{}.icap(next_scale) : co;
        tree = tree + inv{}.make(cload,scale,bias);
        depth--;
        scale = next_scale;
      }
    }
    assert(depth==0);
    return tree;
  }


  template<f64 SE=DSE, f64 SMAX=DSMAX>
  constexpr circuit anding(u64 n, f64 co, f64 scale=1, f64 bias=0.5)
  {
    return nand_nor_tree<SE,SMAX>(n,true,false,co,scale,bias);
  }


  template<f64 SE=DSE, f64 SMAX=DSMAX>
  constexpr circuit oring(u64 n, f64 co, f64 scale=1, f64 bias=0.5)
  {
    return nand_nor_tree<SE,SMAX>(n,false,false,co,scale,bias);
  }


  template<f64 SE=DSE, f64 SMAX=DSMAX>
  constexpr circuit nanding(u64 n, f64 co, f64 scale=1, f64 bias=0.5)
  {
    return nand_nor_tree<SE,SMAX>(n,true,true,co,scale,bias);
  }


  template<f64 SE=DSE, f64 SMAX=DSMAX>
  constexpr circuit noring(u64 n, f64 co, f64 scale=1, f64 bias=0.5)
  {
    return nand_nor_tree<SE,SMAX>(n,false,true,co,scale,bias);
  }


  constexpr circuit xor2(f64 co, f64 bias=0.5)
  {
    f64 scale = std::max(1.,mysqrt(co/(xor_cpl{}.icap())));
    circuit x = xor_cpl{}.make(co,scale,bias);
    circuit i = inv{}.make(x.ci,1.,bias);
    circuit c = i*2+x;
    c.ci = i.ci + x.ci;
    return c;
  }


  constexpr circuit xnor2(f64 co, f64 bias=0.5)
  {
    return xor2(co,bias);
  }


  template<f64 SE=DSE, f64 SMAX=DSMAX>
  constexpr circuit decode1(u64 no/*outputs*/, f64 co, f64 pitch=0/*um*/)
  {
    // single-level decoder
    // co = load capacitance per output (relative to CGATE)
    // pitch = distance between outputs (um)
    assert(no!=0);
    if (no==1) return buffer<SE,SMAX>(co,false);
    assert(no>=2);
    u64 ni = std::bit_width(no-1); // inputs (address bits)
    if (ni == 1) {
      return wire<SE,SMAX>(pitch/2,false,co,false) | wire<SE,SMAX>(pitch/2,true,co,false);
    }
    circuit g = anding<SE,SMAX>(ni,co);
    f64 h = pitch * (no-1); // decoder height (um)
    // each input is connected to no/2 gates
    circuit wireup = wire<SE,SMAX>(h/2,false,g.ci*(no/4)) | wire<SE,SMAX>(h/2,true,g.ci*(no/4));
    circuit wiredown = wire<SE,SMAX>(h/2,false,g.ci*(no/2-no/4)) | wire<SE,SMAX>(h/2,true,g.ci*(no/2-no/4));
    circuit dec = (wireup | wiredown) * ni + g * no;
    return dec;
  }


  template<f64 SE=DSE, f64 SMAX=DSMAX>
  constexpr circuit decode2(u64 no/*outputs*/, f64 co, f64 pitch=0/*um*/)
  {
    // two-level decoder
    // co = load capacitance per output (relative to CGATE)
    // pitch = distance between outputs (um)
    if (no<4) return decode1<SE,SMAX>(no,co,pitch);
    u64 ni = std::bit_width(no-1); // inputs (address bits)
    // 2 predecoders
    u64 pni[2] = {ni/2, ni-ni/2}; // address bits per predecoder
    u64 pno[2] = {1ull<<pni[0], 1ull<<pni[1]}; // ouputs per predecoder
    f64 bias[2] = {1./pno[0], 1./pno[1]}; // one output activated per predecoder
    circuit g = anding<SE,SMAX>(2,co,1,(bias[0]+bias[1])/2);
    f64 h = pitch * (no-1); // decoder height (um)
    circuit pdwire[2] = {wire(h/2,false,g.ci*(pno[1]/2),true,bias[0]) | wire(h/2,false,g.ci*(pno[1]-pno[1]/2),true,bias[0]), wire(h/2,false,g.ci*(pno[0]/2),true,bias[1]) | wire(h/2,false,g.ci*(pno[0]-pno[0]/2),true,bias[1])};
    circuit pdec[2] = {decode1(pno[0],pdwire[0].ci), decode1(pno[1],pdwire[1].ci)};
    return ((pdec[0]+(pdwire[0]*pno[0])) || (pdec[1]+(pdwire[1]*pno[1]))) + g * no;
  }


  template<f64 SE=DSE, f64 SMAX=DSMAX>
  constexpr circuit decode2_rep(u64 no/*outputs*/, f64 co, f64 pitch=0/*um*/, u64 rep=1)
  {
    // two-level decoder with replicated outputs
    // rep = number of times an output is replicated
    // pitch = distance between outputs/replicas (um)
    assert(no>=4);
    u64 ni = std::bit_width(no-1); // inputs (address bits)
    // 2 predecoders
    u64 pni[2] = {ni/2, ni-ni/2}; // address bits per predecoder
    u64 pno[2] = {1ull<<pni[0], 1ull<<pni[1]}; // ouputs per predecoder
    f64 bias[2] = {1./pno[0], 1./pno[1]}; // one output activated per predecoder
    circuit g = anding<SE,SMAX>(2,co,1,(bias[0]+bias[1])/2);
    f64 h = pitch * (no*rep-1); // decoder height (um)
    circuit pdwire[2] = {wire(h/2,false,g.ci*(pno[1]/2)*rep,true,bias[0]) | wire(h/2,false,g.ci*(pno[1]-pno[1]/2)*rep,true,bias[0]), wire(h/2,false,g.ci*(pno[0]/2)*rep,true,bias[1]) | wire(h/2,false,g.ci*(pno[0]-pno[0]/2)*rep,true,bias[1])};
    circuit pdec[2] = {decode1(pno[0],pdwire[0].ci), decode1(pno[1],pdwire[1].ci)};
    return ((pdec[0]+(pdwire[0]*pno[0])) || (pdec[1]+(pdwire[1]*pno[1]))) + g * (no*rep);
  }


  template<f64 SE=DSE, f64 SMAX=DSMAX, u64 ARITY=4>
  constexpr auto mux_tree(u64 n/*inputs (data)*/, u64 m/*bits per input*/, f64 co)
  {
    // select input in encoded form
    // made from inverting MUXes ==> number of stages must be even
    // FIXME: distances and wires are not modeled here
    static_assert(ARITY>=2 && std::has_single_bit(ARITY));
    assert(n>=2);
    assert(m>=1);

    auto multiplexer = [] (u64 w, f64 cload) {
      assert(w>0);
      return (w==1)? inv{}.make(cload) : mux_inv_tri{w}.make(cload);
    };

    auto selector = [m] (u64 w, u64 nmuxes) {
      assert(w>1);
      mux_inv_tri muxi{w};
      if (w==2) {
        // single select signal
        f64 selcap = (muxi.icap<1>() + muxi.icap<2>()) * nmuxes * m;
        return buffer<SE,SMAX>(selcap,true) |  buffer<SE,SMAX>(selcap,false);
      } else {
        assert(w>2);
        f64 selcap = muxi.icap<1>() * nmuxes * m;
        f64 cselcap = muxi.icap<2>() * nmuxes * m;
        circuit sel = buffer<SE,SMAX>(selcap,false) | buffer<SE,SMAX>(cselcap,true);
        return decode1<SE,SMAX>(w,sel.ci) + sel * w;
      }
    };

    f64 muxdatacap = mux_inv_tri{2}.icap<0>(); // independent of MUX width
    circuit sel;
    circuit data;
    u64 nstages = 0;

    while (n>1) {
      nstages++;
      if constexpr (ARITY>2) {
        if ((nstages & 1)==0 && (n+1)==ARITY) {
          n++; // last stage = MUX with unused input
        }
      }
      u64 w = ARITY;
      while (w>n) w>>=1;
      u64 nw = n/w;
      u64 r = n%w;
      assert(nw>=1);
      bool last_mux = (nw==1) && (r==0);
      bool last_stage = last_mux && (nstages & 1)==0;
      f64 cload = (last_stage)? co : (last_mux)? inv{}.icap() : muxdatacap;
      circuit sel_stage;
      circuit mux_stage;
      if (std::has_single_bit(r)) {
        // one smaller MUX
        if (r==1) {
          // just an inverter
          mux_stage = mux_stage || inv{}.make(cload);
        } else {
          sel_stage = sel_stage | selector(r,1);
          mux_stage = mux_stage || multiplexer(r,cload);
        }
      } else if (r!=0) {
        // one extra MUX with unused inputs
        nw++;
        r = 0;
      }
      sel_stage = sel_stage | selector(w,nw);
      mux_stage = mux_stage || multiplexer(w,cload) * nw;
      mux_stage = mux_stage * m;
      sel = sel || sel_stage;
      data = data + mux_stage;
      n = nw + ((r!=0)? 1:0);
    }
    if (nstages & 1) {
      data = data + inv{}.make(co);
      nstages++;
    }
    return std::array{sel,data};
  }


  template<f64 SE=DSE, f64 SMAX=DSMAX>
  constexpr auto mux(u64 n/*inputs (data)*/, u64 m/*bits per input*/, f64 co)
  {
    assert(n>=2);
    assert(m>=1);
    auto mux2 = mux_tree<SE,SMAX,2>(n,m,co);
    auto mux4 = mux_tree<SE,SMAX,4>(n,m,co);
    return ((mux4[0]+mux4[1]).cost() < (mux2[0]+mux2[1]).cost())? mux4 : mux2;
  }


  template<f64 SE=DSE, f64 SMAX=DSMAX>
  constexpr circuit grid_demux(u64 nbits, u64 nx, u64 ny, f64 dx/*um*/, f64 dy/*um*/, f64 cnode=0, u64 abits=1)
  {
    // sends an nbits-bit payload to one of nx*ny nodes
    // the nx*ny nodes form a regular grid
    // distance between nodes is dx (horizontal) and dy (vertical)
    // cnode = capacitance (per bit) of the selected node, relative to CGATE
    // binary tree network; 2-way forks implemented with tristate inverters for reducing wire energy
    // the packet includes the payload and the address bits to route itself through the network
    // when the packet passes through a fork, it drops one address bit (the fork-select one)
    // wires that are off the route should not switch
    // TODO: can we prevent glitching with artificial delays, or do we need clocked logic?
    assert(nbits!=0);
    assert(nx!=0 && ny!=0);
    assert(dx>=0 && dy>=0);
    assert(abits!=0);
    if (nx==1 && ny==1) {
      return {};
    }
    u64 packetbits = nbits + abits-1;
    f64 psel = 1./(nx*ny); // a single branch is selected
    assert(psel<=0.5);
    f64 bias = proba_bias(psel); // pseudo bias
    inv_tri itri; // tristate inverter
    f64 selcap = (itri.icap<1>() + itri.icap<2>()) * packetbits; // FIXME: wire not modeled
    circuit select = buffer<SE,SMAX>(selcap,false,1,bias) | buffer<SE,SMAX>(selcap,true,1,bias);
    if ((nx>=2 && dx<=dy) || ny==1) {
      // demux horizontally
      auto packwire = [=](bool cpl) {return wire(dx/2,cpl,cnode,false,proba_bias(psel*0.5));};
      circuit branch = (itri.make(packwire(true).ci,1,bias) + packwire(true)) * packetbits;
      circuit demux = (select + (branch | branch)) * ((nx/2)*ny);
      demux = demux || packwire(false) * ((nx&1) * packetbits);
      return grid_demux(nbits,(nx+1)/2,ny,dx*2,dy,demux.ci,abits+1) + demux;
    } else {
      // demux vertically
      assert(ny>=2);
      auto packwire = [=](bool cpl) {return wire(dy/2,cpl,cnode,false,proba_bias(psel*0.5));};
      circuit branch = (itri.make(packwire(true).ci,1,bias) + packwire(true)) * packetbits;
      circuit demux = (select + (branch | branch)) * (nx*(ny/2));
      demux = demux || packwire(false) * ((ny&1) * packetbits);
      return grid_demux(nbits,nx,(ny+1)/2,dx,dy*2,demux.ci,abits+1) + demux;
    }
  }


  template<f64 SE=DSE, f64 SMAX=DSMAX>
  constexpr auto grid_mux_preselect(u64 nbits, u64 nx, u64 ny, f64 dx/*um*/, f64 dy/*um*/, bool init=1, bool pol=0)
  {
    // each input data (nbits bits) is tagged with a "select" bit, which has been precomputed
    // all inputs but one have the tag bit reset
    // the tag propagates along with the data and is used as select signal for tristate inverters
    // the output tag is the ORing of the two input tags
    // returns two circuits: one for the tag, one for the data
    assert(nbits!=0);
    assert(nx!=0 && ny!=0);
    assert(dx>=0 && dy>=0);
    if (nx==1 && ny==1) {
      circuit data = (pol)? (inv{}.make(INVCAP) * nbits) : circuit{};
      return std::array{circuit{},data};
    }
    f64 psel = 1./(nx*ny); // a single select bit is set
    f64 bias = proba_bias(psel*0.5); // pseudo bias
    mux_inv_tri muxi{2}; // MUX (two tristate inverters)
    inv_tri itri; // tristate inverter
    f64 selcap = muxi.icap<1>() * nbits; // FIXME: wire not modeled
    f64 cselcap = muxi.icap<2>() * nbits; // FIXME: wire not modeled
    circuit locsel = buffer<SE,SMAX>(selcap,false,1,bias) | buffer<SE,SMAX>(cselcap,true,1,bias);
    if (init) {
      init = false;
      auto rest = grid_mux_preselect<SE,SMAX>(nbits,nx,ny,dx,dy,init,pol^1);
      circuit data = itri.make(rest[1].ci) * (nx*ny*nbits) + rest[1];
      circuit sel = locsel * (nx*ny*nbits) | rest[0];
      return std::array{sel,data};
    } else if ((nx>=2 && dx<=dy) || ny==1) {
      // reduce horizontally
      auto rest = grid_mux_preselect<SE,SMAX>(nbits,(nx+1)/2,ny,dx*2,dy,init,pol^1);
      circuit mux2 = muxi.make(rest[1].ci,1,bias);
      circuit mux1 = itri.make(rest[1].ci,1,bias);
      circuit datawire = wire<SE,SMAX>(dx/2,false,muxi.icap<0>(),false,bias);
      circuit data = (datawire * nx + (mux2 * (nx/2) || mux1 * (nx&1))) * (ny*nbits);
      data = data + rest[1];
      circuit tagcomp = nor{2}.make(rest[0].ci,1,bias);
      auto selwire = [=](bool cpl) {return wire<SE,SMAX>(dx/2,cpl,locsel.ci+tagcomp.ci,false,bias);};
      circuit selwires = selwire(true) * (2*(nx/2)*ny) || selwire(false) * ((nx&1)*ny);
      circuit sel = selwires + ((tagcomp * ((nx/2)*ny) + rest[0]) | locsel * (nx*ny));
      return std::array{sel,data};
    } else {
      // reduce vertically
      assert(ny>=2);
      auto rest = grid_mux_preselect<SE,SMAX>(nbits,nx,(ny+1)/2,dx,dy*2,init,pol^1);
      circuit mux2 = muxi.make(rest[1].ci,1,bias);
      circuit mux1 = itri.make(rest[1].ci,1,bias);
      circuit datawire = wire<SE,SMAX>(dy/2,false,muxi.icap<0>(),false,bias);
      circuit data = (datawire * ny + (mux2 * (ny/2) || mux1 * (ny&1))) * (nx*nbits);
      data = data + rest[1];
      circuit tagcomp = nor{2}.make(rest[0].ci,1,bias);
      auto selwire = [=](bool cpl) {return wire<SE,SMAX>(dy/2,cpl,locsel.ci+tagcomp.ci,false,bias);};
      circuit selwires = selwire(true) * (2*nx*(ny/2)) || selwire(false) * ((ny&1)*nx);
      circuit sel = selwires + ((tagcomp * (nx*(ny/2)) + rest[0]) | locsel * (nx*ny));
      return std::array{sel,data};
    }
  }


  template<bool INCR=false>
  constexpr circuit half_adder(f64 co)
  {
    if constexpr (INCR) {
      // a+1
      circuit c = inv{}.make(co);
      c.ci += co;
      return c;
    } else {
       // a+b
      f64 scale = std::max(1.,mysqrt(co/(xor_cpl{}.icap()+nor{2}.icap())));
      circuit x = xor_cpl{}.make(co,scale); // sum (a^b)
      circuit n = nor{2}.make(co,scale); // carry out (ab)
      circuit c = x|n;
      circuit i = inv{}.make(c.ci);
      c = i*2 + c;
      c.ci += x.ci;
      return c;
    }
  }


  template<bool INCR=false>
  constexpr circuit full_adder(f64 co)
  {
    if constexpr (INCR) {
      // a+b+1
      f64 scale = std::max(1.,mysqrt(co/(xnor_cpl{}.icap()+nand{2}.icap())));
      circuit x = xnor_cpl{}.make(co,scale); // sum = a^~b = ~(a^b)
      circuit n = nand{2}.make(co,scale); // carry = a+b = ~(~a~b)
      circuit xn = x|n;
      circuit c = inv{}.make(xn.ci) * 2 + xn;
      c.ci += xnor_cpl{}.icap(scale);
      return c;
    } else {
      // a+b+c
      // x = ~(abc), y=~(a+b+c), z=ab+ac+bc
      // sum = a^b^c = ~(x(y+z))
      // carry = z
      f64 scale_oai = 1;
      f64 scale_aoi = 1;
      f64 a = and_nor{}.icap<1>() * or_nand{}.icap<1>() / (co*co);
      f64 b = co / or_nand{}.icap<1>();
      auto f = [&](f64 x) {return a*mypow(x,4)-b;};
      f64 x0 = mysqrt(mysqrt(b/a));
      f64 x1 = x0 / (1.-1./(4*a*mypow(x0,3)));
      if (x1 >= x0) {
        // always true if co has reasonable value (>=INVCAP)
        assert(f(x0)<=x0 && f(x1)>=x1);
        f64 xm = (x0+x1)/2;
        for (u64 i=0; i<20; i++) {
          if (f(xm)<xm) {
            x0 = xm;
          } else {
            x1 = xm;
          }
        }
        scale_aoi = std::max(1.,xm*xm * or_nand{}.icap<1>() / co);
        scale_oai = std::max(1.,xm);
      }
      circuit oai = or_nand{}.make(co,scale_oai);
      f64 coai1 = or_nand{}.icap<0>(scale_oai);
      f64 coai2 = or_nand{}.icap<1>(scale_oai);
      circuit x = nand{3}.make(coai1);
      circuit y = nor{3}.make(coai2);
      circuit aoi = and_nor{}.make(co+coai2,scale_aoi);
      f64 caoi1 = and_nor{}.icap<0>(scale_aoi);
      f64 caoi2 = and_nor{}.icap<1>(scale_aoi);
      circuit na = nand{2}.make(caoi2);
      circuit no = nor{2}.make(caoi1);
      circuit inva = inv{}.make(caoi2);
      circuit z = ((na | no) || inva) + aoi;
      circuit fa = (x|y|z) + oai;
      return fa;
    }
  }


  template<bool INCR=false, bool CARRYIN=false>
  constexpr circuit adder_ks(u64 n, f64 co)
  {
    // Kogge-Stone adder (radix-2)
    // wire capacitance not modeled (TODO?)
    // see Harris & Sutherland, "Logical effort of carry propagate adders",
    // Asilomar Conference on Signals, Systems & Computers, 2003
    assert(n!=0);
    if (n==1) {
      return (CARRYIN)? full_adder<INCR>(co) : half_adder<INCR>(co);
    }
    u64 depth = std::bit_width(n-1);
    basic_gate G[2] = {or_nand{},and_nor{}}; // inverting generate gate
    basic_gate P[2] = {nor{2},nand{2}}; // inverting propagate gate
    circuit sum = xor2(co) * (n-(CARRYIN^1)); // final sum
    circuit bws = (INCR)? circuit{} : xor2(sum.ci) * n; // bitwise sum
    circuit bwg; // bitwise generate
    circuit bwp; // bitwise propagate
    if constexpr (! INCR) {
      if (n==2) {
        bwg = nand{2}.make(G[0].icap<1>()+sum.ci) * n;
        bwp = nor{2}.make(G[0].icap<1>()+sum.ci) * (n-1);
      } else {
        bwg = nand{2}.make(G[0].icap<0>()+G[0].icap<1>()) * n;
        bwp = nor{2}.make(G[0].icap<1>()+2*P[0].icap()) * (n-1);
      }
    }
    circuit carryout;
    if (depth & 1) {
      carryout.ci = co;
    } else {
      carryout = inv{}.make(co);
    }
    circuit tree = bwp | bwg;
    for (u64 i=0; i<depth; i++) {
      u64 nd = 1<<i; // generate bits already calculated
      u64 ng = n-nd; // G cells
      u64 np = n-std::min(n,2*nd); // P cells
      u64 n2 = n-std::min(n,3*nd); // cells with 2 consumers
      u64 n1 = ng - n2; // cells with single consumer
      u64 np1 = std::min(n1,np); // P cells with single consumer
      if (i < (depth-1)) {
        f64 cog1 = G[(i+1)&1].icap<0>();
        f64 cog2 = G[(i+1)&1].icap<0>() + G[(i+1)&1].icap<1>();
        f64 cop1 = G[(i+1)&1].icap<1>() + P[(i+1)&1].icap();
        f64 cop2 = G[(i+1)&1].icap<1>() + 2 * P[(i+1)&1].icap();
        circuit g = (G[i&1].make(cog1) * n1) || (G[i&1].make(cog2) * (ng-n1));
        circuit p = (P[i&1].make(cop1) * np1) || (P[i&1].make(cop2) * (np-np1));
        tree = tree + (g | p);
      } else {
        // last generate stage
        assert(np==0 && np1==0 && ng==n1);
        f64 cog1 = (n1==1)? carryout.ci : sum.ci;
        tree = tree + G[i&1].make(cog1) * n1;
      }
    }
    circuit adder = (tree | bws) + (sum || carryout); // large input capacitance (12 CGATE)
    return adder;
  }


  constexpr circuit subtract(u64 n, f64 co)
  {
    // A-B = A+~B+1
    circuit c = adder_ks<false,true>(n,co);
    return inv{}.make(c.ci) * n + c;
    // TODO: can we specialize the adder and get rid of the inverter delay?
  }


  constexpr circuit csa_tree(const std::vector<u64> &icount, f64 co, std::vector<f64> &loadcap, u64 obits=0)
  {
    // tree of full adders (FA) and half adders (HA)
    // icount vector = number of bits per column (not necessarily uniform)
    // loadcap = input capacitances of the returned circuit (output parameter)
    // obits is the number of rightmost bits returned (for the remainder operator)
    // wiring not modeled
    if (icount.empty()) return {};
    u64 n = icount.size(); // number of columns
    assert(n!=0);
    u64 cmax = *std::max_element(icount.begin(),icount.end());
    if (cmax<=2) {
      // final CPA
      assert(loadcap.empty());
      circuit output;
      output.ci = co;
      u64 i2 = 0;
      bool need_cpa = false;
      for (u64 i=0; i<=n; i++) {
        if (i==n || icount.at(i)==0) {
          if (need_cpa) {
            circuit cpa = adder_ks(i-i2,co);
            output = output || cpa;
            for (u64 j=i2; j<i; j++) loadcap.push_back(cpa.ci);
            need_cpa = false;
          }
          if (i<n) loadcap.push_back(0);
        } else if (! need_cpa) {
          if (icount.at(i)==1) {
            loadcap.push_back(co);
          } else {
            assert(icount.at(i)==2);
            need_cpa = true;
            i2 = i;
          }
        }
      }
      return output;
    }
    // Wallace tree
    assert(cmax>=3);
    std::vector<u64> n3 (n+1,0);
    std::vector<u64> n2 (n+1,0);
    std::vector<u64> ocount (n+1,0);
    for (u64 i=0; i<n; i++) {
      n3[i] = icount.at(i)/3;
      n2[i] = 0;
      u64 r = icount.at(i)%3;
      if (r==2) {
        n2[i] = 1;
        r = 0;
      }
      ocount.at(i) += n3[i] + n2[i] + r;
      ocount.at(i+1) += n3[i] + n2[i];
    }
    if (ocount.at(n)==0 || n==obits) ocount.pop_back();
    circuit csa = csa_tree(ocount,co,loadcap,obits);
    circuit stage;
    for (u64 i=0; i<n; i++) {
      if (n3[i]==0 && n2[i]==0) continue;
      f64 ocap = (i+1<loadcap.size())? std::max(loadcap[i],loadcap[i+1]) : loadcap[i];
      circuit col = full_adder(ocap) * n3[i] || half_adder(ocap) * n2[i];
      stage = stage || col;
      loadcap[i] = col.ci;
    }
    return stage + csa;
  }


  constexpr circuit carry_save_adder(const std::vector<u64> &icount, f64 co)
  {
    std::vector<f64> loadcap;
    return csa_tree(icount,co,loadcap);
  }


  constexpr circuit multiplier(u64 n, u64 m, f64 co)
  {
    // multiply two unsigned integers X (n bits) and Y (m bits)
    // X is the multiplier, Y the multiplicand
    if (n>m) {
      return multiplier(m,n,co); // faster and more energy efficient
    }
    assert(m>=n && n>=1);
    u64 cols = m+n-1; // number of CSA columns
    std::vector<u64> count (cols,n); // number of bits per CSA column
    for (u64 i=1; i<n; i++) {
      count.at(i-1) = i;
      count.at(cols-i) = i;
    }
    circuit sum = carry_save_adder(count,co); // sum all partial products
    // compute partial products as ~(~a+~b)
    circuit pp1 = nor{2}.make(sum.ci); // partial product (one bit)
    // inputs have high fanout if n or m is large, buffering needed
    circuit bufx = buffer(pp1.ci*m,true); // inverting buffer
    circuit bufy = buffer(pp1.ci*n,true); // inverting buffer
    return (bufx * n || bufy * m) + pp1 * (m*n) + sum;
  }


  template<typename T>
  constexpr circuit multiply_by_constant(const T &N, u64 m, f64 co, u64 obits=0)
  {
    // multiply an m-bit unsigned integer multiplicand by a fixed, known multiplier N
    // if obits!=0, it is the number of rightmost bits returned (for modulo operator)
    assert(N.size()>0);
    u64 cols = m+N.size()-1; // number of CSA columns
    if (obits) cols = std::min(cols,obits);
    std::vector<u64> count (cols,0); // number of bits per CSA column
    for (u64 i=0; i<N.size(); i++) {
      if (N[i])
        for (u64 j=0; j<m; j++)
          if (i+j < cols) count.at(i+j)++;
    }
    std::vector<f64> loadcap;
    circuit sum = csa_tree(count,co,loadcap,obits); // sum all partial products
    // inputs may have high fanout, buffering needed
    circuit buf;
    for (u64 j=0; j<m; j++) {
      f64 icap = 0;
      for (u64 i=0; i<N.size(); i++) {
        if ((i+j) < loadcap.size() && N[i])
          icap += loadcap.at(i+j);
      }
      buf = buf || buffer(icap,false);
    }
    return buf + sum;
  }


  constexpr circuit multiply_add(u64 na, u64 nb, u64 nc, f64 co)
  {
    // computes A+BxC (unsigned integers)
    // na,nb,nc = number of bits of A,B,C
    // see multiplier
    if (nb>nc) {
      return multiply_add(na,nc,nb,co);
    }
    // B is the multiplier, C the multiplicand
    assert(nc>=nb && nb>=1 && na>=1);
    u64 cols = std::max(na,nb+nc-1); // number of CSA columns
    std::vector<u64> count (cols,nb); // number of bits per CSA column
    for (u64 i=1; i<nb; i++) {
      count.at(i-1) = i;
      count.at(nb+nc-1-i) = i;
    }
    for (u64 i=0; i<na; i++) count.at(i)++; // A bits
    circuit sum = carry_save_adder(count,co); // sum A with all partial products
    // compute partial products as ~(~a+~b)
    circuit pp1 = nor{2}.make(sum.ci); // partial product (one bit)
    // inputs have high fanout if nb or nc is large, buffering needed
    circuit bufb = buffer(pp1.ci*nc,true); // inverting buffer
    circuit bufc = buffer(pp1.ci*nb,true); // inverting buffer
    circuit muladd = (bufb * nb || bufc * nc) + pp1 * (nb*nc) + sum;
    muladd.ci = std::max(muladd.ci,sum.ci);
    return muladd;
  }


  template<unaryfunc<f64,circuit> F>
  constexpr circuit reduction(u64 n, F op, f64 co)
  {
    // n = number of inputs
    // op represents associative 2-input operation (circuit = op(load_cap))
    // co = output capacitance
    // FIXME: switching activity depends on operation
    assert(n!=0);
    if (n==1) {
      circuit nothing;
      nothing.ci = co;
      return nothing;
    } else if (n==2) {
      return op(co);
    }
    circuit tree = reduction(n-n/2,op,co);
    return op(tree.ci) * (n/2) + tree;
  }


  template<u64 N, unaryfunc<f64,circuit> F>
  constexpr circuit parallel_prefix(F op, f64 (&loadcap)[N])
  {
    // Kogge-Stone parallel prefix tree
    // op represents associative 2-input operation (circuit = op(load_cap))
    // N = number of columns
    // loadcap = output capacitances
    // FIXME: switching activity depends on operation
    static_assert(N!=0);
    circuit tree;
    for (u64 step=std::bit_floor(N-1); step>=1; step/=2) {
      circuit stage;
      for (u64 i=N-1; i>=step; i--) {
        if (i+step*2 < N) loadcap[i] += loadcap[i+step*2];
        circuit c = op(loadcap[i]);
        stage = stage || c;
        loadcap[i] = c.ci;
      }
      tree = stage + tree;
    }
    for (u64 i=0; i<N-1; i++)
      loadcap[i] += loadcap[i+1];
    tree.ci = *std::max_element(loadcap,loadcap+N);
    return tree;
  }


  template<u64 N>
  constexpr circuit parallel_prefix(u64 data, circuit (*op1)(f64), circuit (*op2)(f64), f64 (&loadcap)[N])
  {
    // Kogge-Stone parallel prefix tree but with alternating polarity:
    // 1st stage uses op1, 2nd stage uses op2, 3rd stage uses op1, 4th stage uses op2, and so on
    // The actual associative operation is A op B = ~(A op1 B) = (~A op2 ~B)
    // For example, op1=NOR implies op2=NAND and op=OR
    // data = number of data bits
    // N = number of columns
    // loadcap = output capacitances
    // FIXME: switching activity depends on operation
    static_assert(N!=0);
    std::array op = {op1,op2};
    u64 stages = std::bit_width(N-1); // number of computation stages
    bool pol = stages & 1; // polarity of the output of the last computation stage (pol=1 means inverted)
    circuit tree;
    if (pol) {
      // odd number of computation stages: end with a stage of inverters
      for (u64 i=0; i<N; i++) {
        circuit c = inv{}.make(loadcap[i]) * data;
        tree = tree || c;
        loadcap[i] = c.ci;
      }
    }
    for (u64 step=std::bit_floor(N-1); step>=1; step/=2) {
      circuit stage;
      for (u64 i=N-1; i>=step; i--) {
        if (i+step*2 < N) loadcap[i] += loadcap[i+step*2];
        circuit c = op[pol](loadcap[i]);
        stage = stage || c;
        loadcap[i] = c.ci;
      }
      // inverters are used to propagate values that are computed earlier than the last stage
      // (FIXME: some inverters are unnecessary)
      for (u64 i=0; i<step; i++) {
        circuit c = inv{}.make(loadcap[i]) * data;
        stage = stage || c;
        loadcap[i] = c.ci;
      }
      tree = stage + tree;
      pol ^= 1;
    }
    assert(!pol);
    for (u64 i=0; i<N-1; i++)
      loadcap[i] += loadcap[i+1];
    tree.ci = *std::max_element(loadcap,loadcap+N);
    return tree;
  }


  constexpr circuit unsigned_greater_than(u64 n, f64 co, bool gte = false)
  {
    // Inputs: two n-bit unsigned integers A=(An,...,A1) and B=(Bn,...,B1)
    // Outputs: G=(A>B), E=(A==B)
    // Divide and conquer: A=XY, B=ZW ==> G = (X>Z) || (X==Z && Y>W)
    // This 2-bit binary operation is associative: (g,e) * (g',e') = (g+eg',ee')
    // Define (G[i:i],E[i:i]) = (Ai~Bi,~(Ai^Bi)) for all i in [1,n]
    // Define, for i>j>k, (G[i:k],E[i:k]) = (G[i:j],E[i:j]) * (G[j-1:k],E[j-1:k])
    // We have: G=G[n:1] and E=E[n:1]
    // Use reduction tree with alternating polarity: (AOI,NAND) at stage 1, (OAI,NOR) at stage 2
    // NB: A>=B is equivalent to ~(A<B) ==> start with inverted polarity
    assert(n!=0);
    basic_gate I[2] = {nor{2},nand{2}}; // initial G stage, depending on polarity
    basic_gate G[2] = {and_nor{},or_nand{}}; // reduction stage G, depending on polarity
    basic_gate E[2] = {nand{2},nor{2}}; // reduction stage E, depending on polarity
    if (n==1) {
      circuit c = inv{}.make(I[gte].icap()) + I[gte].make(co);
      c.ci = std::max(c.ci,I[gte].icap());
      return c;
    }
    assert(n>=2);
    bool pol = gte; // polarity of the inputs (pol=1 means inverted)
    // initial stage
    f64 cgleft = G[pol].icap<0>();
    f64 celeft = G[pol].icap<1>() + E[pol^1].icap();
    f64 cgright = G[pol].icap<2>();
    f64 ceright = E[pol].icap();
    circuit gleft = (inv{}.make(I[gte].icap()) + I[gte].make(cgleft)) * (n/2);
    gleft.ci = std::max(gleft.ci,I[gte].icap());
    circuit gright = (inv{}.make(I[gte].icap()) + I[gte].make(cgright)) * (n/2);
    gright.ci = std::max(gright.ci,I[gte].icap());
    circuit ginv = (inv{}.make(I[gte].icap()) + I[gte].make(inv{}.icap())) * (n%2);
    ginv.ci = std::max(ginv.ci,I[gte].icap());
    circuit eleft = xnor2(celeft) * (n/2); // XOR and XNOR are identical circuits
    circuit eright = xnor2(ceright) * (n/2-1); // the rightmost E circuit is not needed
    circuit einv = xnor2(inv{}.icap()) * (n%2);
    circuit tree = (gleft | eleft) || (gright | eright) || (ginv | einv);
    // reduction tree
    while (n>=2) {
      u64 nops = n/2; // number of reduction operations in this stage
      u64 nextn = n-nops; // value of n for the next stage
      circuit stage;
      if (nextn == 1) {
        // last reduction stage
        f64 ocap = (pol==gte)? inv{}.icap() : co;
        stage = G[pol].make(ocap); // the E circuit is not needed
      } else {
        // next stage is a reduction stage
        f64 cgleft = G[pol^1].icap<0>();
        f64 celeft = G[pol^1].icap<1>() + E[pol^1].icap();
        f64 cgright = G[pol^1].icap<2>();
        f64 ceright = E[pol^1].icap();
        u64 nextnops = nextn/2; // >=1
        u64 nopsfeedright = nextnops; // >=1
        u64 nopsfeedleft = nextnops - (n%2) * (1-nextn%2);
        // when n is odd, transmit the remaining bundle (g,e) through a pair of inverters
        u64 nopsfeedinv = (1-n%2) * (nextn%2);
        u64 ninvfeedleft = nextnops - nopsfeedleft;
        u64 ninvfeedinv = (n%2) * (nextn%2);
        circuit opsfeedleft = (G[pol].make(cgleft) | E[pol].make(celeft)) * nopsfeedleft;
        // the rightmost E circuit is not needed
        circuit opsfeedright = G[pol].make(cgright) * nopsfeedright | E[pol].make(ceright) * (nopsfeedright-1);
        circuit opsfeedinv = (G[pol].make(INVCAP) | E[pol].make(INVCAP)) * nopsfeedinv;
        circuit invfeedleft = (inv{}.make(cgleft) | inv{}.make(celeft)) * ninvfeedleft;
        circuit invfeedinv = inv{}.make(INVCAP) * (2 * ninvfeedinv); // FIXME: unneeded
        stage = opsfeedleft || opsfeedright || opsfeedinv || invfeedleft || invfeedinv;
      }
      tree = tree + stage;
      n = nextn;
      pol ^= 1;
    }
    // pol is the polarity of the outputs of the reduction tree
    if (pol != gte) {
      tree = tree + inv{}.make(co);
    }
    return tree;
  }


  template<u64 N, u64 M>
  constexpr circuit pseudo_rom(const std::array<std::bitset<M>,N> &data, f64 co)
  {
    // N x M-bit ROM
    // emulate a ROM with CMOS logic ==> only for small ROM (TODO: ROM array)
    // TODO: would it be reasonable to minimize logic at compile time (Espresso heuristic?)
    // wiring is ignored (TODO?)
    static_assert(M!=0);
    static_assert(N!=0);
    std::array<u64,M> col1 {}; // number of ones per column
    for (u64 i=0; i<N; i++) {
      for (u64 j=0; j<M; j++) {
        col1[j] += data[i][j];
      }
    }
    circuit cols;
    // each column is implemented as a multi-input OR gate
    std::array<circuit,M> col;
    f64 bias = 1./N; // assume words have equal probability to be read
    for (u64 j=0; j<M; j++) {
      col[j] = oring(col1[j],co,1,bias);
      cols = cols || col[j];
    }
    f64 maxloadcap = 0;
    for (u64 i=0; i<N; i++) {
      f64 loadcap = 0;
      for (u64 j=0; j<M; j++) {
        loadcap += col[j].ci * data[i][j];
      }
      maxloadcap = std::max(maxloadcap,loadcap);
    }
    circuit dec = decode2(N,maxloadcap);
    circuit buf = buffer(dec.ci,false) * std::bit_width(N-1);
    return buf + dec + cols;
  }


  template<u64 N, u64 D>
  constexpr circuit divide_by_constant(f64 co)
  {
    // quotient of the Euclidean division of N-bit unsigned integer by fixed, known unsigned divisor D
    static_assert(D!=0);
    constexpr u64 R = 7; // log2 ROM size
    if constexpr (D==1) {
      return {};
    } else if constexpr (N < std::bit_width(D)) {
      return {};
    } else if constexpr ((D&1)==0) {
      // even divisor
      // x div I*J = (x div I) div J
      constexpr u64 factor2 = std::countr_zero(D);
      constexpr u64 Dodd = D >> factor2;
      static_assert(N>=factor2);
      return divide_by_constant<N-factor2,Dodd>(co);
    } else if constexpr (N<=R) {
      constexpr u64 romsize = u64(1) << N;
      constexpr u64 OBITS = N - std::bit_width(D) + 1; // output bits
      std::array<std::bitset<OBITS>,romsize> data;
      for (u64 i=0; i<romsize; i++) {
        data.at(i) = i / D;
      }
      return pseudo_rom<romsize,OBITS>(data,co);
    } else {
      // odd divisor
      static_assert((D&1) && D>=3);
      // L = ceil(log2(D-1))
      // x div D = (ceil(2^(N+L)/D) * x) div 2^(N+L)
      constexpr u64 M = std::bit_width(D-2);
      std::array<bool,N+M> INVD {};
      fractional<N+M>(1,D,INVD); // INVD = 2^(N+M) / D
      std::array<bool,N+M+1> IDP1 {}; // INVD+1
      std::copy(INVD.begin(),INVD.end(),IDP1.begin());
      for (u64 i=0; i<IDP1.size(); i++) {
        IDP1[i] ^= 1;
        if (IDP1[i]) break;
      }
      // TODO: full multiplier not needed (N+M rightmost result bits ignored)
      return multiply_by_constant(IDP1,N,co);
    }
  }


  constexpr circuit circular_csa(u64 n, u64 m, f64 co)
  {
    // reduce a sum S of m n-bit digits to an n+1 bit number Y such that Y mod D = S mod D
    // ASSUMPTION: (2^n-1) mod D = 0
    // see Figure 2 in MICRO 2014 paper by Diamond et al. ("Arbitrary Modulus Indexing")
    assert(n!=0);
    assert(m>=2);
    if (m == 2) {
      // two digits, final CPA
      return adder_ks(n,co);
    }
    // Wallace tree
    // 2^n mod D = 1 ==> weight 2^n is equivalent to weight 1
    // the leftmost carry feeds the rightmost column
    // uniform number of bits per column ==> use full adders only (3:2 compression)
    u64 mm = (m/3)*2 + m%3; // stage reduces from m to mm digits
    circuit next = circular_csa(n,mm,co);
    circuit stage = full_adder(next.ci) * (mm * n);
    return stage + next;
  }


  template<u64 N, u64 D>
  constexpr circuit remainder_divide_by_constant(f64 co)
  {
    // remainder of the Euclidean division of N-bit unsigned integer by fixed, known unsigned divisor D
    // use digital root method when there exists a small K>0 such that 2^K = +/-1 mod D
    static_assert(D!=0);
    static_assert(N!=0);
    constexpr u64 R = 8; // log2 ROM size
    constexpr u64 MAXKM = 12; // digital root method (2^K-1 mod D = 0), max bits per digit
    constexpr u64 MAXKP = 6; // digital root method (2^K+1 mod D = 0), max bits per digit
    constexpr u64 OBITS = std::bit_width(D-1); // number of output bits

    if constexpr (D==1) {
      return {};
    } else if constexpr (N < std::bit_width(D)) {
      return {};
    } else if constexpr ((D&1)==0) {
      // even divisor
      // x mod I*J = I*[(x div I) mod J] + (x mod I)
      constexpr u64 factor2 = std::countr_zero(D);
      static_assert(factor2!=0);
      constexpr u64 Dodd = D >> factor2;
      static_assert(N>=factor2);
      return remainder_divide_by_constant<N-factor2,Dodd>(co);
    } else if constexpr (N<=64 && D == (u64(1)<<N)-1) {
      // D = 2^N-1
      // if X=2^N-1 final=0 else final=X
      circuit out = anding(2,co) * N;
      circuit nand = nanding(N,out.ci);
      circuit c = nand + out;
      c.ci += out.ci;
      return c;
    } else if constexpr (N == std::bit_width(D)) {
      // subtract D from the dividend, use sign of result as MUX select
      auto m = mux(2,N,co);
      circuit minusD = adder_ks<true,true>(N,m[0].ci+m[1].ci);
      circuit c = minusD + m[0] + m[1];
      c.ci += m[1].ci;
      return c;
    } else if constexpr (N<=64 && D == (u64(1)<<(N-1))-1) {
      // D = 2^(N-1)-1
      // final = (X[N-2:0] + X[N-1]) mod 2^(N-1)-1
      // Y = X[N-2:0] + X[N-1]
      // if Y[N-1]=1 final=1 else final=Y[N-2:0] mod 2^(N-1)-1
      static_assert(N>2);
      circuit out = nor{2}.make(co) * (N-2);
      circuit rout = oring(2,co); // rightmost bit
      out = (inv{}.make(out.ci) + out) || rout;
      circuit buf = buffer(nor{2}.icap()*(N-2),false); // drive Y[N-1]
      circuit reduc = remainder_divide_by_constant<N-1,D>(rout.ci);
      circuit sum = adder_ks<true,true>(N-1,std::max(reduc.ci,buf.ci));
      return sum + (buf | reduc) + out;
    } else if constexpr (constexpr u64 K = pow2_minus1(D,MAXKM); K!=0 && K+1<N) {
      // digital rooth method
      // reduce to K+1 bits with circular CSA
      circuit reduc = remainder_divide_by_constant<K+1,D>(co);
      u64 nd = (N+K-1) / K; // number of K-bit digits
      circuit csa = circular_csa(K,nd,reduc.ci);
      return csa + reduc; // no buffering (TODO?)
    } else if constexpr (N<=R) {
      // use a ROM
      constexpr u64 romsize = u64(1) << N;
      std::array<std::bitset<OBITS>,romsize> data;
      for (u64 i=0; i<romsize; i++) {
        data.at(i) = i % D;
      }
      return pseudo_rom<romsize,OBITS>(data,co);
    }

    // digital root with 2^K+1 mod D = 0
    // TODO: Diamond et al., MICRO 2014 (use 1,0,-1 redundant format)
    constexpr u64 K = pow2_plus1(D,MAXKP);
    constexpr u64 ND = (K)? (N+K-1)/K : 0; // number of digits
    static_assert(K==0 || ND!=0);
    constexpr u64 NS = ND+1; // number of summands
    constexpr u64 NN = K + std::bit_width(NS); // reduce input to NN bits
    constexpr std::bitset<OBITS> EXTRA = (ND^(ND&1)) % D; // extra summand

    if constexpr (K!=0 && (NN<=R || NN+1<N)) {
      // digital root with alternating sign
      circuit reduc = remainder_divide_by_constant<NN,D>(co);
      constexpr u64 ncols = std::max(K,OBITS);
      std::vector<u64> count (ncols,0);
      for (u64 i=0; i<K; i++)
        count.at(i) = (N%K==0 || i<N%K)? ND : ND-1;
      for (u64 i=0; i<OBITS; i++)
        count.at(i) += EXTRA[i]; // extra summand
      std::vector<f64> loadcap;
      circuit csa = csa_tree(count,reduc.ci,loadcap);
      assert(loadcap.size()>=K);
      // every other digit is complemented
      circuit bufs;
      for (u64 i=0; i<N; i++)
        bufs = bufs || buffer(loadcap[i%K],(i/K)&1);
      return bufs + csa + reduc;
    } else {
      // odd divisor, general method (inefficient)
      assert(D>=3);
      // L = ceil(log2(D-1))
      // x mod D = {[(ceil(2^(N+L)/D) * x) mod 2^(N+L)] * D} div 2^(N+L)
      constexpr u64 M = std::bit_width(D-2);
      std::array<bool,N+M> INVD {};
      fractional<N+M>(1,D,INVD); // INVD = 2^(N+M) / D
      std::array<bool,N+M+1> IDP1 {}; // INVD+1
      std::copy(INVD.begin(),INVD.end(),IDP1.begin());
      for (u64 i=0; i<IDP1.size(); i++) {
        IDP1[i] ^= 1;
        if (IDP1[i]) break;
      }
      constexpr std::bitset<std::bit_width(D)> Dbits {D};
      circuit mulD = multiply_by_constant(Dbits,N+M,co); // TODO: N+M rightmost result bits ignored
      circuit divfrac = multiply_by_constant(IDP1,N,mulD.ci,N+M); // N+M fractional bits of the division
      return divfrac + mulD;
    }
  }


  // ######################################################

  template<u64 N>
  struct flipflops {
    static constexpr circuit oneflop = []() {
      // master-slave flip-flop (2 latches)
      inv i;
      inv_tri t;
      f64 co = INVCAP; // we do not care about the delay anyway
      f64 cfeedback1 = i.icap() + t.cp;
      f64 cfeedback2 = 2*i.icap() + t.cp;
      circuit tri1 = t.make(cfeedback1);
      circuit tri2 = t.make(cfeedback2);
      circuit forward = tri1 + i.make(2*t.icap()) + tri2 + i.make(co);
      circuit backward = tri1 || (i.make(t.icap()) + tri2);
      return forward || backward;
    } ();

    static constexpr circuit flops = oneflop * N;

    // master-latch clock generated locally at each flop
    static constexpr circuit clock1 = inv{}.make(inv_tri{}.icap<1>()*4);

    static constexpr f64 width = SRAM_CELL.wordline_length * N; // um
    static constexpr f64 height = SRAM_CELL.bitline_length * (oneflop.f+clock1.f) / SRAM_CELL.fins; // um

    static constexpr circuit clocking = []() {
      f64 phicap2 = inv_tri{}.icap<2>() * 4;
      circuit clock2 = wire<4.,DSMAX,0>(width,false,(phicap2+clock1.ci)*N); // slave-latch clock
      return clock2 + clock1 * N;
    } ();

    static constexpr u64 xtors = flops.t + clocking.t;
    static constexpr u64 fins = flops.f + clocking.f;

    // clocking.e corresponds to a transition probability of 0.5
    // actually, the clock transitions with probability 1 twice per write
    static constexpr f64 write_energy_fJ = flops.e + clocking.e * 4;
  };


  // ######################################################
  // SRAM

  template<typename T>
  struct sram_common {
    static constexpr u64 NBITS = T::num_bits();
    static constexpr u64 XTORS = T::num_xtors();
    static constexpr u64 FINS = T::num_fins();
    static constexpr f64 LATENCY = T::read_latency(); // ps
    static constexpr f64 EREAD = T::read_energy(); // fJ
    static constexpr f64 EWRITE = T::write_energy(); // fJ
    static constexpr f64 WIDTH = T::array_width(); // um
    static constexpr f64 HEIGHT = T::array_height(); // um

    static constexpr f64 area_um2()
    {
      return WIDTH * HEIGHT;
    }

    static constexpr f64 area_mm2()
    {
      return area_um2() / 1000000;
    }

    static constexpr f64 leakage_mW()
    {
      return leakage_power_mW(FINS,NBITS);
    }

    // the following cost function is arbitrary (prioritizes latency over energy and reads over writes)
    // ignores write latency and leakage power (TODO?)
    static constexpr f64 COST = (2*EREAD+EWRITE) * mypow(LATENCY,3);

    static void print(std::string s = "", std::ostream & os = std::cout)
    {
      os << s;
      os << std::setprecision(4);
      os << "bits: " << NBITS;
      os << " ; xtors: " << XTORS;
      os << " ; ps: " << LATENCY;
      os << " ;  fJ R|W: " << EREAD << "|" << EWRITE;
      os << " ;  um W|H: " << WIDTH << "|" << HEIGHT;
      os << std::endl;
    }
  };


  struct sram_null {
    static constexpr u64 num_bits() {return 0;}
    static constexpr u64 num_xtors() {return 0;}
    static constexpr u64 num_fins() {return 0;}
    static constexpr f64 read_latency() { return 0;}
    static constexpr f64 read_energy() {return 0;}
    static constexpr f64 write_energy() {return 0;}
    static constexpr f64 array_width() {return 0;}
    static constexpr f64 array_height() {return 0;}
  };


  template<u64 N, u64 M, u64 D>
  struct sram_bank {};


  template<u64 N, u64 M>
  struct sram_bank<N,M,0> : sram_common<sram_null>
  {
    static constexpr circuit ABUS {};
    static constexpr circuit WBUS {};
  };


  template<u64 N, u64 M, u64 D> requires (D<=M)
  struct sram_bank<N,M,D> : sram_common<sram_bank<N,M,D>> {
    // single R/W port, N wordlines, M cells per wordline
    // D = data width, M multiple of D, M/D power of 2
    static_assert(N!=0 && D!=0);
    static_assert(M%D==0);
    static_assert(std::has_single_bit(M/D));

    // TODO: precharge circuit
    // TODO: double wordline, flying bitline (Chang, IEEE JSSC, jan 2021)

    static constexpr f64 WLCAP_fF = SRAM_CELL.wordline_capacitance;
    static constexpr f64 WLCAP = WLCAP_fF / CGATE_fF;
    static constexpr f64 WLCAP_pF = WLCAP * CGATE_pF;
    static constexpr f64 BLCAP_fF = SRAM_CELL.bitline_capacitance;
    static constexpr f64 BLCAP = BLCAP_fF / CGATE_fF;
    static constexpr f64 BLCAP_pF = BLCAP * CGATE_pF;
    static constexpr f64 WLRES = SRAM_CELL.wordline_resistance; // Ohm
    static constexpr f64 BLRES = SRAM_CELL.bitline_resistance; // Ohm
    static constexpr f64 WWCAP = SRAM_CELL.wordline_length * METALCAP;
    static constexpr f64 PERI_LOGIC_DENSITY = SRAM_CELL.density / 3;

    // sense amplifier (SA) = latch type (cross coupled inverters)
    // SAVBLMIN value taken from Amrutur & Horowitz, IEEE JSSC, feb. 2000
    // TODO: SACAPMAX is currently a random number
    static constexpr f64 SACAPMAX = 20*INVCAP; // maximum SA input capacitance relative to CGATE
    static constexpr f64 SACAPMIN = INVCAP * (1+PINV); // minimum SA input capacitance relative to CGATE
    static_assert(SACAPMIN<=SACAPMAX);
    static constexpr f64 SAVBLMIN = 0.1; // minimum bitline voltage swing (V)
    // SA intrinsic delay
    static constexpr f64 SADELAY = 2 * inv{}.make(4*INVCAP).d; // 2 FO4 (Amrutur & Horowitz)
    static constexpr f64 XCSA = 0.4; // SA input capacitance relative to bitline capacitance

    // assumption: SA offset voltage (and bitline swing) inversely proportional to sqrt of SA input capacitance Csa (Kim, IEEE JSSC april 2023)
    static constexpr f64 SACAP = std::max(SACAPMIN,std::min(SACAPMAX,N*BLCAP*XCSA)); // sense amp capacitance relative to CGATE
    static constexpr f64 SACAP_pF = SACAP * CGATE_pF;
    static constexpr f64 SASCALE = SACAP/SACAPMIN;
    static_assert(SASCALE>=1);

    static constexpr f64 BLSWING = SAVBLMIN * mysqrt(SACAPMAX/SACAP); // bitline voltage swing (V)

    // drive wordline from the mid point (Amrutur & Horowitz)
    static constexpr f64 WLRC = wire_res_delay((M/2)*WLRES, (M/2)*WLCAP_pF);

    // BLRC is minimized when XCSA = 1
    // however, if Csa=Cbl, SA consumes more energy than bitline
    // sqrt(x)+1/sqrt(x) is ~10% above min delay at x = Csa/Cbl = 0.4
    static constexpr f64 BLRC = []() { // bitline RC delay (ps)
      // https://files.inria.fr/pacap/michaud/rc_delay.pdf
      // V = Vdd - Idsat * (t-t0) / (Cbl+Csa)
      // t0 = Rbl*Cbl*(Cbl+3Csa)/(6Cbl+6Csa)
      f64 CBL = N * BLCAP_pF; // fF
      f64 RBL = N * BLRES; // Ohm
      f64 T0 = RBL * CBL * (CBL+3*SACAP_pF) / (6*(CBL+SACAP_pF)); // ps
      return T0;
    } ();

    static constexpr f64 BLDELAY = []() { // bitline total delay (ps)
      // model cell drive as current source (Amrutur & Horowitz)
      // assume current is Idsat
      // neglect resistance of sense-amp isolation transistor
      f64 CBL = N * BLCAP_pF; // fF
      return BLRC + (CBL+SACAP_pF) * BLSWING / IDSAT_SRAM;
    } ();

    // gates layout in peripheric logic is constrained by the wordline/bitline pitch
    // not sure to what extent this limits the gate size (TODO?)
    static constexpr f64 SX = 10; // maximum inverter scale at bitline pitch
    static constexpr f64 SY = 10; // maximum inverter scale at wordline pitch
    static constexpr f64 SEFF = 4;

    static constexpr circuit RDEC = decode2<SEFF,SY>(N,M*WLCAP,SRAM_CELL.bitline_length); // row decoder

    // TODO: SA inverters are skewed
    // TODO: SA footer transistor (big capacitance)
    static constexpr circuit SA = inv{}.make(SACAP,SASCALE) * 2 * M; // sense amplifiers

    // column read MUX (if D<M) after SA (wire capacitances not modeled, TODO)
    static constexpr auto CMUX = (D<M)? mux<SEFF,SX>(M/D,D,INVCAP) : std::array<circuit,2>{};

    static constexpr auto WDR = []() { // write driver
      // TODO(?): MY wires resistance delay
      std::array<circuit,2> wdr;
      circuit BLBUF = buffer<SEFF,SX>(N*BLCAP,false) | buffer<SEFF,SX>(N*BLCAP,true);
      // assume last inverter of buffer is tristate (impact on delay not modeled, TODO)
      if constexpr (D<M) {
        // data bits are interleaved
        // separate column decoder drives tristates
        constexpr u64 DEMUX = M/D;
        assert(DEMUX >= 2);
        f64 CTRI = 2 * N*BLCAP * TAU_ps / BLBUF.d; // input cap of tristate select (FIXME)
        if constexpr (DEMUX <= 16) {
          // decode outputs run parallel to wordlines
          f64 CSEL = CTRI*D + WWCAP*M;
          circuit BUFTRI = buffer(CSEL,false) | buffer(CSEL,true);
          circuit CDEC = decode2(DEMUX,BUFTRI.ci);
          wdr[0] = CDEC + BUFTRI * DEMUX;
          wdr[0].e += energy_fJ(WWCAP*M*CGATE_fF,VDD); // two wires switch (worst case)
        } else {
          // predecode outputs run parallel to wordlines, AND2 gates are replicated (D replicas)
          circuit BUFTRI = buffer<SEFF>(CTRI,false) | buffer<SEFF>(CTRI,true);
          circuit CDEC = decode2_rep(DEMUX, BUFTRI.ci, SRAM_CELL.wordline_length, D);
          wdr[0] = CDEC + BUFTRI * M;
        }
        wdr[1] = buffer((BLBUF.ci+WWCAP)*DEMUX,false);
        wdr[1].e += 0.5 * energy_fJ(WWCAP*DEMUX*CGATE_fF,VDD) * 0.5/*switch proba*/;
      }
      wdr[1] = wdr[1] * D + BLBUF * M;
      return wdr;
    } ();

    static constexpr f64 COLSELCAP = CMUX[0].ci + WDR[0].ci;
    static constexpr circuit ABUS = (buffer(RDEC.ci,false) * std::bit_width(N-1)) || (buffer(COLSELCAP,false) * std::bit_width(M/D-1));
    static constexpr const circuit &WBUS = WDR[1];
    static constexpr circuit WLSEL = ABUS + RDEC;

    static constexpr f64 EWCL = D * 2 * inv{}.make(INVCAP).e; // cell energy per write
    static constexpr f64 EWL = energy_fJ(WLCAP_fF*M,VDD); // one wordline switches on and off
    // currently, assume perfect bitline voltage clamping (TODO?)
    // half-selected bitlines (D<M) consume energy on read (neglect on write, TODO?)
    static constexpr f64 EBLR = M * energy_fJ(N*BLCAP_fF,BLSWING); // bitline read + precharge
    static constexpr f64 EBLW = D * energy_fJ(N*BLCAP_fF,VDD); // bitline write + precharge

    static constexpr u64 CELL_XTORS = SRAM_CELL.transistors * N * M;
    static constexpr u64 CELL_FINS = SRAM_CELL.fins * N * M;
    static constexpr u64 PERI_XTORS = WLSEL.t + SA.t + WDR[0].t + WDR[1].t + CMUX[0].t + CMUX[1].t;
    static constexpr u64 PERI_FINS = WLSEL.f + SA.f + WDR[0].f + WDR[1].f + CMUX[0].f + CMUX[1].f;
    static constexpr f64 CELLS_AREA = SRAM_CELL.area * N * M; // um^2
    static constexpr f64 PERI_AREA = f64(PERI_FINS) / PERI_LOGIC_DENSITY; // um^2
    static constexpr f64 AREA = CELLS_AREA + PERI_AREA; // um^2
    static constexpr f64 CELLS_WIDTH = SRAM_CELL.wordline_length * M; // um
    static constexpr f64 CELLS_HEIGHT = SRAM_CELL.bitline_length * N; // um
    // assume that the overall aspect ratio equals the cells aspect ratio
    static constexpr f64 ASPECT_RATIO = CELLS_WIDTH / CELLS_HEIGHT;

    static constexpr u64 num_bits() {return N * M;}
    static constexpr f64 array_width() {return mysqrt(AREA*ASPECT_RATIO);} // um
    static constexpr f64 array_height() {return mysqrt(AREA/ASPECT_RATIO);} // um
    static constexpr u64 num_xtors() {return CELL_XTORS + PERI_XTORS;}
    static constexpr u64 num_fins() {return CELL_FINS + PERI_FINS;}

    static constexpr f64 read_latency()
    {
      return std::max(WLSEL.d + WLRC + BLDELAY + SADELAY, CMUX[0].d) + CMUX[1].d; // ps
    }

    static constexpr f64 read_energy()
    {
      return WLSEL.e + EWL + EBLR + SA.e + CMUX[0].e + CMUX[1].e; // fJ
    }

    static constexpr f64 write_energy()
    {
      return WLSEL.e + EWL + WDR[0].e + WDR[1].e + EBLW + EWCL; // fJ
    }

    static void print2(std::string s = "", std::ostream & os = std::cout)
    {
      os << s;
      WLSEL.print("wordline select: ",os);
      //SA.print("sense amps: ",os);
      if constexpr (D<M) {
        CMUX[0].print("column mux select: ",os);
        CMUX[1].print("column mux data: ",os);
      }
      WDR[0].print("write driver select: ",os);
      WDR[1].print("write driver data: ",os);
      os << std::setprecision(4);
      os << "bitline read voltage swing (V): " << BLSWING << std::endl;
      os << "wordline RC delay (ps): " << WLRC << std::endl;
      os << "bitline RC delay (ps): " << BLRC << std::endl;
      os << "bitline total delay (ps): " << BLDELAY << std::endl;
      os << "wordline energy (fJ): " << EWL << std::endl;
      os << "bitlines energy R/W (fJ): " << EBLR << " / " << EBLW << std::endl;
      os << "cells write energy: " << EWCL << std::endl;
      os << "sense amp scale: " << SASCALE << std::endl;
      os << "sense amp delay (ps): " << SADELAY << std::endl;
      os << "sense amps energy (fJ): " << SA.e << std::endl;
      os << "cell transistors: " << CELL_XTORS << std::endl;
      os << "periphery transistors: " << PERI_XTORS << std::endl;
      //os << "width (um): " << sram_bank::WIDTH << std::endl;
      //os << "height (um): " << sram_bank::HEIGHT << std::endl;
      os << "array efficiency: " << CELLS_AREA/AREA << std::endl;
    }

  };


  template<u64 N, u64 M, u64 D> requires (D>M)
  struct sram_bank<N,M,D> : sram_common<sram_bank<N,M,D>> {
    // single R/W port, N wordlines, M cells per wordline
    // split data over several banks, access all banks in parallel
    static_assert(N!=0 && M!=0);
    static constexpr u64 NB = D/M; // number of M-wide banks
    static constexpr u64 R = D%M; // extra bank provides R remaining bits
    using BANK = sram_bank<N,M,M>;
    using BANKR = sram_bank<N,R,R>;

    // send address to all banks
    static constexpr u64 ADDRESS_BITS = std::bit_width(N-1);
    static constexpr f64 LENGTH = BANK::WIDTH * NB + BANKR::WIDTH - 0.5 * (BANK::WIDTH + BANKR::WIDTH);
    static constexpr f64 LOADCAP = BANK::ABUS.ci * NB + BANKR::ABUS.ci;
    static constexpr circuit AWIRE = wire(LENGTH/2,false,LOADCAP,true);
    static constexpr circuit ABUS = (AWIRE | AWIRE) * ADDRESS_BITS; // address bus
    static constexpr circuit WBUS = BANK::WBUS * NB || BANKR::WBUS;

    static constexpr u64 num_bits() {return BANK::NBITS * NB + BANKR::NBITS;}
    static constexpr f64 read_latency() { return ABUS.d + BANK::LATENCY;}
    static constexpr f64 read_energy() {return ABUS.e + BANK::EREAD * NB + BANKR::EREAD;}
    static constexpr f64 write_energy() {return ABUS.e + BANK::EWRITE * NB + BANKR::EWRITE;}
    static constexpr f64 array_width() {return BANK::WIDTH * NB + BANKR::WIDTH;}
    static constexpr f64 array_height() {return BANK::HEIGHT;}
    static constexpr u64 num_xtors() {return BANK::XTORS * NB + BANKR::XTORS + ABUS.t;}
    static constexpr u64 num_fins() {return BANK::FINS * NB + BANKR::FINS + ABUS.f;}

    static void print2(std::string s = "", std::ostream & os = std::cout)
    {
      os << s;
      ABUS.print("ABUS: ",os);
      BANK::print("BANK: ",os);
      if constexpr (R!=0) {
        BANKR::print("BANKR: ",os);
      }
    }
  };


  template<u64 BX, u64 BY, u64 N, u64 M, u64 D>
  struct sram_array : sram_common<sram_array<BX,BY,N,M,D>> {
    // BX x BY banks
    // Accessed data (D bits) lies in single bank
    // Bank is selected with XY-decoder before being accessed
    // Assume non-selected banks do not consume energy (TODO?)
    // Assume unidirectional interconnects (separate read/write data wires)
    // Area of inter-bank wiring and logic is not modeled (TODO?)
    static_assert(N!=0 && M!=0 && D!=0);
    using BANK = sram_bank<N,M,D>;
    static constexpr u64 NB = BX * BY; // number of banks
    static_assert(NB!=0);
    // N must be a power of 2 unless there is a single bank
    static_assert(NB==1 || std::has_single_bit(N));
    // BY must be a power of 2 unless BX=1
    static_assert(BX==1 || std::has_single_bit(BY));
    static constexpr f64 HW = BANK::WIDTH * ((BX-1)/2); // half width
    static constexpr f64 HH = BANK::HEIGHT * ((BY-1)/2); // half height
    static constexpr circuit SELG = (BX==1 || BY==1)? circuit{} : anding(2,INVCAP,1,(1./BX+1./BY)/2);
    static constexpr circuit XSEL = wire(HW,false,SELG.ci*(BY/2),true,1./BX) | wire(HW,false,SELG.ci*(BY-BY/2),true,1./BX);
    static constexpr circuit YSEL = wire(HH,false,SELG.ci*(BX/2),true,1./BY) | wire(HH,false,SELG.ci*(BX-BX/2),true,1./BY);
    static constexpr circuit XDEC = (BX==1)? circuit{} : decode2(BX,YSEL.ci,BANK::WIDTH) + YSEL * BX;
    static constexpr circuit YDEC = (BY==1)? circuit{} : decode2(BY,XSEL.ci,BANK::HEIGHT) + XSEL * BY;
    static constexpr circuit SEL = (XDEC || YDEC) + SELG * (BX*BY); // bank select
    static constexpr u64 ABITS = (N>=2)? std::bit_width(N-1) : 1; // local (bank) address bits
    static constexpr auto ETCW = [] (f64 co){ // edge-to-center wire
      if (BX==1 || BY==1) {
        circuit c;
        c.ci = co;
        return c;
      } else {
        f64 w = BANK::WIDTH * BX;
        f64 h = BANK::HEIGHT * BY;
        return wire(std::min(w,h)/2,false,co);
      }
    };
    static constexpr circuit ATREE = grid_demux(ABITS,BX,BY,BANK::WIDTH,BANK::HEIGHT,BANK::ABUS.ci); // address tree
    static constexpr circuit WTREE = grid_demux(D,BX,BY,BANK::WIDTH,BANK::HEIGHT,BANK::WBUS.ci); // data write tree
    static constexpr auto RTREE = grid_mux_preselect(D,BX,BY,BANK::WIDTH,BANK::HEIGHT); // read TREE
    static constexpr circuit ANET = ETCW(ATREE.ci) * ABITS + ATREE;
    static constexpr circuit WNET = ETCW(WTREE.ci) * D + WTREE;
    static constexpr circuit ACC = SEL || ANET;
    static constexpr circuit READ = []() {
      circuit bankread;
      bankread.d = BANK::LATENCY;
      bankread.e = BANK::EREAD;
      // bank access starts after bank select signal has been broadcast
      return ACC + (RTREE[0] || (bankread + RTREE[1])) + ETCW(INVCAP) * D;
    }();

    static void print2(std::string s = "", std::ostream & os = std::cout)
    {
      ANET.print(s+"address network: ",os);
      SEL.print(s+"bank select: ",os);
      RTREE[0].print(s+"data-read network select: ",os);
      RTREE[1].print(s+"data-read network data: ",os);
      WNET.print(s+"data-write network: ",os);
      BANK::print(s+"NODE: ",os);
      //BANK::print2(s,os);
    }

    static constexpr u64 num_bits() {return NB * BANK::NBITS;}
    static constexpr f64 read_latency() {return READ.d;}
    static constexpr f64 read_energy() {return READ.e;}
    static constexpr f64 write_energy() {return ACC.e + WNET.e + BANK::EWRITE;}
    static constexpr f64 array_width() {return BANK::WIDTH * BX;}
    static constexpr f64 array_height() {return BANK::HEIGHT * BY;}

    static constexpr u64 num_xtors()
    {
      return NB * BANK::XTORS + ACC.t + WNET.t + RTREE[0].t + RTREE[1].t;
    }

    static constexpr u64 num_fins()
    {
      return NB * BANK::FINS + ACC.f + WNET.f + RTREE[0].f + RTREE[1].f;
    }
  };


  constexpr bool ok_config(u64 E/*entries*/, u64 D/*data bits*/, u64 MAXN, u64 MAXM)
  {
    bool ok = E!=0 && D!=0; // mandatory
    ok = ok && std::has_single_bit(MAXN); // mandatory: MAXN must be a power of 2
    // prune off configs that are likely bad
    u64 banksize = MAXN * std::max(D,MAXM);
    ok = ok && (16*E*D <= banksize*banksize || (MAXN>=1024 && MAXM>=512));
    return ok;
  }


  template<u64 E, u64 D, u64 MAXN, u64 MAXM>
  struct sram_banked;


  template<u64 E, u64 D, u64 MAXN, u64 MAXM> requires (ok_config(E,D,MAXN,MAXM))
  struct sram_banked<E,D,MAXN,MAXM> : sram_common<sram_banked<E,D,MAXN,MAXM>> {
    static_assert(E!=0 && D!=0 && std::has_single_bit(MAXN));
    static constexpr bool ok = true;

    template<u64 R/*rows*/, u64 W/*row width*/>
    static constexpr std::tuple<u64,u64> banking()
    {
      constexpr u64 N = MAXN;
      static_assert(R>=N);
      static_assert(W>=MAXM/2);
      constexpr f64 BIAR = f64(N) / (W*SRAM_CELL.aspect_ratio); // bank inverse aspect ratio
      constexpr f64 FACTOR = std::max(2.,BIAR*BIAR);
      static_assert(FACTOR>=1);
      if constexpr (R <= W * SRAM_CELL.aspect_ratio * FACTOR) {
        // single column
        return {1/*BX*/,(R+N-1)/N/*BY*/};
      } else {
        // multiple columns (BY=2^k)
        // we want squarish shape
        constexpr f64 S = mysqrt(R*W*SRAM_CELL.aspect_ratio); // square side
        static_assert(S>=N);
        static_assert(S<=R);
        constexpr u64 BY = std::min(to_pow2(S/N),std::bit_floor(R/N));
        static_assert(BY>=1 && std::has_single_bit(BY));
        constexpr u64 BX = (R+BY*N-1)/(BY*N);
        static_assert(BX>=1);
        return {BX,BY};
      }
    }

    static constexpr auto params = []() {
      if constexpr (D > MAXM) {
        // data width = bank width
        constexpr u64 M = MAXM;
        if constexpr (E <= MAXN) {
          // single bank
          return std::tuple{1/*BX*/,1/*BY*/,E/*N*/,M};
        } else {
          // multiple banks (N=MAXN)
          auto [BX,BY] = banking<E,D>();
          return std::tuple{BX,BY,MAXN,M};
        }
      } else {
        // D <= MAXM
        constexpr u64 MAXK = std::bit_floor(MAXM/D);
        if constexpr (E <= MAXN*MAXK) {
          // single bank
          u64 K = MAXK;
          u64 N = (E+K-1)/K;
          // we want squarish shape
          while (K*D*SRAM_CELL.aspect_ratio > 2*N && K>=2 && (E+K/2-1)/(K/2)<=MAXN) {
            K /= 2;
            N = (E+K-1)/K;
          }
          u64 M = K * D;
          return std::tuple{1/*BX*/,1/*BY*/,N,M};
        } else {
          // multiple banks (N=MAXN)
          constexpr u64 M = MAXK * D;
          constexpr u64 EK = (E+MAXK-1) / MAXK;
          auto [BX,BY] = banking<EK,M>();
          return std::tuple{BX,BY,MAXN,M};
        }
      }
    } ();

    static constexpr u64 BX = std::get<0>(params);
    static constexpr u64 BY = std::get<1>(params);
    static constexpr u64 N = std::get<2>(params);
    static constexpr u64 M = std::get<3>(params);

    using ARR = sram_array<BX,BY,N,M,D>;
    static_assert(ARR::NBITS >= E*D);

    static void print2(std::string s = "", std::ostream & os = std::cout)
    {
      os << s;
      os << "BX=" << BX;
      os << " BY=" << BY;
      os << " N=" << N;
      os << " M=" << M;
      os << std::endl;
      //ARR::print2("",os);
    }

    static constexpr u64 num_bits() {return ARR::NBITS;}
    static constexpr u64 num_xtors() {return ARR::XTORS;}
    static constexpr u64 num_fins() {return ARR::FINS;}
    static constexpr f64 read_latency() {return ARR::LATENCY;}
    static constexpr f64 read_energy() {return ARR::EREAD;}
    static constexpr f64 write_energy() {return ARR::EWRITE;}
    static constexpr f64 array_width() {return ARR::WIDTH;}
    static constexpr f64 array_height() {return ARR::HEIGHT;}
  };


  template<u64 E, u64 D, u64 MAXN, u64 MAXM> requires (! ok_config(E,D,MAXN,MAXM))
  struct sram_banked<E,D,MAXN,MAXM> : sram_common<sram_null> {
    static constexpr bool ok = false;
  };


  template<typename T>
  concept sram_type = requires(T& x) {[]<u64 E, u64 D, u64 N, u64 M>(sram_banked<E,D,N,M>&){}(x);};

  template<sram_type T1, sram_type ...T>
  struct best_config;

  template<sram_type T>
  struct best_config<T> {
    using type = T;
  };

  template<sram_type T1, sram_type T2, sram_type ...T>
  struct best_config<T1,T2,T...> {
    using best = best_config<T2,T...>::type;
    static constexpr bool better = ! best::ok || (T1::ok && (T1::COST <= best::COST));
    using type = std::conditional_t<better,T1,best>;
  };

  template<u64 E, u64 D, u64 N, u64 ...M>
  using sram_bestM = best_config<sram_banked<E,D,N,M>...>::type;

  template<u64 E, u64 D, u64 ...N>
  using sram_bestN = best_config<sram_bestM<E,D,N,64,128,256>...>::type;

  template<u64 E, u64 D>
  using sram = sram_bestN<E,D,64,128,256>;


  // ######################################################

  inline constexpr f64 OUTCAP = INVCAP;

  inline constexpr circuit INV = inv{}.make(OUTCAP);

  template<u64 N>
  inline constexpr circuit AND = anding(N,OUTCAP);

  template<u64 N>
  inline constexpr circuit OR = oring(N,OUTCAP);

  template<u64 N>
  inline constexpr circuit NAND = nanding(N,OUTCAP);

  template<u64 N>
  inline constexpr circuit NOR = noring(N,OUTCAP);

  template<u64 N> requires (N>=2)
  inline constexpr circuit XOR = reduction(N,[](f64 co){return xor2(co);},OUTCAP);

  template<u64 N> requires (N>=2)
  inline constexpr circuit XNOR = reduction(N,[](f64 co){return xnor2(co);},OUTCAP);

  template<u64 WIDTH>
  inline constexpr circuit ADD = adder_ks<false,false>(WIDTH,OUTCAP);

  template<u64 WIDTH>
  inline constexpr circuit INC = adder_ks<true,true>(WIDTH,OUTCAP);

  template<u64 WIDTH>
  inline constexpr circuit SUB = subtract(WIDTH,OUTCAP);

  template<u64 WIDTH, u64 N> requires (N>=2)
  inline constexpr circuit ADDN = []() {
    // add N integers, WIDTH bits each
    std::vector<u64> count (WIDTH,N);
    return carry_save_adder(count,OUTCAP);
  }();

  template<u64 N>
  inline constexpr circuit EQUAL = xnor2(AND<N>.ci) * N + AND<N>;

  template<u64 N>
  inline constexpr circuit NEQ = xor2(OR<N>.ci) * N + OR<N>;

  template<u64 N>
  inline constexpr circuit GT = unsigned_greater_than(N,OUTCAP,false); // TODO: signed

  template<u64 N>
  inline constexpr circuit GTE = unsigned_greater_than(N,OUTCAP,true); // TODO: signed

  template<u64 N, u64 WIDTH> requires (N>=2)
  inline constexpr auto MUX = mux(N,WIDTH,OUTCAP);

  template<u64 N>
  inline constexpr circuit priority_encoder = []() {
    static_assert(N!=0);
    if constexpr (N==1) {
      return circuit{};
    } else {
      f64 co = INVCAP;
      circuit output = nor{2}.make(co);
      f64 ocap[N-1];
      for (u64 i=0; i<(N-1); i++) ocap[i] = output.ci;
      auto nor2 = [] (f64 co) {return nor{2}.make(co);};
      auto nand2 = [] (f64 co) {return nand{2}.make(co);};
      circuit prefix_or = parallel_prefix(1,nor2,nand2,ocap);
      circuit input = inv{}.make(prefix_or.ci);
      return input * (N-1) + prefix_or + output * (N-1);
    }
  }();

  // circuit for replicating a value M times
  template<u64 DATABITS, u64 COPIES>
  inline constexpr circuit REP = []() {
    // wires not modeled (TODO)
    circuit tree = broadcast_tree(COPIES);
    if (tree.ci > INVCAP) {
      // composing replication must not allow to build a large replicator at no hardware cost,
      // so we charge a hardware cost corresponding to that of an inverter
      tree = inv{}.make(tree.ci) + tree;
    }
    return tree * DATABITS;
  }();

  // signed mul has roughly same complexity as unsigned
  template<u64 N, u64 M>
  inline constexpr circuit IMUL = multiplier(N,M,OUTCAP); // N-bit x M-bit

  template<u64 NA, u64 NB, u64 NC>
  inline constexpr circuit IMAD = multiply_add(NA,NB,NC,OUTCAP); // A+BxC; NX=#bits of X

  template<u64 HARDN, u64 M>
  inline constexpr circuit HIMUL = [] () {
    std::bitset<std::bit_width(HARDN)> N {HARDN};
    return multiply_by_constant(N,M,OUTCAP); // HARDN x M-bit
  }();

  template<u64 N, u64 HARDD>
  inline constexpr circuit UDIV = divide_by_constant<N,HARDD>(OUTCAP); // unsigned N-bit / HARDD

  template<u64 N, u64 HARDD>
  inline constexpr circuit UMOD = remainder_divide_by_constant<N,HARDD>(OUTCAP); // unsigned N-bit % HARDD

  template<u64 N, u64 M>
  constexpr circuit ROM(const std::array<std::bitset<M>,N> &data)
  {
    return pseudo_rom(data,OUTCAP);
  }

  template<u64 OUTPUTS>
  inline constexpr circuit decoder = decode2(OUTPUTS,OUTCAP);


  // ######################################################
  // FOR FLORPLANNING (one rectangle per RAM)

  class rectangle {
    friend class globals;
    friend class region;
    friend class zone;
    template<memdatatype,u64> friend class ram;
    f64 w = 0; // rectangle width
    f64 h = 0; // rectangle height
    bool rotated = false;
    std::vector<rectangle*> subs {}; // sub-rectangles
    // (x,y) coord of lower left corner
    f64 x = 0;
    f64 y = 0;
    u64 color = 0;
    bool placed = false;
    std::string label;
    const u64 id;

    f64 area() const {return w*h;}
    f64 tall() const {return h>=w;};

    bool empty() const
    {
      return subs.empty();
    }

    std::array<f64,2> coord_f64() const
    {
      // the ram access point is at the center of the longest edge
      // coordinates in um (integer)
      if (w>h) {
        return {x+w/2,y};
      } else {
        return {x,y+h/2};
      }
    }

    bool similar(const rectangle &r1, const rectangle &r2) const
    {
      return r1.w==r2.w && r1.h==r2.h;
    }

    bool similar(const rectangle *p1, const rectangle *p2) const
    {
      assert(p1 && p2);
      return similar(*p1,*p2);
    }

    void rotate()
    {
      std::swap(w,h);
      rotated ^= 1;
      for (auto &p : subs) {
        assert(p);
        p->rotate();
      }
    }

    u64 num_similar(std::span<rectangle*> s) const
    {
      for (auto it=s.begin(); it!=s.end(); it++) {
        if (! similar(s[0],*it)) {
          return std::distance(s.begin(),it);
        }
      }
      return s.size();
    }

    std::array<u64,2> grid_size(std::span<rectangle*> s) const
    {
      u64 n = s.size();
      assert(n);
      assert(num_similar(s)==n);
      u64 ny = mysqrt(n);
      u64 nx = (n+ny-1)/ny;
      assert(nx*ny>=n);
      return {nx,ny};
    }

    rectangle() : id(std::numeric_limits<u64>::max())
    {
      static u64 col = 0;
      color = col++;
    }

    rectangle(u64 id, f64 w, f64 h, std::string label="") : w(w), h(h), label(label), id(id)
    {
      assert(w>0);
      assert(h>0);
      rotated = false;
      if (! tall()) rotate();
    }

    rectangle(std::span<rectangle*> s) : id(std::numeric_limits<u64>::max())
    {
      assert(empty());
      assert(s.size()>=2);
      for ([[maybe_unused]] auto &p : s) assert(p->tall());
      u64 n = num_similar(s);
      if (n == s.size()) {
        // similar rectangles
        auto [nx,ny] = grid_size(s);
        w = nx * s[0]->w;
        h = ny * s[0]->h;
      } else {
        // two dissimilar rectangles
        assert(s.size()==2);
        w = s[0]->w + s[1]->w;
        h = std::max(s[0]->h,s[1]->h);
      }
      for (auto &p : s) subs.push_back(p);
      if (! tall()) rotate();
    }

    void sort()
    {
      std::stable_sort(subs.begin(),subs.end(),[](rectangle *p,rectangle *q){return p->area()<q->area();});
    }

    bool group_similar()
    {
      sort();
      std::span<rectangle*> s = subs;
      for (u64 i=0; i<s.size()-1; i++) {
        if (similar(s[i],s[i+1])) {
          auto tail = s.last(s.size()-i);
          u64 n = num_similar(tail);
          rectangle* newrect = new rectangle (s.subspan(i,n));
          subs.erase(subs.begin()+i,subs.begin()+i+n);
          subs.insert(subs.begin()+i,std::move(newrect));
          return true;
        }
      }
      return false;
    }

    void group_two_smallest()
    {
      if (subs.size()<2) return;
      sort();
      std::span<rectangle*> s = subs;
      rectangle* newrect = new rectangle (s.first(2));
      subs.erase(subs.begin(),subs.begin()+2);
      subs.insert(subs.begin(),std::move(newrect));
    }

    void new_subrec(rectangle *r)
    {
      assert(r);
      if (r->empty() || ! r->subs.back()->empty()) {
        r->subs.push_back(new rectangle);
      }
    }

    void new_region()
    {
      new_subrec(this);
    }

    void new_zone()
    {
      // new zone in current region
      if (empty()) new_region();
      assert(!empty());
      new_subrec(subs.back());
    }

    void zone_contains(rectangle &p)
    {
      subs.push_back(&p);
    }

    void region_contains(rectangle &p)
    {
      if (empty()) new_subrec(this);
      p.color = color;
      subs.back()->zone_contains(p);
    }

    void contains(rectangle &p)
    {
      if (empty()) new_subrec(this);
      assert(!empty());
      subs.back()->region_contains(p);
    }

    void place(f64 xx, f64 yy)
    {
      assert(!placed);
      x = xx;
      y = yy;
      placed = true;
      if (empty()) {
        return;
      } else if (subs.size()==1) {
        subs[0]->place(x,y);
      } else if (! similar(subs[0],subs[1])) {
        // two dissimilar rectangles
        subs[0]->place(x,y);
        if (rotated) {
          subs[1]->place(x,y+subs[0]->h);
        } else {
          subs[1]->place(x+subs[0]->w,y);
        }
      } else {
        // grid of similar rectangles
        auto [nx,ny] = grid_size(subs);
        if (rotated) {
          std::swap(nx,ny);
          for (u64 i=0; i<subs.size(); i++)
            subs[i]->place(x+(i%nx)*subs[0]->w, y+(i/nx)*subs[0]->h);
        } else {
          for (u64 i=0; i<subs.size(); i++)
            subs[i]->place(x+(i/ny)*subs[0]->w, y+(i%ny)*subs[0]->h);
        }
      }
    }

    void make_tree()
    {
      for (auto &p : subs) {
        assert(p);
        if (! p->empty())
          p->make_tree();
      }
      while (subs.size() > 1) {
        while (group_similar());
        group_two_smallest();
      }
      assert(subs.size()==1);
      assert(w==0 && h==0);
      w = subs[0]->w;
      h = subs[0]->h;
      assert(area()!=0);
    }

    void place()
    {
      assert(!placed);
      make_tree();
      place(0,0);
    }

    void print_rect(std::ostream & os = std::cout)
    {
      if (id < std::numeric_limits<u64>::max()) {
        u64 randcol = ((color+39) * 10368889)  & 0xFFFFFF;
        std::string lab = std::to_string(id);
        if (! label.empty()) lab += "\\n" + label;
        os << "rect" << id << " [label=\"" << lab << "\", width=" << w << ", height=" << h << ", pos=\"" << x+w/2 << "," << y+h/2 << "!\", fillcolor=\"#" << std::hex << randcol << std::dec << "\"];\n";
      }
      for (auto &p : subs) {
        p->print_rect(os);
      }
    }

    void output_floorplan()
    {
      assert(h!=0);
      std::ofstream file;
      file.open("floorplan.gv");
      file << "/*\n" << "dot floorplan.gv -Tpdf -o floorplan.pdf\n" << "*/\n";
      file << "graph {\n";
      u64 fontsize =  std::max(u64(1),u64(w));
      file << "node [shape=rectangle, fixedsize=true, style=filled, fontsize=" << fontsize << "];\n";
      print_rect(file);
      file << "layout=neato;\n";
      file << "}\n";
      file.close();
    }
  };


  // ######################################################
  // FOR CONDITIONAL EXECUTION

  inline class exec_control {
    template<u64,arith> friend class val;
    template<u64,arith> friend class reg;
    template<valtype,u64> friend class arr;
    template<memdatatype,u64> friend class ram;
    friend class globals;
    template<valtype T, action A> requires (std::same_as<return_type<A>,void>)
      friend void execute_if(T&&,const A&);
    template<valtype T, action A> friend auto execute_if(T&&,const A&);
  private:
    bool active = true;
    u64 time = 0;
    u64 location = 0;
    exec_control(const exec_control &s) = default;
    exec_control& operator=(const exec_control &s) = default;

    void set_state(bool cond, u64 t=0, u64 l=0)
    {
      active = cond;
      time = t;
      location = l;
    }

    auto to_val() const; // defined after class val
  public:
    exec_control() : active(true), time(0), location(0) {}
  } exec;


  // ######################################################
  // COUNTERS (STATISTICS)

  template<typename T> requires (std::integral<T> || std::floating_point<T>)
  class global {
    friend class ::harcom_superuser;
    friend class globals;
    friend class region;
    template<u64,arith> friend class val;
    template<u64,arith> friend class reg;
    template<valtype,u64> friend class arr;
    template<memdatatype,u64> friend class ram;
  private:
    T data = 0;
    global() : data(0) {}
    global(const global&) = default;
    global& operator= (const global& x) = default;
    void operator+= (T i) {data+=i;}
    void operator++(int) {data++;}
    void operator&() = delete;
  public:
    global(T x) : data(x) {}
    operator T() const {return data;}

    void print(std::string s = "", std::ostream & os = std::cout) const
    {
      os << s << data << std::endl;
    }
  };


  // ######################################################
  // REGION

  class region {
    friend class globals;
    template<u64,arith> friend class reg;
  private:
    u64 ramid; // minimum RAM id in this region
    global<u64> storage;
    global<u64> storage_sram;
    global<f64> energy_fJ;
    global<u64> xtors_this_cycle;
    global<u64> fins_this_cycle;
    global<u64> transistors;
    global<u64> xtor_fins;
    global<f64> area_sram_mm2;

    void update_xtors(u64 xtors, u64 fins)
    {
      xtors_this_cycle += xtors;
      fins_this_cycle += fins;
    }

    void update_transistors()
    {
      transistors = std::max(transistors,xtors_this_cycle);
      xtor_fins = std::max(xtor_fins,fins_this_cycle);
      xtors_this_cycle = 0;
      fins_this_cycle = 0;
    }

    void update_storage(u64 nbits, bool is_sram)
    {
      storage += nbits;
      if (is_sram) storage_sram += nbits;
    }

    void update_energy(f64 e)
    {
      energy_fJ += e;
    }

    void update_sram_mm2(f64 area)
    {
      area_sram_mm2 += area;
    }

  public:
    // circular dependency with panel :(
    region();
    void enter();
  };


  // ######################################################
  // PANEL

  inline class globals {
    friend class region;
    friend class zone;
    template<u64,arith> friend class val;
    template<u64,arith> friend class reg;
    template<valtype,u64> friend class arr;
    template<memdatatype,u64> friend class ram;
    friend class proxy;
    friend class ::harcom_superuser;
    static constexpr u64 first_cycle = 1;
  public:
    global<u64> clock_cycle_ps = 300;
    global<u64> cycle = first_cycle;

  private:
    bool arr_of_regs_ctor = false;
    bool storage_destroyed = false;

    std::vector<region*> regions;
    static region default_region;
    rectangle floorplan;

    std::vector<rectangle*> rams;
    std::vector<region*> ram_region;
    u64 log2_numrams = 0;
    region *current_region = &default_region;

    struct connection {
      circuit one_wire{};
      u64 delay = 0;
      u64 distance = 0;
      u64 use = 0;
      void set(f64 dist)
      {
        assert(dist>=0);
        one_wire = wire(dist);
        delay = one_wire.delay();
        distance = dist;
      }
    };

    std::vector<connection> connect;

    void next_cycle()
    {
      if (clock_cycle_ps == 0) {
        std::cerr << "clock cycle must be non-null" << std::endl;
        std::terminate();
      }
      for (const auto &r : regions) {
        r->update_transistors();
      }
      cycle++;
    }

    u64 register_location() const
    {
      if (rams.empty()) return 0;
      return std::max(current_region->ramid,rams.back()->id);
    }

    u64 default_location() const
    {
      return current_region->ramid;
    }

    void new_location(u64 id, region *r)
    {
      assert(r!=nullptr);
      if (id < ram_region.size()) {
        ram_region[id] = r;
      } else {
        assert(id==ram_region.size());
        ram_region.push_back(r);
      }
    }

    void check_floorplan() const
    {
      if (rams.size() > 1 && ! floorplan.placed) {
        std::cerr << "call to make_floorplan() missing" << std::endl;
        std::terminate();
      }
    }

    region* get_region(u64 ramid) const
    {
      if (ram_region.empty()) {
        return &default_region;
      } else {
        return ram_region[ramid];
      }
    }

    void update_storage(u64 nbits, bool is_sram)
    {
      current_region->update_storage(nbits,is_sram);
    }

    void update_xtors(u64 xtors, u64 fins)
    {
      current_region->update_xtors(xtors,fins);
    }

    void update_sram_mm2(f64 area)
    {
      current_region->update_sram_mm2(area);
    }

    void update_energy(u64 locus, f64 e)
    {
      get_region(locus)->update_energy(e);
    }

    void update_logic(u64 locus, const circuit &c, bool actual = exec.active)
    {
      region *r = get_region(locus);
      r->update_xtors(c.t,c.f);
      if (actual)
        r->update_energy(c.e);
    }

    void add_ram(rectangle &p)
    {
      if (floorplan.placed) {
        std::cerr << "RAM cannot be created after floorplanning" << std::endl;
        std::terminate();
      }
      assert(p.id == rams.size());
      rams.push_back(&p);
      floorplan.contains(p);
      new_location(p.id,current_region);
    }

    void enter_region(region *r, bool init)
    {
      assert(r!=nullptr);
      current_region = r;
      if (init) {
        new_location(r->ramid,r);
      }
    }

    u64 index(u64 srcid, u64 dstid) const
    {
      //assert(srcid<rams.size() && dstid<rams.size());
      return srcid<<log2_numrams | dstid;
    }

    u64 connect_delay([[maybe_unused]] u64 srcid,
                      [[maybe_unused]] u64 dstid,
                      [[maybe_unused]] u64 nbits)
    {
#ifdef FREE_WIRING
      return 0;
#else
      if (srcid == dstid) {
        return 0; // no connection
      } else {
        // make_floorplan() must have been called
        connection &c =  connect[index(srcid,dstid)];
        region *rs = get_region(srcid);
        region *rd = get_region(dstid);
        region *r = (rs==rd)? rs : &default_region;
        r->update_xtors(nbits*c.one_wire.t, nbits*c.one_wire.f);
        if (exec.active) {
          c.use += nbits; // for energy calculation (deferred)
        }
        return c.delay;
      }
#endif
    }

    u64 connect_distance(u64 srcid, u64 dstid) const
    {
      if (srcid == dstid) return 0;
      return connect[index(srcid,dstid)].distance;
    }

    void update_wiring_energy()
    {
      if (connect.empty()) return;
      assert(rams.size() <= ram_region.size());
      for (u64 i=0; i<rams.size(); i++) {
        for (u64 j=0; j<rams.size(); j++) {
          connection &c = connect[index(i,j)];
          if (j==i) {
            assert(c.use==0);
            continue;
          }
          if (ram_region[i] == ram_region[j]) {
            ram_region[i]->update_energy(c.one_wire.e * c.use);
          } else {
            default_region.update_energy(c.one_wire.e * c.use);
          }
          c.use = 0;
        }
      }
    }

    template<typename T>
    global<T> total_cost(global<T> region::* stat)
    {
      if constexpr (std::floating_point<T>) {
        if (stat == &region::energy_fJ) {
          update_wiring_energy();
        }
      }
      global<T> sum;
      for (const auto &r : regions) {
        sum += r->*stat;
      }
      return sum;
    }

  public:

    auto storage(region &r = default_region)
    {
      if (&r == &default_region) {
        return total_cost(&region::storage);
      } else {
        return r.storage;
      }
    }

    auto storage_sram(region &r = default_region)
    {
      if (&r == &default_region) {
        return total_cost(&region::storage_sram);
      } else {
        return r.storage_sram;
      }
    }

    auto energy_fJ(region &r = default_region)
    {
      if (&r == &default_region) {
        return total_cost(&region::energy_fJ);
      } else {
        update_wiring_energy();
        return r.energy_fJ;
      }
    }

    auto transistors(region &r = default_region)
    {
      if (cycle == first_cycle) {
        if (&r == &default_region) {
          return total_cost(&region::xtors_this_cycle);
        } else {
          return r.xtors_this_cycle;
        }
      } else {
        if (&r == &default_region) {
          return total_cost(&region::transistors);
        } else {
          return r.transistors;
        }
      }
    }

    auto xtor_fins(region &r = default_region)
    {
      if (cycle == first_cycle) {
        if (&r == &default_region) {
          return total_cost(&region::fins_this_cycle);
        } else {
          return r.fins_this_cycle;
        }
      } else {
        if (&r == &default_region) {
          return total_cost(&region::xtor_fins);
        } else {
          return r.xtor_fins;
        }
      }
    }

    auto area_sram_mm2(region &r = default_region)
    {
      if (&r == &default_region) {
        return total_cost(&region::area_sram_mm2);
      } else {
        return r.area_sram_mm2;
      }
    }

    f64 dyn_power_mW(region &r = default_region)
    {
      assert(cycle>first_cycle);
      assert(clock_cycle_ps != 0);
      return energy_fJ(r) / ((cycle-first_cycle)*clock_cycle_ps); // mW
    }

    f64 sta_power_mW(region &r = default_region)
    {
      return leakage_power_mW(xtor_fins(r),storage_sram(r));
    }

    void make_floorplan()
    {
      if (floorplan.placed || rams.empty()) {
        return;
      }
      assert(connect.empty());
      if (regions.back()->ramid >= rams.size()) {
        std::cerr << "region contains no RAMs" << std::endl;
        std::terminate();
      }
      floorplan.place();
#ifdef FLOORPLAN
      floorplan.output_floorplan();
#endif
      log2_numrams = std::bit_width(rams.size()-1);
      connect.resize(index(rams.size()-1,rams.size()-1)+1);
      for (u64 i=0; i<rams.size(); i++) {
        for (u64 j=0; j<rams.size(); j++) {
          auto [xi,yi] = rams.at(i)->coord_f64();
          auto [xj,yj] = rams.at(j)->coord_f64();
          f64 dist = fabs(xi-xj) + fabs(yi-yj); // Manhattan distance (um)
          connect[index(i,j)].set(dist);
        }
      }
      default_region.enter();
    }

    void print(region &r, std::ostream & os = std::cout)
    {
      os << std::setprecision(3);
      if (clock_cycle_ps != 0 && cycle>first_cycle) {
        clock_cycle_ps.print("clock cycle (ps): ",os);
        os << "clock frequency (GHz): " << 1000./clock_cycle_ps << std::endl;
      }
      storage(r).print("storage (bits): ",os);
      auto xtors = transistors(r);
      xtors.print("transistors: ",os);
      if (xtors != 0)
        os << "fins/transistor: " << f64(xtor_fins(r)) / xtors << std::endl;
      area_sram_mm2(r).print("SRAM area (mm2): ",os);
      if (cycle == first_cycle) {
        energy_fJ(r).print("dynamic energy (fJ): ",os);
      } else if (clock_cycle_ps != 0) {
        os << "dynamic power (mW): " << dyn_power_mW(r) << std::endl;
      }
      os << "static power (mW): " << sta_power_mW(r) << std::endl;
    }

    void print(std::ostream & os = std::cout)
    {
      print(default_region,os);
    }

  } panel;


  inline region globals::default_region;


  // ######################################################
  // REGION (cont'd)

  inline region::region() : ramid(panel.rams.size())
  {
    if (panel.floorplan.placed) {
      std::cerr << "region cannot be defined after floorplanning" << std::endl;
      std::terminate();
    }
    panel.floorplan.new_region();
    panel.regions.push_back(this);
    panel.enter_region(this,true);
  }

  inline void region::enter()
  {
    panel.enter_region(this,false);
  }

  // ######################################################
  // ZONE

  class zone {
  public:
    zone()
    {
      panel.floorplan.new_zone();
    }
  };

  // ######################################################
  // VAL

  template<u64 N, arith T = u64>
  class val {
    static_assert(N!=0,"number of val bits cannot be null");
    static_assert(N<=bitwidth<T>,"number of val bits exceeds the underlying C++ type");
    static_assert(N==bitwidth<T> || std::integral<T>);

    template<u64,arith> friend class val;
    template<u64,arith> friend class reg;
    template<valtype, u64> friend class arr;
    template<memdatatype,u64> friend class ram;
    template<valtype,u64> friend class rom;
    friend class proxy;
    friend class exec_control;
    friend class ::harcom_superuser;
    template<valtype U, action A> friend auto execute_if(U&&,const A&);

  private:

    T data = 0;
    u64 timing = 0; // time (picoseconds)
    u64 location = 0; // location (RAM id)
    u64 read_credit = 0;
    const bool actual = exec.active;

    T fit(T x) const requires std::integral<T>
    {
      return truncate<N>(x);
    }

    val(intlike auto x, u64 t, u64 l=0) requires std::integral<T> : data(fit(x)), timing(t), location(l) {}

    T sign_extended() const requires std::signed_integral<T>
    {
      if constexpr (N < bitwidth<T>) {
        assert((std::make_unsigned_t<T>(data) >> N) == 0);
      }
      constexpr T signbit = T(1) << (N-1);
      return (data ^ signbit) - signbit;
    }

    void set_time(u64 t)
    {
      // update even when exec.active is false (potential timing)
      timing = t;
    }

    void set_location(u64 ramid)
    {
      location = ramid;
    }

    void read()
    {
#ifndef FREE_FANOUT
      if (read_credit!=0) {
        read_credit--;
      } else {
#ifdef CHECK_FANOUT
        if (read_credit == 0) {
          std::cerr<< "fanout exhausted" << std::endl;
          std::terminate();
        }
#else
        // fanout exhausted: delay increases linearly with the number of reads
        // delay increment is that of a FO2 inverter (wires not modeled, TODO?)
        constexpr circuit fo2inv = inv{}.make(2*INVCAP) * N; // N inverters in parallel
        static_assert(N==0 || fo2inv.delay()!=0);
        // if the value is captured by a lambda executed by execute_if,
        // fanout costs energy regardless of the predicate
        panel.update_logic(site(),fo2inv,actual);
        set_time(time()+fo2inv.delay());
#endif // CHECK_FANOUT
      }
#endif // FREE_FANOUT
    }

    T get() & // lvalue
    {
      read();
      if constexpr (std::signed_integral<T>) {
        return sign_extended();
      } else {
        return data;
      }
    }

    T get() && // rvalue
    {
      T old_data;
      if constexpr (std::signed_integral<T>) {
        old_data = sign_extended();
      } else {
        old_data = data;
      }
#ifndef FREE_FANOUT
      data = 0; // destructive read (even if exec.active false)
#endif
      return old_data;
    }

#ifndef CHEATING_MODE
    u64 time() const
    {
      return std::max(exec.time,timing);
    }
#endif

    u64 site() const
    {
      return location;
    }

    auto get_vt() & // lvalue
    {
      return std::tuple {get(),time()}; // list initialization, get() executes before time()
    }

    auto get_vt() && // rvalue
    {
      return std::tuple {std::move(*this).get(),time()}; // list initialization
    }

    void operator= (val & x)
    {
      static_assert(std::integral<T>);
      auto [vx,tx] = x.get_vt();
      if (exec.active) data = fit(vx);
      set_time(tx);
      set_location(x.site());
    }

    void operator= (val && x)
    {
      static_assert(std::integral<T>);
      auto [vx,tx] = std::move(x).get_vt();
      if (exec.active) data = fit(vx);
      set_time(tx);
      set_location(x.site());
    }

    void operator&() = delete;

    template<u64 W>
    auto make_array(const std::tuple<T,u64> &vt)
    {
      static_assert(std::unsigned_integral<T>);
      static_assert(W!=0 && W<=64);
      constexpr u64 mask = (W==64)? -1 : (u64(1)<<W)-1;
      constexpr u64 M = (N+W-1)/W;
      auto [v,t] = vt;
      auto l = site();
      return arr<val<W>,M> {
        [&](){
          T vv = v;
          v >>= W;
          return val<W>{vv&mask,t,l};
        }
      };
    }

    val rotate_left(i64 shift, const std::tuple<T,u64> &vt)
    {
      static_assert(std::unsigned_integral<T>,"rotate_left() applies to unsigned int");
      i64 k = (shift>=0)? shift % N : (shift % i64(N))+N;
      assert(k>=0);
      auto [v,t] = vt;
      if (k==0) {
        return {v,t,site()};
      } else {
        auto y = v<<k | v>>(N-k);
        return {y,t,site()};
      }
    }

    auto ones(const std::tuple<T,u64> &vt)
    {
      static_assert(std::unsigned_integral<T>,"ones() applies to unsigned int");
      constexpr circuit c = (N>=2)? ADDN<1,N> : circuit{};
      panel.update_logic(site(),c);
      auto [v,t] = vt;
      auto n = std::popcount(truncate<N>(v));
      return val<std::bit_width(N)> {n,t+c.delay(),site()};
    }

    val one_hot(const std::tuple<T,u64> &vt)
    {
      static_assert(std::unsigned_integral<T>,"one_hot() applies to unsigned int");
      constexpr circuit c = priority_encoder<N>;
      auto [v,t] = vt;
      u64 y = v & (v^(v-1));
      panel.update_logic(site(),c);
      return {y,t+c.delay(),site()};
    }

    auto decode(const std::tuple<T,u64> &vt)
    {
      static_assert(std::unsigned_integral<T>,"decode() applies to unsigned int");
      constexpr u64 outputs = u64(1) << N;
      constexpr circuit c = decoder<outputs>;
      panel.update_logic(site(),c);
      auto [v,t] = vt;
      auto l = site();
      t += c.delay();
      return arr<val<1>,outputs> {
        [&](u64 i){return val<1>{i==v,t,l};}
      };
    }

    template<memdatatype U, u64 M>
    val connect(const ram<U,M> &dest, const std::tuple<T,u64> &vt)
    {
      panel.check_floorplan();
      auto [v,t] = vt;
      t += panel.connect_delay(site(),dest.ram_id(),size);
      return {v,t,dest.ram_id()};
    }

    template<ramtype U, u64 K>
    void distribute(std::span<U,K> mem,
                    const std::array<std::array<i64,4>,K> &send,
                    u64 node,
                    std::array<val,K> &out)
    {
      const auto &dst = send[node];
      u64 fo = (dst[0]>=0) + (dst[1]>=0) + (dst[2]>=0) + (dst[3]>=0);
      if (fo==0) {
        return;
      }
      switch (fo) {
      case 1: out[node].fanout(hard<2>{}); break;
      case 2: out[node].fanout(hard<3>{}); break;
      case 3: out[node].fanout(hard<4>{}); break;
      case 4: out[node].fanout(hard<5>{}); break;
      }
      for (u64 i=0; i<4; i++) {
        if (dst[i]<0) continue;
        out[dst[i]] = out[node].connect(mem[dst[i]]);
        distribute(mem,send,dst[i],out);
      }
    }

    template<ramtype U, u64 K>
    arr<val,K> distribute(std::span<U,K> mem, const std::tuple<T,u64> &vt)
    {
      static_assert(K!=0);
      panel.check_floorplan();

      struct info {
        u64 loc;  // original location
        u64 ramid; // id of first RAM in the range
        u64 entry; // closest RAM (range index)
        std::array<std::array<i64,4>,K> send; // send matrix
      };
      static std::vector<info> memo;

      // memoized?
      auto it = std::find_if(memo.begin(),memo.end(),[&](const info &e){
        return e.ramid == mem[0].ram_id() && e.loc == site();
      });

      if (it == memo.end()) {
        // not memoized yet, calculate the info
        auto coord = [&](u64 i){
          return mem[i].coord();
        };
        // make the grid
        std::array<i64,K> left, right, down, up;
        std::fill(left.begin(),left.end(),-1);
        std::fill(right.begin(),right.end(),-1);
        std::fill(down.begin(),down.end(),-1);
        std::fill(up.begin(),up.end(),-1);
        for (u64 i=0; i<K; i++) {
          auto [x,y] = coord(i);
          for (u64 j=i+1; j<K; j++)
            if (coord(j)[0] == x) {up[i] = j; down[j] = i; break;}
          for (u64 j=i+1; j<K; j++)
            if (coord(j)[1] == y) {right[i] = j; left[j] = i; break;}
        }
        // the graph can be disconnected if some RAMs not in the array are identical to those in the array
        // one extra link makes the graph connected (guaranteed by floorplanning algo)
        if (right[0]<0 && coord(K-1)[0] != coord(0)[0]) {
          down[0] = K-1;
          up[K-1] = 0;
        } else if (up[0]<0 && coord(K-1)[1] != coord(0)[1]) {
          left[0] = K-1;
          right[K-1] = 0;
        }

        memo.emplace_back();
        it = memo.end()-1;
        it->loc = site();
        it->ramid = mem[0].ram_id();
        auto closest = std::min_element(mem.begin(), mem.end(), [&](const U &r1, const U &r2) {
          return panel.connect_distance(site(),r1.ram_id()) < panel.connect_distance(site(),r2.ram_id());
        });
        it->entry = std::distance(mem.begin(),closest);

        // make the send matrix (K x [L,R,D,U])
        std::bitset<K> received;
        received[it->entry] = true;
        for (u64 i=0; i<K; i++)
          for (u64 j=0; j<4; j++)
            it->send[i][j] = -1;
        for (u64 i=0; i<K; i++) {
          // mostly horizontal links
          if (coord(i)[0] < coord(it->entry)[0]) { // left side
            if (right[i] >= 0 && ! received[i]) {
              it->send[right[i]][0] = i;
              received[i] = true;
            }
          } else if (coord(i)[0] > coord(it->entry)[0]) { // right side
            if (left[i] >= 0 && ! received[i]) {
              it->send[left[i]][1] = i;
              received[i] = true;
            }
          }
          // vertical links for remaining nodes
          if (coord(i)[1] < coord(it->entry)[1]) { // bottom
            if (up[i] >= 0 && ! received[i]) {
              it->send[up[i]][2] = i;
              received[i] = true;
            }
          } else if (coord(i)[1] > coord(it->entry)[1]) { // top
            if (down[i] >= 0 && ! received[i]) {
              it->send[down[i]][3] = i;
              received[i] = true;
            }
          }
        }
        assert(received.all());
      }

      // output
      assert(it!=memo.end());
      std::array<val,K> out;
      out[it->entry] = connect(mem[it->entry],vt);
      distribute(mem,it->send,it->entry,out);
      return out;
    }

  public:

    static constexpr u64 size = N;
    using type = T;

#ifdef CHEATING_MODE
    operator T() {return data;}

    u64 time() const
    {
      return std::max(exec.time,timing);
    }
#endif

    static constexpr T maxval = []() {
      if constexpr (N == bitwidth<T>) {
        return std::numeric_limits<T>::max();
      } else {
        static_assert(N < bitwidth<T>);
        if constexpr (std::unsigned_integral<T>) {
          return (T(1)<<N)-1;
        } else {
          static_assert(std::signed_integral<T>);
          return (T(1)<<(N-1))-1;
        }
      }
    }();

    static constexpr T minval = []() {
      if constexpr (N == bitwidth<T>) {
        return std::numeric_limits<T>::min();
      } else {
        static_assert(N < bitwidth<T>);
        if constexpr (std::unsigned_integral<T>) {
          return 0;
        } else {
          static_assert(std::signed_integral<T>);
          return -(T(1)<<(N-1));
        }
      }
    }();

    val() {}

    val(intlike auto x) requires std::integral<T> : val{x,0,panel.default_location()} {}

    val(val &x) : val{x.get(),x.time(),x.site()} {} // list initialization, get() executes before time()

    template<valtype U> requires std::unsigned_integral<T>
    val(U && x) : val{to_unsigned(std::forward<U>(x).get()),x.time(),x.site()} {} // list initialization

    template<valtype U>
    val(U && x) : val{std::forward<U>(x).get(),x.time(),x.site()} {} // list initialization

    template<std::integral auto FO>
    void fanout(hard<FO>) & // lvalue
    {
#ifndef FREE_FANOUT
      static_assert(FO>=2);
      if (FO < read_credit) return;
      // delay logarithmic with fanout
      panel.update_logic(site(),REP<N,FO>);
      set_time(time()+REP<N,FO>.delay());
      read_credit = FO;
#endif
    }

    [[nodiscard]] val&& fo1() & // lvalue
    {
      return std::move(*this);
    }

    void fanout(auto) && = delete; // rvalue, not needed

    void print(std::string before="", std::string after="\n", bool all=true, std::ostream & os=std::cout) const
    {
      if constexpr (std::signed_integral<T>) {
        os << before << +sign_extended();
      } else {
        os << before << +data;
      }
      if (all)
        os << " (t=" << time() << " ps, loc=" << site() << ")";
      os << after << std::flush;
    }

    void printb(std::string before="", std::string after="\n", bool all=true, std::ostream & os=std::cout) const
    {
      os << before;
      if constexpr (std::integral<T>) {
        os << std::bitset<N>(data);
      } else if constexpr (std::same_as<T,f32>) {
        os << std::bitset<N>(std::bit_cast<u32>(data));
      } else if constexpr (std::same_as<T,f64>) {
        os << std::bitset<N>(std::bit_cast<u64>(data));
      }
      if (all)
        os << " (t=" << time() << " ps, loc=" << site() << ")";
      os << after << std::flush;
    }

    template<u64 W>
    [[nodiscard]] auto make_array(val<W>&&) & // lvalue
    {
      return make_array<W>(get_vt());
    }

    template<u64 W>
    [[nodiscard]] auto make_array(val<W>&&) && // rvalue
    {
      return make_array<W>(std::move(*this).get_vt());
    }

    [[nodiscard]] val reverse() & // lvalue
    {
      static_assert(std::unsigned_integral<T>,"reverse() applies to unsigned int");
      return {reverse_bits(get()) >> (bitwidth<T>-N), time(), site()};
    }

    [[nodiscard]] val reverse() && // rvalue
    {
      static_assert(std::unsigned_integral<T>,"reverse() applies to unsigned int");
      return {reverse_bits(std::move(*this).get()) >> (bitwidth<T>-N), time(), site()};
    }

    [[nodiscard]] val rotate_left(intlike auto shift) & // lvalue
    {
      return rotate_left(shift,get_vt());
    }

    [[nodiscard]] val rotate_left(intlike auto shift) && // rvalue
    {
      return rotate_left(shift,std::move(*this).get_vt());
    }

    [[nodiscard]] auto ones() & // lvalue
    {
      return ones(get_vt());
    }

    [[nodiscard]] auto ones() && // rvalue
    {
      return ones(std::move(*this).get_vt());
    }

    [[nodiscard]] val one_hot() & // lvalue
    {
      return one_hot(get_vt());
    }

    [[nodiscard]] val one_hot() && // rvalue
    {
      return one_hot(std::move(*this).get_vt());
    }

    [[nodiscard]] auto decode() & // lvalue
    {
      return decode(get_vt());
    }

    [[nodiscard]] auto decode() && // rvalue
    {
      return decode(std::move(*this).get_vt());
    }

    template<memdatatype U, u64 M>
    [[nodiscard]] val connect(const ram<U,M> &dest) & // lvalue
    {
      return connect(dest,get_vt());
    }

    template<memdatatype U, u64 M>
    [[nodiscard]] val connect(const ram<U,M> &dest) && // rvalue
    {
      return connect(dest,std::move(*this).get_vt());
    }

    template<std::integral auto M>
    [[nodiscard]] arr<val,M> replicate(hard<M>) & // lvalue
    {
      // only the user knows the actual fanout (>=M) and can set it
      return arr<val,M> {[&](){return *this;}};
    }

    template<std::integral auto M>
    [[nodiscard]] arr<val,M> replicate(hard<M>) && // rvalue
    {
      // the user cannot set the fanout (rvalue), but the fanout is known
      if constexpr (M>1) fanout(hard<M>{});
      auto [v,t] = std::move(*this).get_vt();
      return arr<val,M> {[&](){return val{v,t,site()};}};
    }

    template<ramtype U, u64 M>
    [[nodiscard]] auto distribute(const U (&mem)[M]) & // lvalue
    {
      static_assert(M!=0);
      return distribute(std::span(mem),get_vt());
    }

    template<ramtype U, u64 M>
    [[nodiscard]] auto distribute(const U (&mem)[M]) && // rvalue
    {
      static_assert(M!=0);
      return distribute(std::span(mem),std::move(*this).get_vt());
    }

    template<ramtype U, u64 M>
    [[nodiscard]] auto distribute(const std::array<U,M> &mem) & // lvalue
    {
      static_assert(M!=0);
      return distribute(std::span(mem),get_vt());
    }

    template<ramtype U, u64 M>
    [[nodiscard]] auto distribute(const std::array<U,M> &mem) && // rvalue
    {
      static_assert(M!=0);
      return distribute(std::span(mem),std::move(*this).get_vt());
    }
  };


  // ######################################################

  inline auto exec_control::to_val() const
  {
    return val<1>(active,time,location);
  }

  // ######################################################

  class proxy {
  private:
    proxy() = delete;
    ~proxy() = delete;

    template<valtype T>
    static T make_val(T::type v, u64 t, u64 l)
    {
      static_assert(!regtype<T>);
      return {v,t,l};
    }

    static u64 connect_delay(const valtype auto &srcval, u64 dstid)
    {
      return panel.connect_delay(srcval.site(), dstid, srcval.size);
    }

    template<arith T>
    static auto get_vtl(T x)
    {
      return std::tuple<decltype(x),u64,u64> {x,0,0};
    }

    template<hardval T>
    static auto get_vtl(T x)
    {
      return std::tuple<decltype(x.value),u64,u64> {x.value,0,0};
    }

    template<valtype T1, valtype... Ti>
    static auto get_vtl(T1 && x1, Ti && ...xi)
    {
      // computation locus = site of operand with greatest timing
      // if tie, locus = leftmost among latest operands
      constexpr u64 N = 1 + sizeof...(xi);
      // read values before timing
      const auto vtup = std::make_tuple(std::forward<T1>(x1).get(),std::forward<Ti>(xi).get()...);
      constexpr std::array<u64,N> sz = {valt<T1>::size,valt<Ti>::size...};
      std::array<u64,N> tm = {x1.time(),xi.time()...};
      const std::array<u64,N> loc = {x1.site(),xi.site()...};
      auto latest = std::max_element(tm.begin(),tm.end());
      auto index = std::distance(tm.begin(),latest);
      u64 locus = loc[index];
      for (u64 i=0; i<N; i++) tm[i] += panel.connect_delay(loc[i],locus,sz[i]);
      u64 t = *std::max_element(tm.begin(),tm.end());
      return std::tuple_cat(vtup,std::make_tuple(t,locus));
    }

    template<arrayofval T>
    static auto get_vtl(T && aov)
    {
      // multi-input computation (input = std::array of valtype)
      // computation locus = site of operand with greatest timing
      // if tie, locus = site of operand with smallest index among latests
      using arraytype = std::remove_reference_t<decltype(aov)>;
      constexpr u64 N = std::tuple_size_v<arraytype>;
      static_assert(N!=0);
      using rawtype = arraytype::value_type::type;
      std::array<rawtype,N> rawv;
      if constexpr (std::is_rvalue_reference_v<decltype(aov)>) {
        for (u64 i=0; i<N; i++) rawv[i] = std::move(aov[i]).get();
      } else {
        for (u64 i=0; i<N; i++) rawv[i] = aov[i].get();
      }
      std::array<u64,N> vt;
      for (u64 i=0; i<N; i++) vt[i] = aov[i].time();
      const auto latest = std::max_element(vt.begin(),vt.end());
      const auto index = std::distance(vt.begin(),latest);
      const u64 locus = aov[index].site();
      for (u64 i=0; i<N; i++) vt[i] += panel.connect_delay(aov[i].site(),locus,aov[i].size);
      u64 t = *std::max_element(vt.begin(),vt.end());
      return std::tuple<std::array<rawtype,N>,u64,u64> {rawv,t,locus};
    }

    template<arrtype T>
    static auto get_vtl(T && a)
    {
      if constexpr (std::is_rvalue_reference_v<decltype(a)>) {
        return get_vtl(std::move(a.elem));
      } else {
        return get_vtl(a.elem);
      }
    }

    static void update_logic(u64 locus, const circuit &c)
    {
      panel.update_logic(locus,c);
    }

    // for splitting bits
    template<u64 N1, u64... Ni>
    class split_helper {
      static_assert(N1!=0 && ((Ni!=0) && ...),"splits must have a non-null size");
    public:
      std::tuple<val<N1>,val<Ni>...> tup;

      template<valtype T>
      split_helper(T && x)
      {
        constexpr u64 sum = (N1+...+Ni);
        static_assert(valt<T>::size==sum,"sum of split sizes must match number of bits");
        constexpr std::array N = {N1,Ni...};
        auto [v,t] = std::forward<T>(x).get_vt();
        u64 pos = sum;
        static_loop<1+sizeof...(Ni)>([&]<u64 I>() {
            assert(pos>=N[I]);
            pos -= N[I];
            std::get<I>(tup) = {v>>pos,t,x.site()};
          });
      }
    };

    template<u64,arith> friend class reg;
    template<valtype,u64> friend class arr;
    template<memdatatype,u64> friend class ram;
    template<valtype,u64> friend class rom;

    template<valtype T1, valtype T2>
    friend val<1> operator== (T1&&, T2&&);

    template<valtype T1, arith T2>
    friend val<1> operator== (T1&&, T2);

    template<valtype T1, hardval T2>
    friend val<1> operator== (T1&&, T2);

    template<valtype T1, valtype T2>
    friend val<1> operator!= (T1&&, T2&&);

    template<valtype T1, arith T2>
    friend val<1> operator!= (T1&&, T2);

    template<valtype T1, hardval T2>
    friend val<1> operator!= (T1&&, T2);

    template<valtype T1, valtype T2> requires (ival<T1> && ival<T2>)
    friend val<1> operator> (T1&&, T2&&);

    template<valtype T1, std::integral T2> requires (ival<T1>)
    friend val<1> operator> (T1&&, T2);

    template<valtype T1, hardval T2> requires (ival<T1> && intlike<T2>)
    friend val<1> operator> (T1&&, T2);

    template<valtype T1, valtype T2> requires (ival<T1> && ival<T2>)
    friend val<1> operator< (T1&&, T2&&);

    template<valtype T1, std::integral T2> requires (ival<T1>)
    friend val<1> operator< (T1&&, T2);

    template<valtype T1, hardval T2> requires (ival<T1> && intlike<T2>)
    friend val<1> operator< (T1&&, T2);

    template<valtype T1, valtype T2> requires (ival<T1> && ival<T2>)
    friend val<1> operator>= (T1&&, T2&&);

    template<valtype T1, std::integral T2> requires (ival<T1>)
    friend val<1> operator>= (T1&&, T2);

    template<valtype T1, hardval T2> requires (ival<T1> && intlike<T2>)
    friend val<1> operator>= (T1&&, T2);

    template<valtype T1, valtype T2> requires (ival<T1> && ival<T2>)
    friend val<1> operator<= (T1&&, T2&&);

    template<valtype T1, std::integral T2> requires (ival<T1>)
    friend val<1> operator<= (T1&&, T2);

    template<valtype T1, hardval T2> requires (ival<T1> && intlike<T2>)
    friend val<1> operator<= (T1&&, T2);

    template<valtype T1, valtype T2> requires (ival<T1> && ival<T2>)
    friend auto operator+ (T1&&, T2&&);

    template<valtype T1, intlike T2> requires (ival<T1>)
    friend auto operator+ (T1&&, T2);

    template<valtype T>
    friend auto operator- (T&&);

    template<valtype T1, valtype T2> requires (ival<T1> && ival<T2>)
    friend auto operator- (T1&&, T2&&);

    template<valtype T1, intlike T2> requires (ival<T1>)
    friend auto operator- (T1&&, T2);

    template<intlike T1, valtype T2> requires (ival<T2>)
    friend auto operator- (T1, T2&&);

    template<valtype T1, intlike T2> requires (ival<T1>)
    friend auto operator<< (T1&&, T2);

    template<valtype T1, intlike T2> requires (std::unsigned_integral<base<T1>>)
    friend auto operator>> (T1&&, T2);

    template<valtype T1, intlike T2> requires (std::signed_integral<base<T1>>)
    friend auto operator>> (T1&&, T2);

    template<valtype T1, valtype T2> requires (ival<T1> && ival<T2>)
    friend auto operator* (T1&&, T2&&);

    template<valtype T1, intlike T2> requires (ival<T1>)
    friend auto operator* (T1&&, T2);

    template<valtype T1, intlike T2> requires (ival<T1>)
    friend auto operator/ (T1&&, T2);

    template<valtype T1, intlike T2> requires (ival<T1>)
    friend auto operator% (T1&&, T2);

    template<valtype T1, valtype T2>
    friend auto operator& (T1&&, T2&&);

    template<valtype T1, intlike T2>
    friend auto operator& (T1&&, T2);

    template<valtype T1, valtype T2>
    friend auto operator| (T1&&, T2&&);

    template<valtype T1, intlike T2>
    friend auto operator| (T1&&, T2);

    template<valtype T1, valtype T2>
    friend auto operator^ (T1&&, T2&&);

    template<valtype T1, std::integral T2>
    friend auto operator^ (T1&&, T2);

    template<valtype T1, hardval T2>
    friend auto operator^ (T1&&, T2);

    template<valtype T>
    friend auto operator~ (T&&);

    template<valtype TA, valtype TB, valtype TC> requires (ival<TA> && ival<TB> && ival<TC>)
    friend auto a_plus_bc(TA&&,TB&&,TC&&);

    template<valtype T1, valtype... Ti>
    friend auto concat(T1&&, Ti&&...);

    template<u64 N1, u64... Ni, valtype T>
    friend auto split(T&&);

    template<valtype T, valtype T1, valtype T2>
    friend auto select(T&&, T1&&, T2&&);

    template<valtype T, action A> requires (std::same_as<return_type<A>,void>)
    friend void execute_if(T&&,const A&);

    template<valtype T, action A>
    friend auto execute_if(T&&,const A&);
  };


  // ######################################################

  template<u64 N, arith T = u64>
  class reg : public val<N,T> {
    template<u64,arith> friend class val;
    friend class proxy;
    friend class ::harcom_superuser;
  public:
    using stg = flipflops<N>;

  private:
    u64 last_write_cycle = 0;

    void create()
    {
      if (panel.storage_destroyed) {
        std::cerr << "all storage (reg,ram) must have the same lifetime" << std::endl;
        std::terminate();
      }
      val<N,T>::set_location(panel.register_location());
      panel.update_storage(N,false);
      panel.update_xtors(stg::xtors,stg::fins);
    }

    T get() & {return val<N,T>::get();}
    T get() && = delete;

  public:

    reg()
    {
      create();
    }

    reg(reg &other) : val<N,T>{other}
    {
      create();
    }

    reg(reg &&) = delete;

    template<std::convertible_to<val<N,T>> U>
    reg(U &&x) : val<N,T>{std::forward<U>(x)}
    {
      create();
    }

    ~reg()
    {
      panel.storage_destroyed = true;
    }

    void assign_from(T v, u64 t, u64 loc)
    {
      if (panel.cycle <= last_write_cycle) {
        std::cerr << "single register write per cycle" << std::endl;
        std::terminate();
      }
      if (! panel.arr_of_regs_ctor)
        last_write_cycle = panel.cycle;
      t += panel.connect_delay(loc,val<N,T>::site(),val<N,T>::size);
      if (t < exec.time) {
        // exec.time is not null,  we are inside an execute_if
        val<N,T>::set_time(exec.time);
      } else {
        val<N,T>::set_time(t);
      }
      // location is fixed
      if (exec.active) {
        val<N,T>::data = val<N,T>::fit(v);
        panel.update_energy(val<N,T>::site(),stg::write_energy_fJ);
      }
    }

    void operator= (reg &x)
    {
      auto [vx,tx] = x.get_vt();
      assign_from(vx,tx,x.site());
    }

    void operator= (reg &&) = delete;

    template<typename U>
    void operator= (U && x)
    {
      if constexpr (valtype<U>) {
        if constexpr (std::is_rvalue_reference_v<decltype(x)>) {
          auto [vx,tx] = std::move(x).get_vt();
          assign_from(vx,tx,x.site());
        } else {
          auto [vx,tx] = x.get_vt();
          assign_from(vx,tx,x.site());
        }
      } else {
        assign_from(x,0,val<N,T>::site());
      }
    }

    void set_location(u64) = delete; // register location cannot be modified
    void fo1() = delete; // register value is persistent
  };


  // ######################################################

  template<valtype T, u64 N>
  class arr {
    template<valtype,u64> friend class arr;
    template<u64,arith> friend class val;
    template<memdatatype,u64> friend class ram;
    friend class proxy;
    friend class ::harcom_superuser;
  public:
    static constexpr u64 size = N;
    using type = T;
    using rawtype = T::type;
    static constexpr u64 nbits = N * T::size;

  private:
    std::array<T,N> elem {};

    using vtlinfo = std::tuple<std::array<rawtype,N>,u64,u64>;

    template<arrtype U>
    void copy_from(U && x)
    {
      static_assert(std::remove_reference_t<U>::size==N,"destination and source array must have the same size");
      if constexpr (std::is_rvalue_reference_v<decltype(x)>) {
        for (u64 i=0; i<N; i++) elem[i] = x.elem[i].fo1();
      } else {
        for (u64 i=0; i<N; i++) elem[i] = x.elem[i];
      }
    }

    void operator= (arr &x)
    {
      copy_from(x);
    }

    void operator& () = delete;

    auto get() & // lvalue
    {
      std::array<rawtype,N> b;
      for (u64 i=0; i<N; i++) b[i] = elem[i].get();
      return b;
    }

    auto get() && // rvalue
    {
      std::array<rawtype,N> b;
      for (u64 i=0; i<N; i++) b[i] = elem[i].fo1().get();
      return b;
    }

    void set_time(u64 t)
    {
      for (u64 i=0; i<N; i++) elem[i].set_time(t);
    }

    void set_location(u64 l)
    {
      for (u64 i=0; i<N; i++) elem[i].set_location(l);
    }

    auto concat(const vtlinfo &vtl) const
    {
      // element 0 is at rightmost position
      static_assert(N!=0);
      auto [v,t,l] = vtl;
      u64 y = 0;
      for (i64 i=N-1; i>=0; i--) {
        y = (y<<T::size) | v[i];
      }
      return val<N*T::size> {y,t,l};
    }

    template<u64 W>
    auto make_array(const vtlinfo &vtl) const
    {
      static_assert(std::unsigned_integral<rawtype>);
      static_assert(N!=0);
      static_assert(W!=0 && W<=64);
      constexpr u64 NBITS = T::size*N;
      constexpr u64 M = (NBITS+W-1)/W;
      auto [v,t,l] = vtl;
      auto a = pack_bits<T::size>(v);
      auto aa = unpack_bits<W>(a);
      static_assert(aa.size()>=M);
      return arr<val<W>,M> {[&](u64 i){return val<W>{aa[i],t,l};}};
    }

    template<valtype U>
    auto shift_left(U && x, const vtlinfo &vtl) const
    {
      static_assert(std::unsigned_integral<rawtype>);
      static_assert(std::unsigned_integral<base<U>>);
      static_assert(N!=0);
      static_assert(valt<U>::size!=0);
      auto [v,t,l] = vtl;
      auto [vx,tx] = std::forward<U>(x).get_vt();
      auto a = pack_bits<T::size>(v);
      if constexpr (valt<U>::size == 64) {
        for (u64 i=a.size()-1; i!=0; i--) a[i] = a[i-1];
        a[0] = vx;
      } else {
        static_assert(valt<U>::size<64);
        for (u64 i=a.size()-1; i!=0; i--) {
          a[i] = (a[i] << x.size) | (a[i-1] >> (64-x.size));
        }
        a[0] = (a[0] << x.size) | (vx & ((u64(1)<<x.size)-1));
      }
      auto aa = unpack_bits<T::size>(a);
      static_assert(aa.size()>=N);
      return arr<valt<T>,N> {[&](u64 i){return valt<T>{aa[i],std::max(t,tx),l};}};
    }

    template<valtype U>
    auto shift_right(U && x, const vtlinfo &vtl) const
    {
      static_assert(std::unsigned_integral<rawtype>);
      static_assert(std::unsigned_integral<base<U>>);
      static_assert(N!=0);
      static_assert(valt<U>::size!=0);
      auto [v,t,l] = vtl;
      auto [vx,tx] = std::forward<U>(x).get_vt();
      auto a = pack_bits<T::size>(v);
      if constexpr (valt<U>::size == 64) {
        for (u64 i=0; i<a.size()-1; i++) a[i] = a[i+1];
        a[a.size()-1] = 0;
      } else {
        static_assert(valt<U>::size<64);
        for (u64 i=0; i<a.size()-1; i++) {
          a[i] = (a[i] >> x.size) | (a[i+1] << (64-x.size));
        }
        a[a.size()-1] >>= x.size;
      }
      u64 shiftin = (x.size==64)? vx : vx & ((u64(1)<<x.size)-1);
      u64 k = T::size*N - x.size;
      u64 j = k/64;
      u64 pos = k%64;
      a[j] |= shiftin << pos;
      if (pos+x.size > 64) {
        a[j+1] = shiftin >> (64-pos);
      }
      auto aa = unpack_bits<T::size>(a);
      static_assert(aa.size()>=N);
      return arr<valt<T>,N> {[&](u64 i){return valt<T>{aa[i],std::max(t,tx),l};}};
    }

    valt<T> fold_xor(const vtlinfo &vtl) const
    {
      static_assert(ival<T>);
      static_assert(N>=2);
      constexpr circuit c = XOR<N> * T::size;
      auto [v,t,l] = vtl;
      panel.update_logic(l,c);
      rawtype x = 0;
      for (auto e : v) x ^= e;
      return {x,t+c.delay(),l};
    }

    valt<T> fold_or(const vtlinfo &vtl) const
    {
      static_assert(ival<T>);
      static_assert(N>=2);
      constexpr circuit c = OR<N> * T::size;
      auto [v,t,l] = vtl;
      panel.update_logic(l,c);
      rawtype x = 0;
      for (auto e : v) x |= e;
      return {x,t+c.delay(),l};
    }

    valt<T> fold_and(const vtlinfo &vtl) const
    {
      static_assert(ival<T>);
      static_assert(N>=2);
      constexpr circuit c = AND<N> * T::size;
      auto [v,t,l] = vtl;
      panel.update_logic(l,c);
      rawtype x = -1;
      for (auto e : v) x &= e;
      return {x,t+c.delay(),l};
    }

    auto fold_add(const vtlinfo &vtl) const
    {
      static_assert(ival<T>);
      static_assert(N>=2);
      constexpr u64 RBITS = T::size + std::bit_width(N-1); // output bits
      constexpr circuit c = ADDN<T::size,N>;
      auto [v,t,l] = vtl;
      panel.update_logic(l,c);
      rawtype x = 0;
      for (auto e : v) x += e;
      return val<RBITS,rawtype>{x,t+c.delay(),l};
    }

    valt<T> fold_nor(const vtlinfo &vtl) const
    {
      static_assert(ival<T>);
      static_assert(N>=2);
      constexpr circuit c = NOR<N> * T::size;
      auto [v,t,l] = vtl;
      panel.update_logic(l,c);
      rawtype x = 0;
      for (auto e : v) x |= e;
      return {~x,t+c.delay(),l};
    }

    valt<T> fold_nand(const vtlinfo &vtl) const
    {
      static_assert(ival<T>);
      static_assert(N>=2);
      constexpr circuit c = NAND<N> * T::size;
      auto [v,t,l] = vtl;
      panel.update_logic(l,c);
      rawtype x = -1;
      for (auto e : v) x &= e;
      return {~x,t+c.delay(),l};
    }

    valt<T> fold_xnor(const vtlinfo &vtl) const
    {
      static_assert(ival<T>);
      static_assert(N>=2);
      constexpr circuit c = XNOR<N> * T::size;
      auto [v,t,l] = vtl;
      panel.update_logic(l,c);
      rawtype x = 0;
      for (auto e : v) x ^= e;
      return {~x,t+c.delay(),l};
    }

  public:

    arr() {}

    arr(arr &x)
    {
      copy_from(x);
    }

    template<std::convertible_to<T> ...U>
    arr(U && ...args) : elem{std::forward<U>(args)...} {}

    arr(unaryfunc<u64,T> auto f)
    {
      panel.arr_of_regs_ctor = true;
      for (u64 i=0; i<N; i++) {
        elem[i] = f(i);
      }
      panel.arr_of_regs_ctor = false;
    }

    arr(unaryfunc<void,T> auto f)
    {
      panel.arr_of_regs_ctor = true;
      for (u64 i=0; i<N; i++) {
        elem[i] = f();
      }
      panel.arr_of_regs_ctor = false;
    }

    template<arraylike U>
    arr(U && a)
    {
      static_assert(arraysize<U> == N,"array size mismatch");
      panel.arr_of_regs_ctor = true;
      if constexpr (std::is_rvalue_reference_v<decltype(a)>) {
        for (u64 i=0; i<N; i++) elem[i] = std::move(a[i]);
      } else {
        for (u64 i=0; i<N; i++) elem[i] = a[i];
      }
      panel.arr_of_regs_ctor = false;
    }

    template<arrtype U>
    arr(U && x)
    {
      panel.arr_of_regs_ctor = true;
      copy_from(std::forward<U>(x));
      panel.arr_of_regs_ctor = false;
    }

    void operator= (arr &x) requires (regtype<T>)
    {
      copy_from(x);
    }

    template<arrtype U> requires (regtype<T>)
    void operator= (U && x)
    {
      copy_from(std::forward<U>(x));
    }

    operator T() & requires (N==1) // lvalue
    {
      return elem[0];
    }

    operator T() && requires (N==1) // rvalue
    {
      return elem[0].fo1();
    }

    T& operator[] (u64 i)
    {
      if (i>=N) {
        std::cerr << "array bound exceeded (" << i << ">=" << N << ")" << std::endl;
        std::terminate();
      }
      return elem[i];
    }

    template<valtype U>
    [[nodiscard]] valt<T> select(U && index)
    {
      // only for reading
      // NB: we do not bother providing an rvalue version
      static_assert(ival<U>,"index must be an integer");
      static_assert(N!=0);
      if constexpr (N>=2) {
        static_assert(valt<U>::size<=std::bit_width(N-1),"array index has too many bits");
        constexpr auto c = MUX<N,T::size>;
        auto [i,ti] = std::forward<U>(index).get_vt();
        if (i>=N) {
          std::cerr << "array bound exceeded (" << i << ">=" << N << ")" << std::endl;
          std::terminate();
        }
        // all array elements participate, even those that are not selected
        auto [v,t,l] = proxy::get_vtl(elem); // lvalue
        panel.update_logic(l,c[0]); // MUX select
        panel.update_logic(l,c[1]); // MUX data
        t = std::max(t,ti+proxy::connect_delay(index,l)+c[0].delay()) + c[1].delay();
        return {v[i],t,l};
      } else {
        static_assert(N==1);
        return elem[0];
      }
    }

    template<std::integral auto FO>
    void fanout(hard<FO>) & // lvalue
    {
      static_assert(FO>=2);
      for (auto &e : elem) e.fanout(hard<FO>{});
    }

    [[nodiscard]] arr&& fo1() & // lvalue
    {
      static_assert(! regtype<T>,"cannot apply fo1() to registers");
      return std::move(*this);
    }

    void fanout(auto) && = delete; // rvalue, not needed

    void print(std::string before="", std::string after="\n", bool all=true, std::ostream & os=std::cout) const
    {
      for (u64 i=0; i<N; i++) elem[i].print(before+std::to_string(i)+": ",after,all,os);
    }

    void printb(std::string before="", std::string after="\n", bool all=true, std::ostream & os=std::cout) const
    {
      for (u64 i=0; i<N; i++) elem[i].printb(before+std::to_string(i)+": ",after,all,os);
    }

    [[nodiscard]] auto concat() & requires std::unsigned_integral<rawtype> // lvalue
    {
      return concat(proxy::get_vtl(elem));
    }

    [[nodiscard]] auto concat() && requires std::unsigned_integral<rawtype> // rvalue
    {
      return concat(proxy::get_vtl(std::move(elem)));
    }

    template<std::convertible_to<valt<T>> U>
    [[nodiscard]] auto append(U && x) & // lvalue
    {
      return arr<valt<T>,N+1> {
        [&](u64 i) -> valt<T> {
          if (i==N) {
            return std::forward<U>(x);
          } else {
            return elem[i];
          }
        }
      };
    }

    template<std::convertible_to<valt<T>> U>
    [[nodiscard]] auto append(U && x) && // rvalue
    {
      return arr<valt<T>,N+1> {
        [&](u64 i) -> valt<T> {
          if (i==N) {
            return std::forward<U>(x);
          } else {
            return elem[i].fo1();
          }
        }
      };
    }

    template<std::integral auto L>
    [[nodiscard]] auto truncate(hard<L>) & // lvalue
    {
      static_assert(L<N,"truncate means making the array shorter");
      return arr<valt<T>,L> {
        [&](u64 i) -> valt<T> {
          return elem[i];
        }
      };
    }

    template<std::integral auto L>
    [[nodiscard]] auto truncate(hard<L>) && // rvalue
    {
      static_assert(L<N,"truncate means making the array shorter");
      return arr<valt<T>,L> {
        [&](u64 i) -> valt<T> {
          return elem[i].fo1();
        }
      };
    }

    template<u64 W>
    [[nodiscard]] auto make_array(val<W>&&) & // lvalue
    {
      // FIXME: this is not a reduction
      return make_array<W>(proxy::get_vtl(elem));
    }

    template<u64 W>
    [[nodiscard]] auto make_array(val<W>&&) && // rvalue
    {
      // FIXME: this is not a reduction
      return make_array<W>(proxy::get_vtl(std::move(elem)));
    }

    template<valtype U>
    [[nodiscard]] auto shift_left(U && x) & // lvalue
    {
      // FIXME: this is not a reduction
      return shift_left(std::forward<U>(x),proxy::get_vtl(elem));
    }

    template<valtype U>
    [[nodiscard]] auto shift_left(U && x) && // rvalue
    {
      // FIXME: this is not a reduction
      return shift_left(std::forward<U>(x),proxy::get_vtl(std::move(elem)));
    }

    template<valtype U>
    [[nodiscard]] auto shift_right(U && x) & // lvalue
    {
      // FIXME: this is not a reduction
      return shift_right(std::forward<U>(x),proxy::get_vtl(elem));
    }

    template<valtype U>
    [[nodiscard]] auto shift_right(U && x) && // rvalue
    {
      // FIXME: this is not a reduction
      return shift_right(std::forward<U>(x),proxy::get_vtl(std::move(elem)));
    }

    [[nodiscard]] valt<T> fold_xor() & // lvalue
    {
      if constexpr (N>=2) {
        return fold_xor(proxy::get_vtl(elem));
      } else if constexpr (N==1) {
        return elem[0];
      } else {
        return 0;
      }
    }

    [[nodiscard]] valt<T> fold_xor() && // rvalue
    {
      if constexpr (N>=2) {
        return fold_xor(proxy::get_vtl(std::move(elem)));
      } else if constexpr (N==1) {
        return elem[0].fo1();
      } else {
        return 0;
      }
    }

    [[nodiscard]] valt<T> fold_or() & // lvalue
    {
      if constexpr (N>=2) {
        return fold_or(proxy::get_vtl(elem));
      } else if constexpr (N==1) {
        return elem[0];
      } else {
        return 0;
      }
    }

    [[nodiscard]] valt<T> fold_or() && // rvalue
    {
      if constexpr (N>=2) {
        return fold_or(proxy::get_vtl(std::move(elem)));
      } else if constexpr (N==1) {
        return elem[0].fo1();
      } else {
        return 0;
      }
    }

    [[nodiscard]] valt<T> fold_and() & // lvalue
    {
      if constexpr (N>=2) {
        return fold_and(proxy::get_vtl(elem));
      } else if constexpr (N==1) {
        return elem[0];
      } else {
        return 0;
      }
    }

    [[nodiscard]] valt<T> fold_and() && // rvalue
    {
      if constexpr (N>=2) {
        return fold_and(proxy::get_vtl(std::move(elem)));
      } else if constexpr (N==1) {
        return elem[0].fo1();
      } else {
        return 0;
      }
    }

    [[nodiscard]] auto fold_add() & // lvalue
    {
      if constexpr (N>=2) {
        return fold_add(proxy::get_vtl(elem));
      } else if constexpr (N==1) {
        return valt<T>{elem[0]};
      } else {
        return valt<T>{0};
      }
    }

    [[nodiscard]] auto fold_add() && // rvalue
    {
      if constexpr (N>=2) {
        return fold_add(proxy::get_vtl(std::move(elem)));
      } else if constexpr (N==1) {
        return valt<T>{elem[0].fo1()};
      } else {
        return valt<T>{0};
      }
    }

    [[nodiscard]] valt<T> fold_nor() & // lvalue
    {
      if constexpr (N>=2) {
        return fold_nor(proxy::get_vtl(elem));
      } else if constexpr (N==1) {
        return ~elem[0];
      } else {
        return 0;
      }
    }

    [[nodiscard]] valt<T> fold_nor() && // rvalue
    {
      if constexpr (N>=2) {
        return fold_nor(proxy::get_vtl(std::move(elem)));
      } else if constexpr (N==1) {
        return ~elem[0].fo1();
      } else {
        return 0;
      }
    }

    [[nodiscard]] valt<T> fold_nand() & // lvalue
    {
      if constexpr (N>=2) {
        return fold_nand(proxy::get_vtl(elem));
      } else if constexpr (N==1) {
        return ~elem[0];
      } else {
        return 0;
      }
    }

    [[nodiscard]] valt<T> fold_nand() && // rvalue
    {
      if constexpr (N>=2) {
        return fold_nand(proxy::get_vtl(std::move(elem)));
      } else if constexpr (N==1) {
        return ~elem[0].fo1();
      } else {
        return 0;
      }
    }

    [[nodiscard]] valt<T> fold_xnor() & // lvalue
    {
      if constexpr (N>=2) {
        return fold_xnor(proxy::get_vtl(elem));
      } else if constexpr (N==1) {
        return ~elem[0];
      } else {
        return 0;
      }
    }

    [[nodiscard]] valt<T> fold_xnor() && // rvalue
    {
      if constexpr (N>=2) {
        return fold_xnor(proxy::get_vtl(std::move(elem)));
      } else if constexpr (N==1) {
        return ~elem[0].fo1();
      } else {
        return 0;
      }
    }
  };


  // ######################################################

  template<memdatatype T>
  struct rawdata {};

  template<memdatatype T> requires valtype<T>
  struct rawdata<T> {
    using type = T::type;
    static constexpr u64 width = T::size;
  };

  template<memdatatype T> requires arrtype<T>
  struct rawdata<T> {
    using type = std::array<typename T::type::type,T::size>;
    static constexpr u64 width = T::nbits;
  };


  template<memdatatype T, u64 N>
  class ram {
    template<u64,arith> friend class val;
  public:
    using type = T;
    using valuetype = rawdata<T>::type;
    using static_ram = sram<N,rawdata<T>::width>;

  private:
    std::array<valuetype,N> data {};
    u64 last_read_cycle = 0;
    u64 last_write_cycle = 0;
    u64 last_access_cycle = 0;
    rectangle rect;

    struct writeop {
      u64 addr{};
      valuetype dataval{};
      u64 t{};

      bool operator< (const writeop &rhs) {return t > rhs.t;}

      void commit(ram &mem) const
      {
        assert(addr<N);
        mem.data[addr] = dataval;
      }
    };

    std::vector<writeop> writes; // pending writes

    ram(const ram&) = delete;
    ram& operator= (const ram&) = delete;
    void operator& () = delete;

    u64 ram_id() const
    {
      return rect.id;
    }

    auto coord() const
    {
      return rect.coord_f64();
    }

  public:

    ram(std::string label="") : rect(panel.rams.size(),static_ram::WIDTH,static_ram::HEIGHT,label)
    {
      if (panel.storage_destroyed) {
        std::cerr << "all storage (reg,ram) must have the same lifetime" << std::endl;
        std::terminate();
      }
      //static_ram::print();
      panel.update_storage(static_ram::NBITS,true);
      panel.update_xtors(static_ram::XTORS,static_ram::FINS);
      panel.update_sram_mm2(static_ram::area_mm2());
      panel.add_ram(rect);
    }

    ram(const char *label) : ram{std::string{label}} {}

    ~ram()
    {
      panel.storage_destroyed = true;
      while (! writes.empty()) {
        writes[0].commit(*this);
        std::pop_heap(writes.begin(),writes.end());
        writes.pop_back();
      }
    }

    template<ival TYPEA, std::convertible_to<T> TYPED>
    void write(TYPEA && address, TYPED && dataval)
    {
      panel.check_floorplan();
      const u64 ramid = ram_id();
      if (panel.cycle <= last_write_cycle) {
        std::cerr << "single RAM write per cycle" << std::endl;
        std::terminate();
      }
      // counts as a write even if execution is gated by execute_if
      // (otherwise this would allow free MUXing at the address and data ports)
      last_write_cycle = panel.cycle;
      auto [va,ta,la] = proxy::get_vtl(std::forward<TYPEA>(address));
      if constexpr (valtype<TYPEA>) {
        ta += panel.connect_delay(la,ramid,address.size);
      }
      if (is_less(va,0) || is_greater_equal(va,N)) {
        std::cerr << "out-of-bounds RAM write (N=" << N << "; addr=" << va << ")" << std::endl;
        std::terminate();
      }
      auto [vd,td,ld] = proxy::get_vtl(std::forward<TYPED>(dataval));
      if constexpr (valtype<TYPED>) {
        td += panel.connect_delay(ld,ramid,dataval.size);
      } else if constexpr (arrtype<TYPED>) {
        td += panel.connect_delay(ld,ramid,dataval.nbits);
      }
      if (exec.active) {
#ifndef READ_WRITE_RAM
        if (panel.cycle <= last_access_cycle) {
          std::cerr << "single RAM access per cycle" << std::endl;
          std::terminate();
        }
#endif
        last_access_cycle = panel.cycle;
        panel.update_energy(ramid,static_ram::EWRITE);
        writes.push_back({u64(va),valuetype(vd),std::max(ta,td)});
        std::push_heap(writes.begin(),writes.end());
      }
    }

    template<ival U>
    [[nodiscard]] T read(U && address)
    {
      panel.check_floorplan();
      const u64 ramid = ram_id();
      if (panel.cycle <= last_read_cycle) {
        std::cerr << "single RAM read per cycle" << std::endl;
        std::terminate();
      }
      // counts as a read even if execution is gated by execute_if
      // (otherwise this would allow free MUXing at the address port and free DEMUXing at the data port)
      last_read_cycle = panel.cycle;
      if (exec.active) {
#ifndef READ_WRITE_RAM
        if (panel.cycle <= last_access_cycle) {
          std::cerr << "single RAM access per cycle" << std::endl;
          std::terminate();
        }
#endif
        last_access_cycle = panel.cycle;
        panel.update_energy(ramid,static_ram::EREAD);
      }
      auto [va,ta,la] = proxy::get_vtl(std::forward<U>(address));
      if constexpr (valtype<U>) {
        ta += panel.connect_delay(la,ramid,address.size);
      }
      while (! writes.empty() && writes[0].t <= ta) {
        writes[0].commit(*this);
        std::pop_heap(writes.begin(),writes.end());
        writes.pop_back();
      }
      u64 t = ta + myllround(static_ram::LATENCY); // time at which the read completes
      if (is_less(va,0) || is_greater_equal(va,N)) {
        std::cerr << "out-of-bounds RAM read (N=" << N << "; addr=" << va << ")" << std::endl;
        std::terminate();
      }
      T readval = (exec.active)? T{data[va]} : T{};
      readval.set_time(t);
      readval.set_location(ramid);
      return readval;
    }

    void reset()
    {
      if (! exec.active) return;
      panel.update_energy(rect.id,static_ram::NBITS*2*INV.e);
      for (auto &v : data) v = {}; // instantaneous (FIXME?)
      writes.clear();
    }

    void print(std::string s = "", std::ostream & os = std::cout)
    {
      static_ram::print(s,os);
    }
  };


  // ######################################################


  template<u64 N, u64 M>
  class rom_content {
    template<valtype,u64> friend class rom;
    using T = std::bitset<M>;
    std::array<T,N> data;

    template<std::convertible_to<T> U>
    constexpr rom_content(std::initializer_list<U> l)
    {
      if (l.size() != N) {
        std::cerr << "the list size must match the ROM size" << std::endl;
        std::terminate();
      }
      std::copy(l.begin(),l.end(),data.begin());
    }

    template<std::convertible_to<T> U>
    constexpr rom_content(const U (&a)[N])
    {
      std::copy(a,a+N,data.begin());
    }

    template<std::convertible_to<T> U>
    constexpr rom_content(const std::array<U,N> &a)
    {
      std::copy(a.begin(),a.end(),data.begin());
    }

    constexpr rom_content(unaryfunc<u64,T> auto f)
    {
      for (u64 i=0; i<N; i++) {
        data[i] = f(i);
      }
    }

    auto operator[] (u64 i) const
    {
      if (i>=N) {
        std::cerr << "out-of-bounds ROM read (" << i << ">=" << N << ")" << std::endl;
        std::terminate();
      }
      return data[i].to_ullong();
    }
  };


  template<valtype T, u64 N>
  class rom {
    static_assert(! regtype<T>);
    static_assert(N!=0);

    const rom_content<N,T::size> content;
    const circuit hw;

  public:

    rom() = delete;

    template<std::convertible_to<T> U>
    constexpr rom(std::initializer_list<U> l) : content(l), hw(ROM(content.data)) {}

    template<std::convertible_to<T> U>
    constexpr rom(const U (&a)[N]) : content(a), hw(ROM(content.data)) {}

    template<std::convertible_to<T> U>
    constexpr rom(const std::array<U,N> &a) : content(a), hw(ROM(content.data)) {}

    constexpr rom(unaryfunc<u64,T> auto f) : content(f), hw(ROM(content.data)) {}

    template<ival U>
    [[nodiscard]] T operator() (U && address) const
    {
      auto [v,t,l] = proxy::get_vtl(std::forward<U>(address));
      proxy::update_logic(l,hw);
      return {content[v], t+hw.delay(), l};
    }
  };


  // ######################################################

  // EQUALITY
  template<valtype T1, valtype T2>
  val<1> operator== (T1 && x1, T2 && x2)
  {
    static_assert(valt<T1>::size == valt<T2>::size);
    constexpr circuit c = EQUAL<valt<T1>::size>;
    auto [v1,v2,t,l] = proxy::get_vtl(std::forward<T1>(x1),std::forward<T2>(x2));
    proxy::update_logic(l,c);
    return proxy::make_val<val<1>>(is_equal(v1,v2), t+c.delay(), l);
  }

  template<valtype T1, arith T2> // second argument is a constant
  val<1> operator== (T1 && x1, T2 x2)
  {
    constexpr circuit reduc = NOR<valt<T1>::size>;
    const circuit c = INV * ones<valt<T1>::size>(x2) + reduc; // not constexpr
    auto [v1,t1,l1] = proxy::get_vtl(std::forward<T1>(x1));
    proxy::update_logic(l1,c);
    return proxy::make_val<val<1>>(is_equal(v1,x2), t1+c.delay(), l1);
  }

  template<valtype T1, hardval T2> // second argument is a hard constant
  val<1> operator== (T1 && x1, T2 x2)
  {
    constexpr circuit reduc = NOR<valt<T1>::size>;
    constexpr circuit c = INV * ones<valt<T1>::size>(x2.value) + reduc;
    auto [v1,t1,l1] = proxy::get_vtl(std::forward<T1>(x1));
    proxy::update_logic(l1,c);
    return proxy::make_val<val<1>>(is_equal(v1,x2.value), t1+c.delay(), l1);
  }

  template<arithlike T1, valtype T2> // first argument is a constant
  val<1> operator== (T1 x1, T2 && x2)
  {
    return std::forward<T2>(x2) == x1;
  }

  // INEQUALITY
  template<valtype T1, valtype T2>
  val<1> operator!= (T1 && x1, T2 && x2)
  {
    static_assert(valt<T1>::size == valt<T2>::size);
    constexpr circuit c = NEQ<valt<T1>::size>;
    auto [v1,v2,t,l] = proxy::get_vtl(std::forward<T1>(x1),std::forward<T2>(x2));
    proxy::update_logic(l,c);
    return proxy::make_val<val<1>>(is_different(v1,v2), t+c.delay(), l);
  }

  template<valtype T1, arith T2> // second argument is a constant
  val<1> operator!= (T1 && x1, T2 x2)
  {
    constexpr circuit reduc = OR<valt<T1>::size>;
    const circuit c = INV * ones<valt<T1>::size>(x2) + reduc; // not constexpr
    auto [v1,t1,l1] = proxy::get_vtl(std::forward<T1>(x1));
    proxy::update_logic(l1,c);
    return proxy::make_val<val<1>>(is_different(v1,x2), t1+c.delay(), l1);
  }

  template<valtype T1, hardval T2> // second argument is a hard constant
  val<1> operator!= (T1 && x1, T2 x2)
  {
    constexpr circuit reduc = OR<valt<T1>::size>;
    constexpr circuit c = INV * ones<valt<T1>::size>(x2.value) + reduc;
    auto [v1,t1,l1] = proxy::get_vtl(std::forward<T1>(x1));
    proxy::update_logic(l1,c);
    return proxy::make_val<val<1>>(is_different(v1,x2.value), t1+c.delay(), l1);
  }

  template<arithlike T1, valtype T2> // first argument is a constant
  val<1> operator!= (T1 x1, T2 && x2)
  {
    return std::forward<T2>(x2) != x1;
  }

  // GREATER THAN
  template<valtype T1, valtype T2> requires (ival<T1> && ival<T2>)
  val<1> operator> (T1 && x1, T2 && x2)
  {
    static_assert(valt<T1>::size == valt<T2>::size);
    constexpr circuit c = GT<valt<T1>::size>;
    auto [v1,v2,t,l] = proxy::get_vtl(std::forward<T1>(x1),std::forward<T2>(x2));
    proxy::update_logic(l,c);
    return proxy::make_val<val<1>>(is_greater(v1,v2), t+c.delay(), l);
  }

  template<valtype T1, std::integral T2> requires (ival<T1>) // second argument is constant
  val<1> operator> (T1 && x1, T2 x2)
  {
    constexpr u64 N = valt<T1>::size;
    static_assert(N!=0);
    constexpr circuit comp = GT<N>;
    constexpr circuit comp0 = (N==1)? circuit{} : (std::signed_integral<base<T1>>)? NOR<N-1> + NOR<2> : OR<N>;
    const circuit &c = (x2==0)? comp0 : comp; // not constexpr
    auto [v1,t1,l1] = proxy::get_vtl(std::forward<T1>(x1));
    proxy::update_logic(l1,c);
    return proxy::make_val<val<1>>(is_greater(v1,x2), t1+c.delay(), l1);
  }

  template<valtype T1, hardval T2> requires (ival<T1> && intlike<T2>) // second argument is hard constant
  val<1> operator> (T1 && x1, T2 x2)
  {
    constexpr u64 N = valt<T1>::size;
    static_assert(N!=0);
    constexpr circuit comp = GT<N>;
    constexpr circuit comp0 = (N==1)? circuit{} : (std::signed_integral<base<T1>>)? NOR<N-1> + NOR<2> : OR<N>;
    constexpr circuit c = (x2.value==0)? comp0 : comp; // TODO: specialize more
    auto [v1,t1,l1] = proxy::get_vtl(std::forward<T1>(x1));
    proxy::update_logic(l1,c);
    return proxy::make_val<val<1>>(is_greater(v1,x2.value), t1+c.delay(), l1);
  }

  template<intlike T1, valtype T2> requires (ival<T2>) // first argument is constant
  val<1> operator> (T1 x1, T2 && x2)
  {
    return std::forward<T2>(x2) < x1;
  }

  // LESS THAN
  template<valtype T1, valtype T2> requires (ival<T1> && ival<T2>)
  val<1> operator< (T1 && x1, T2 && x2)
  {
    static_assert(valt<T1>::size == valt<T2>::size);
    constexpr circuit c = GT<valt<T1>::size>;
    auto [v1,v2,t,l] = proxy::get_vtl(std::forward<T1>(x1),std::forward<T2>(x2));
    proxy::update_logic(l,c);
    return proxy::make_val<val<1>>(is_less(v1,v2), t+c.delay(), l);
  }

  template<valtype T1, std::integral T2> requires (ival<T1>) // second argument is constant
  val<1> operator< (T1 && x1, T2 x2)
  {
    constexpr circuit comp = GT<valt<T1>::size>;
    constexpr circuit comp0;
    const circuit &c = (x2==0)? comp0 : comp; // not constexpr
    auto [v1,t1,l1] = proxy::get_vtl(std::forward<T1>(x1));
    proxy::update_logic(l1,c);
    return proxy::make_val<val<1>>(is_less(v1,x2), t1+c.delay(), l1);
  }

  template<valtype T1, hardval T2> requires (ival<T1> && intlike<T2>) // second argument is hard constant
  val<1> operator< (T1 && x1, T2 x2)
  {
    constexpr circuit comp = GT<valt<T1>::size>;
    constexpr circuit comp0;
    constexpr circuit c = (x2.value==0)? comp0 : comp; // TODO: specialize more
    auto [v1,t1,l1] = proxy::get_vtl(std::forward<T1>(x1));
    proxy::update_logic(l1,c);
    return proxy::make_val<val<1>>(is_less(v1,x2.value), t1+c.delay(), l1);
  }

  template<intlike T1, valtype T2> requires (ival<T2>) // first argument is constant
  val<1> operator< (T1 x1, T2 && x2)
  {
    return std::forward<T2>(x2) > x1;
  }

  // GREATER THAN OR EQUAL
  template<valtype T1, valtype T2> requires (ival<T1> && ival<T2>)
  val<1> operator>= (T1 && x1, T2 && x2)
  {
    static_assert(valt<T1>::size == valt<T2>::size);
    constexpr circuit c = GTE<valt<T1>::size>;
    auto [v1,v2,t,l] = proxy::get_vtl(std::forward<T1>(x1),std::forward<T2>(x2));
    proxy::update_logic(l,c);
    return proxy::make_val<val<1>>(is_greater_equal(v1,v2), t+c.delay(), l);
  }

  template<valtype T1, std::integral T2> requires (ival<T1>) // second argument is constant
  val<1> operator>= (T1 && x1, T2 x2)
  {
    constexpr circuit comp = GTE<valt<T1>::size>;
    constexpr circuit comp0 = (std::signed_integral<base<T1>>)? INV : circuit{};
    const circuit &c = (x2==0)? comp0 : comp; // not constexpr
    auto [v1,t1,l1] = proxy::get_vtl(std::forward<T1>(x1));
    proxy::update_logic(l1,c);
    return proxy::make_val<val<1>>(is_greater_equal(v1,x2), t1+c.delay(), l1);
  }

  template<valtype T1, hardval T2> requires (ival<T1> && intlike<T2>) // second argument is hard constant
  val<1> operator>= (T1 && x1, T2 x2)
  {
    constexpr circuit comp = GTE<valt<T1>::size>;
    constexpr circuit comp0 = (std::signed_integral<base<T1>>)? INV : circuit{};
    constexpr circuit c = (x2.value==0)? comp0 : comp; // TODO: specialize more
    auto [v1,t1,l1] = proxy::get_vtl(std::forward<T1>(x1));
    proxy::update_logic(l1,c);
    return proxy::make_val<val<1>>(is_greater_equal(v1,x2.value), t1+c.delay(), l1);
  }

  template<intlike T1, valtype T2> requires (ival<T2>) // first argument is constant
  val<1> operator>= (T1 x1, T2 && x2)
  {
    return std::forward<T2>(x2) <= x1;
  }

  // LESS THAN OR EQUAL
  template<valtype T1, valtype T2> requires (ival<T1> && ival<T2>)
  val<1> operator<= (T1 && x1, T2 && x2)
  {
    static_assert(valt<T1>::size == valt<T2>::size);
    constexpr circuit c = GTE<valt<T1>::size>;
    auto [v1,v2,t,l] = proxy::get_vtl(std::forward<T1>(x1),std::forward<T2>(x2));
    proxy::update_logic(l,c);
    return proxy::make_val<val<1>>(is_less_equal(v1,v2), t+c.delay(), l);
  }

  template<valtype T1, std::integral T2> requires (ival<T1>) // second argument is constant
  val<1> operator<= (T1 && x1, T2 x2)
  {
    constexpr u64 N = valt<T1>::size;
    static_assert(N!=0);
    constexpr circuit comp = GTE<N>;
    constexpr circuit comp0 = (N==1)? circuit{} : (std::signed_integral<base<T1>>)? NOR<N-1> + OR<2> : NOR<N>;
    const circuit &c = (x2==0)? comp0 : comp; // not constexpr
    auto [v1,t1,l1] = proxy::get_vtl(std::forward<T1>(x1));
    proxy::update_logic(l1,c);
    return proxy::make_val<val<1>>(is_less_equal(v1,x2), t1+c.delay(), l1);
  }

  template<valtype T1, hardval T2> requires (ival<T1> && intlike<T2>) // second argument is hard constant
  val<1> operator<= (T1 && x1, T2 x2)
  {
    constexpr u64 N = valt<T1>::size;
    static_assert(N!=0);
    constexpr circuit comp = GTE<N>;
    constexpr circuit comp0 = (N==1)? circuit{} : (std::signed_integral<base<T1>>)? NOR<N-1> + OR<2> : NOR<N>;
    constexpr circuit c = (x2.value==0)? comp0 : comp; // TODO: specialize more
    auto [v1,t1,l1] = proxy::get_vtl(std::forward<T1>(x1));
    proxy::update_logic(l1,c);
    return proxy::make_val<val<1>>(is_less_equal(v1,x2.value), t1+c.delay(), l1);
  }

  template<intlike T1, valtype T2> requires (ival<T2>) // first argument is constant
  val<1> operator<= (T1 x1, T2 && x2)
  {
    return std::forward<T2>(x2) >= x1;
  }

  // ADDITION
  template<valtype T1, valtype T2> requires (ival<T1> && ival<T2>)
  auto operator+ (T1 && x1, T2 && x2)
  {
    constexpr circuit c = (valt<T1>::size==1)? INC<valt<T2>::size> : (valt<T2>::size==1)? INC<valt<T1>::size> : ADD<valt<T1,T2>::size>;
    auto [v1,v2,t,l] = proxy::get_vtl(std::forward<T1>(x1),std::forward<T2>(x2));
    proxy::update_logic(l,c);
    using rtype = val<std::min(val<64>::size,valt<T1,T2>::size+1),decltype(v1+v2)>;
    return proxy::make_val<rtype>(v1+v2, t+c.delay(), l);
  }

  template<valtype T1, intlike T2> requires (ival<T1>) // 2nd arg constant
  auto operator+ (T1 && x1, T2 x2)
  {
    auto [v1,t1,l1] = proxy::get_vtl(std::forward<T1>(x1));
    using rtype = val<std::min(val<64>::size,valt<T1>::size+1),decltype(v1+x2)>;
    if (x2==0) return proxy::make_val<rtype>(v1,t1,l1);
    constexpr circuit c = INC<valt<T1>::size>; // TODO: specialize more
    proxy::update_logic(l1,c);
    return proxy::make_val<rtype>(v1+x2, t1+c.delay(), l1);
  }

  template<intlike T1, valtype T2> requires (ival<T2>) // 1st arg constant
  auto operator+ (T1 x1, T2 && x2)
  {
    return std::forward<T2>(x2) + x1;
  }

  // CHANGE SIGN
  template<valtype T>
  auto operator- (T && x)
  {
    constexpr circuit c = INV * valt<T>::size + INC<valt<T>::size>;
    auto [v,t,l] = proxy::get_vtl(std::forward<T>(x));
    proxy::update_logic(l,c);
    using rtype = val<valt<T>::size,decltype(-v)>;
    return proxy::make_val<rtype>(-v, t+c.delay(), l);
  }

  // SUBTRACTION
  template<valtype T1, valtype T2> requires (ival<T1> && ival<T2>)
  auto operator- (T1 && x1, T2 && x2)
  {
    constexpr circuit c = SUB<valt<T1,T2>::size>;
    auto [v1,v2,t,l] = proxy::get_vtl(std::forward<T1>(x1),std::forward<T2>(x2));
    proxy::update_logic(l,c);
    using rtype = val<std::min(val<64>::size,valt<T1,T2>::size+1),decltype(v1-v2)>;
    return proxy::make_val<rtype>(v1-v2, t+c.delay(), l);
  }

  template<valtype T1, intlike T2> requires (ival<T1>) // 2nd arg constant
  auto operator- (T1 && x1, T2 x2)
  {
    auto [v1,t1,l1] = proxy::get_vtl(std::forward<T1>(x1));
    using rtype = val<std::min(val<64>::size,valt<T1>::size+1),decltype(v1-x2)>;
    if (x2==0) return proxy::make_val<rtype>(v1,t1,l1);
    constexpr circuit c = INC<valt<T1>::size>; // TODO: specialize more
    proxy::update_logic(l1,c);
    return proxy::make_val<rtype>(v1-x2, t1+c.delay(), l1);
  }

  template<intlike T1, valtype T2> requires (ival<T2>) // 1st arg constant
  auto operator- (T1 x1, T2 && x2)
  {
    auto [v2,t2,l2] = proxy::get_vtl(std::forward<T2>(x2));
    using rtype = val<std::min(val<64>::size,valt<T2>::size+1),decltype(x1-v2)>;
    if (x1==0) return proxy::make_val<rtype>(-v2,t2,l2);
    constexpr circuit c = INC<valt<T2>::size>; // TODO: specialize more
    proxy::update_logic(l2,c);
    return proxy::make_val<rtype>(x1-v2, t2+c.delay(), l2);
  }

  // LEFT SHIFT
  template<valtype T1, intlike T2> requires (ival<T1>) // 2nd arg constant
  auto operator<< (T1 && x1, T2 x2)
  {
    auto [v1,t1,l1] = proxy::get_vtl(std::forward<T1>(x1));
    return proxy::make_val<valt<T1>>(v1<<x2, t1, l1);
  }

  // RIGHT SHIFT
  template<valtype T1, intlike T2> requires (std::unsigned_integral<base<T1>>) // 2nd arg constant
  auto operator>> (T1 && x1, T2 x2)
  {
    auto [v1,t1,l1] = proxy::get_vtl(std::forward<T1>(x1));
    return proxy::make_val<valt<T1>>(v1>>x2, t1, l1);
  }

  template<valtype T1, intlike T2> requires (std::signed_integral<base<T1>>) // 2nd arg constant
  auto operator>> (T1 && x1, T2 x2)
  {
    // the sign bit is replicated
    static_assert(hardval<T2>,"right shift of signed integer: shift amount must be a hard value");
    constexpr circuit c = REP<1,x2+1>; // do not use a buffer here (see comment in REP's definition)
    auto [v1,t1,l1] = proxy::get_vtl(std::forward<T1>(x1));
    proxy::update_logic(l1,c);
    return proxy::make_val<valt<T1>>(v1>>x2, t1+c.delay(), l1);
  }

  // MULTIPLICATION
  template<valtype T1, valtype T2> requires (ival<T1> && ival<T2>)
  auto operator* (T1 && x1, T2 && x2)
  {
    constexpr circuit c = IMUL<valt<T1>::size,valt<T2>::size>;
    constexpr u64 rbits = valt<T1>::size + valt<T2>::size; // result bits
    static_assert(rbits<=val<64>::size,"multiplication result must not exceed 64 bits");
    auto [v1,v2,t,l] = proxy::get_vtl(std::forward<T1>(x1),std::forward<T2>(x2));
    proxy::update_logic(l,c);
    using rtype = val<rbits,decltype(v1*v2)>;
    return proxy::make_val<rtype>(v1*v2, t+c.delay(), l);
  }

  template<valtype T1, intlike T2> requires (ival<T1>) // 2nd argument is constant
  auto operator* (T1 && x1, T2 x2)
  {
    static_assert(hardval<T2>,"constant multiplier must be a hard value (hard<N>{})");
    constexpr u64 u2 = (x2>=0)? x2 : truncate<minbits(x2.value)>(x2.value); // convert x2 to unsigned
    constexpr circuit c = HIMUL<u2,valt<T1>::size>; // TODO: signed multiplication
    auto [v1,t1,l1] = proxy::get_vtl(std::forward<T1>(x1));
    proxy::update_logic(l1,c);
    if constexpr (std::unsigned_integral<decltype(v1*x2)>) {
      using rtype = val<minbits(valt<T1>::maxval*u2),decltype(v1*x2)>;
      return proxy::make_val<rtype>(v1*x2, t1+c.delay(), l1);
    } else {
      static_assert(std::signed_integral<decltype(v1*x2)>);
      constexpr u64 rsize = std::min(minbits(valt<T1>::maxval*x2),minbits(valt<T1>::minval*x2));
      using rtype = val<rsize,decltype(v1*x2)>;
      return proxy::make_val<rtype>(v1*x2, t1+c.delay(), l1);
    }
  }

  template<intlike T1, valtype T2> requires (ival<T2>) // 1st argument is constant
  auto operator* (T1 x1, T2 && x2)
  {
    return std::forward<T2>(x2) * x1;
  }

  // DIVISION
  template<valtype T1, intlike T2> requires (ival<T1>) // constant divisor
  auto operator/ (T1 && x1, T2 x2)
  {
    static_assert(hardval<T2>,"divisor must be a hard constant (hard<N>{})");
    static_assert(std::unsigned_integral<base<T1>>,"signed division not implemented"); // TODO
    static_assert(x2>0,"signed division not implemented"); // TODO
    constexpr circuit c = UDIV<valt<T1>::size,x2>;
    auto [v1,t1,l1] = proxy::get_vtl(std::forward<T1>(x1));
    proxy::update_logic(l1,c);
    constexpr u64 N = std::bit_width(valt<T1>::maxval/x2);
    using rtype = val<std::max(N,u64(1)),decltype(v1/x2)>;
    return proxy::make_val<rtype>(v1/x2, t1+c.delay(), l1);
  }

  // MODULO (REMAINDER)
  template<valtype T1, intlike T2> requires (ival<T1>) // constant divisor
  auto operator% (T1 && x1, T2 x2)
  {
    static_assert(hardval<T2>,"divisor must be a hard constant (hard<N>{})");
    static_assert(std::unsigned_integral<base<T1>>,"dividend must be unsigned");
    static_assert(x2>0,"divisor must be positive");
    constexpr circuit c = UMOD<valt<T1>::size,x2>;
    auto [v1,t1,l1] = proxy::get_vtl(std::forward<T1>(x1));
    proxy::update_logic(l1,c);
    constexpr u64 N = std::bit_width(u64(x2)-1);
    using rtype = val<std::max(N,u64(1)),decltype(v1%x2)>;
    return proxy::make_val<rtype>(v1%x2, t1+c.delay(), l1);
  }

  // BITWISE AND
  template<valtype T1, valtype T2>
  auto operator& (T1 && x1, T2 && x2)
  {
    constexpr circuit c = AND<2> * std::min(valt<T1>::size,valt<T2>::size);
    auto [v1,v2,t,l] = proxy::get_vtl(std::forward<T1>(x1),std::forward<T2>(x2));
    proxy::update_logic(l,c);
    return proxy::make_val<valt<T1,T2>>(v1&v2, t+c.delay(), l);
  }

  template<valtype T1, intlike T2> // second argument is constant
  auto operator& (T1 && x1, T2 x2)
  {
    // no transistors
    auto [v1,t1,l1] = proxy::get_vtl(std::forward<T1>(x1));
    return proxy::make_val<valt<T1>>(v1&x2, t1, l1);
  }

  template<intlike T1, valtype T2> // first argument is constant
  auto operator& (T1 x1, T2 && x2)
  {
    // no transistors
    return std::forward<T2>(x2) & x1;
  }

  // BITWISE OR
  template<valtype T1, valtype T2>
  auto operator| (T1 && x1, T2 && x2)
  {
    constexpr circuit c = OR<2> * std::min(valt<T1>::size,valt<T2>::size);
    auto [v1,v2,t,l] = proxy::get_vtl(std::forward<T1>(x1),std::forward<T2>(x2));
    proxy::update_logic(l,c);
    return proxy::make_val<valt<T1,T2>>(v1|v2, t+c.delay(), l);
  }

  template<valtype T1, intlike T2> // second argument is constant
  auto operator| (T1 && x1, T2 x2)
  {
    // no transistors
    auto [v1,t1,l1] = proxy::get_vtl(std::forward<T1>(x1));
    return proxy::make_val<valt<T1>>(v1|x2, t1, l1);
  }

  template<intlike T1, valtype T2> // first argument is constant
  auto operator| (T1 x1, T2 && x2)
  {
    // no transistors
    return std::forward<T2>(x2) | x1;
  }

  // BITWISE EXCLUSIVE OR
  template<valtype T1, valtype T2>
  auto operator^ (T1 && x1, T2 && x2)
  {
    constexpr circuit c = XOR<2> * std::min(valt<T1>::size,valt<T2>::size);
    auto [v1,v2,t,l] = proxy::get_vtl(std::forward<T1>(x1),std::forward<T2>(x2));
    proxy::update_logic(l,c);
    return proxy::make_val<valt<T1,T2>>(v1^v2, t+c.delay(), l);
  }

  template<valtype T1, std::integral T2> // second argument is constant
  auto operator^ (T1 && x1, T2 x2)
  {
    const circuit c = INV * ones<valt<T1>::size>(x2); // not constexpr
    auto [v1,t1,l1] = proxy::get_vtl(std::forward<T1>(x1));
    proxy::update_logic(l1,c);
    return proxy::make_val<valt<T1>>(v1^x2, t1+c.delay(), l1);
  }

  template<valtype T1, hardval T2> // second argument is hard constant
  auto operator^ (T1 && x1, T2 x2)
  {
    constexpr circuit c = INV * ones<valt<T1>::size>(x2.value);
    auto [v1,t1,l1] = proxy::get_vtl(std::forward<T1>(x1));
    proxy::update_logic(l1,c);
    return proxy::make_val<valt<T1>>(v1^x2, t1+c.delay(), l1);
  }

  template<intlike T1, valtype T2> // first argument is constant
  auto operator^ (T1 x1, T2 && x2)
  {
    return std::forward<T2>(x2) ^ x1;
  }

  // BITWISE COMPLEMENT
  template<valtype T>
  auto operator~ (T && x)
  {
    constexpr circuit c = INV * valt<T>::size;
    auto [v,t,l] = proxy::get_vtl(std::forward<T>(x));
    proxy::update_logic(l,c);
    return proxy::make_val<valt<T>>(~v, t+c.delay(), l);
  }

  // MULTIPLY-ACCUMULATE (A+BxC)
  template<valtype TA, valtype TB, valtype TC> requires (ival<TA> && ival<TB> && ival<TC>)
  [[nodiscard]] auto a_plus_bc(TA && xa, TB && xb, TC && xc)
  {
    constexpr circuit c = IMAD<valt<TA>::size,valt<TB>::size,valt<TC>::size>;
    constexpr u64 bcbits = valt<TB>::size + valt<TC>::size; // bits BxC
    static_assert(bcbits<=val<64>::size,"multiplication result must not exceed 64 bits");
    constexpr u64 apbcbits = std::max(valt<TA>::size,bcbits)+1; // bits A+BxC
    auto [va,vb,vc,t,l] = proxy::get_vtl(std::forward<TA>(xa),std::forward<TB>(xb),std::forward<TC>(xc));
    proxy::update_logic(l,c);
    using rtype = val<std::min(val<64>::size,apbcbits),decltype(va+vb*vc)>;
    return proxy::make_val<rtype>(va+vb*vc, t+c.delay(), l);
  }

  // CONCATENATE BITS
  template<valtype T1, valtype... Ti>
  [[nodiscard]] auto concat(T1 && x1, Ti && ...xi)
  {
    static_assert(std::unsigned_integral<base<T1>>,"concat takes unsigned integers");
    static_assert((std::unsigned_integral<base<Ti>> && ...),"concat takes unsigned integers");
    constexpr u64 N = 1 + sizeof...(xi);
    constexpr std::array si = {valt<T1>::size, valt<Ti>::size...};
    constexpr u64 SIZE = (valt<T1>::size + ... + valt<Ti>::size);
    static_assert(SIZE<=64,"concatenation exceeds 64 bits");
    auto tup = proxy::get_vtl(std::forward<T1>(x1), std::forward<Ti>(xi)...);
    static_assert(std::tuple_size_v<decltype(tup)> == N+2);
    u64 t = std::get<N>(tup);
    u64 l = std::get<N+1>(tup);
    u64 x = 0;
    static_loop<N>([&]<u64 I>() {
        x = (x<<si[I]) | std::get<I>(tup);
      });
    return proxy::make_val<val<SIZE>>(x,t,l);
  }

  // SPLIT BITS (INVERSE OF CONCAT)
  template<u64 N1, u64... Ni, valtype T>
  [[nodiscard]] auto split(T && x)
  {
    return proxy::split_helper<N1,Ni...>(x).tup;
  }

  // SELECT BETWEEN TWO VALUES
  template<valtype T, valtype T1, valtype T2>
  [[nodiscard]] auto select(T && cond, T1 && x1, T2 && x2)
  {
    // this is NOT conditional execution: both sides are evaluated
    static_assert(valt<T>::size == 1,"the condition of a select is a single bit");
    static_assert(valt<T1>::size == valt<T2>::size,"both sides of a select must have the same size");
    constexpr auto c = MUX<2,valt<T1>::size>;
    auto [vc,tc,lc] = proxy::get_vtl(std::forward<T>(cond));
    auto [v1,v2,t,l] = proxy::get_vtl(std::forward<T1>(x1),std::forward<T2>(x2));
    proxy::update_logic(l,c[0]); // MUX select
    proxy::update_logic(l,c[1]); // MUX data
    tc += proxy::connect_delay(cond,l) + c[0].delay();
    t = std::max(t,tc) + c[1].delay();
    if (bool(vc)) {
      return proxy::make_val<valt<T1,T2>>(v1,t,l);
    } else {
      return proxy::make_val<valt<T1,T2>>(v2,t,l);
    }
  }

  // CONDITIONAL EXECUTION
  template<valtype T, action A> requires (std::same_as<return_type<A>,void>)
  void execute_if(T && mask, const A &f)
  {
    // the delay/energy for gating the execution is not modeled (TODO?)
    static_assert(std::unsigned_integral<base<T>>);
    constexpr u64 N = valt<T>::size;
    auto prev_exec = exec;
    // nesting execute_ifs increases the predicate's delay
    auto prev_cond = exec.to_val().replicate(hard<N>{}).concat();
    auto cond_mask = prev_cond.fo1() & std::forward<T>(mask);
    auto [vm,tm,lm] = proxy::get_vtl(cond_mask.fo1());
    for (u64 i=0; i<N; i++) {
      bool cond = (vm>>i) & 1;
      exec.set_state(cond,tm,lm);
      // we execute the action even when the condition is false
      // (otherwise, this primitive could be used to leak any bit)
      if constexpr (std::invocable<A>) {
        f();
      } else {
        static_assert(std::invocable<A,u64>);
        f(i);
      }
    }
    exec = prev_exec;
  }

  template<valtype T, action A>
  [[nodiscard]] auto execute_if(T && mask, const A &f)
  {
    // the delay/energy for gating the execution is not modeled (TODO?)
    static_assert(valtype<return_type<A>>);
    static_assert(std::unsigned_integral<base<T>>);
    constexpr u64 N = valt<T>::size;
    using rtype = valt<return_type<A>>;
    // return 0 if the condition is false (AND gate)
    constexpr circuit gate = AND<2> * rtype::size;
    constexpr circuit buf = buffer(AND<2>.ci*rtype::size,false);
    constexpr circuit out = buf + gate;
    auto prev_exec = exec;
    // nesting execute_ifs increases the predicate's delay
    auto prev_cond = exec.to_val().replicate(hard<N>{}).concat();
    auto cond_mask = prev_cond.fo1() & std::forward<T>(mask);
    auto [vm,tm,lm] = proxy::get_vtl(cond_mask.fo1());
    arr<rtype,N> result;
    for (u64 i=0; i<N; i++) {
      bool cond = (vm>>i) & 1;
      exec.set_state(cond,tm,lm);
      // we execute the action even when the condition is false
      // (otherwise, this primitive could be used to leak any bit)
      if constexpr (std::invocable<A>) {
        result[i] = f();
      } else {
        static_assert(std::invocable<A,u64>);
        result[i] = f(i);
      }
      result[i].set_time(std::max(result[i].time(),tm+buf.delay())+gate.delay());
      proxy::update_logic(result[i].site(),out);
    };
    exec = prev_exec;
    return result;
  }


  // ######################################################
  // UTILITIES
  // below are functions that users could write themselves

  template<u64 N, typename T>
  [[nodiscard]] val<N> absolute_value(val<N,T> x)
  {
    static_assert(std::signed_integral<T>,"unsigned value is always positive");
    x.fanout(hard<2>{});
    return select(x < hard<0>{},-x,x);
  }

  // encode(x) takes a one-hot value and returns the index of the hot bit
  // if the input is not a one-hot value, the result is meaningless

  template<u64 N>
  [[nodiscard]] auto encode(val<N> x)
  {
    static_assert(N!=0);
    constexpr u64 W = std::bit_width(N-1);
    auto xbits = x.fo1().make_array(val<1>{});
    arr<val<W>,N> a = [&](u64 i){return i & xbits[i].replicate(hard<W>{}).concat();};
    return a.fo1().fold_or();
  }

  // fold(x,op) takes an array x, a 2-input function op doing a binary associative operation
  // and reduces the array to a single value

  template<arith T, u64 W, u64 N>
  [[nodiscard]] val<W,T> fold(arr<val<W,T>,N> x, auto op)
  {
    static_assert(N!=0);
    if constexpr (N==1) {
      return x[0].fo1();
    }
    arr<val<W,T>,(N+1)/2> y = [&] (u64 i) -> val<W,T> {
      if (2*i+1 < N) {
        return op(x[2*i].fo1(),x[2*i+1].fo1());
      } else {
        return x[2*i].fo1();
      }
    };
    return fold(y.fo1(),op);
  }

  template<arith T, u64 W, u64 N>
  [[nodiscard]] val<W,T> fold(arr<reg<W,T>,N> &x, auto op)
  {
    return fold(arr<val<W,T>,N>{x},op);
  }

  // scan(x,op) takes an array x={x1,x2,...,xn}, a 2-input function op doing a binary associative operation
  // and returns an array y={y1,y2,...,yn} with y1=x1, y2=op(y1,x2), y3=op(y2,x3),...
  // AKA inclusive scan AKA parallel prefix

  template<arith T, u64 W, u64 N>
  [[nodiscard]] arr<val<W,T>,N> scan(arr<val<W,T>,N> x, auto op, u64 step=1)
  {
    // Kogge-Stone tree
    static_assert(N!=0);
    x.fanout(hard<2>{});
    arr<val<W,T>,N> y = [&] (u64 i) -> val<W,T> {
      if (i >= step) {
        return op(x[i],x[i-step]);
      } else {
        return x[i];
      }
    };
    if (step >= std::bit_floor(N-1)) {
      return y.fo1();
    } else {
      return scan(y.fo1(),op,step*2);
    }
  }

  template<arith T, u64 W, u64 N>
  [[nodiscard]] arr<val<W,T>,N> scan(arr<reg<W,T>,N> &x, auto op)
  {
    return scan(arr<val<W,T>,N>{x},op);
  }

}

#endif // HARCOM_H
