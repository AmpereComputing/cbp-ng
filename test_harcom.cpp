#include "harcom.hpp"

using namespace hcm;


#define EXEC(...)                               \
  std::cout << #__VA_ARGS__ << std::endl;       \
  __VA_ARGS__;


class harcom_superuser {
public:
  void next_cycle() {panel.next_cycle();}
  auto get(valtype auto x) {return x.get();}
} hsu;



template<typename TX, typename TY>
void binary_operators(TX x, TY y)
{
  (x==y).print("x==y:","\n",false);
  (x!=y).print("x!=y:","\n",false);
  (x>y).print("x>y:","\n",false);
  (x<y).print("x<y:","\n",false);
  (x>=y).print("x>=y:","\n",false);
  (x<=y).print("x<=y:","\n",false);
  (x+y).print("x+y=","\n",false);
  (x-y).print("x-y=","\n",false);
  if constexpr ((valtype<TX> || hardval<TX>) && (valtype<TY> || hardval<TY>))
    (x*y).print("x*y=","\n",false);
  if constexpr (valtype<TX> && hardval<TY>)
    (x/y).print("x/y=","\n",false);
  if constexpr (valtype<TX> && hardval<TY>)
    (x%y).print("x%y=","\n",false);
  (x&y).printb("x&y=","\n",false);
  (x|y).printb("x|y=","\n",false);
  (x^y).printb("x^y=","\n\n",false);
}


template<valtype T>
void other_operators(T x)
{
  (-x).print("-x=","\n",false);
  if constexpr (std::unsigned_integral<base<T>>)
    (x>>1).printb("x>>1=","\n",false);
  (x>>hard<1>{}).printb("x>>hard<1>{}=","\n",false);
  (x<<1).printb("x<<1=","\n",false);
  (x<<hard<1>{}).printb("x<<hard<1>{}=","\n",false);
  (~x).printb("~x=","\n\n",false);
}


int main()
{
  {
    EXEC(val<4> x=15);
    EXEC(val<4,u64> y=2);
    binary_operators(x,y);
  }
  {
    EXEC(val<4> x=15);
    EXEC(u64 y=2);
    binary_operators(x.fo1(),y);
  }
  {
    EXEC(val<4> x=15);
    EXEC(hard<2> y);
    binary_operators(x,y);
  }
  {
    EXEC(u64 x=2);
    EXEC(val<4> y=15);
    binary_operators(x,y);
  }
  {
    EXEC(hard<2> x);
    EXEC(val<4> y=15);
    binary_operators(x,y.fo1());
  }
  {
    EXEC(val<4,i64> x=-1);
    other_operators(x);
  }
  {
    EXEC(val<4,i64> x=5);
    EXEC(val<4,i64> y=-2);
    EXEC(val<4,i64> z=4);
    EXEC(a_plus_bc(x,y.fo1(),z).print("x+yz=","\n\n",false));
  }
  {
    EXEC(val<12> xyz=0b000111000111);
    EXEC(auto [x,y,z] = split<4,3,5>(xyz));
    x.printb("x=","\n",false);
    y.printb("y=","\n",false);
    z.printb("z=","\n",false);
    EXEC(concat(x,y,z.fo1()).printb("concat(x,y,z)=","\n\n",false));
  }
  {
    EXEC(val<1> c=1);
    EXEC(val<4> x=3);
    EXEC(val<4> y=4);
    EXEC(select(c,x,y).print("select(c,x,y)=","\n\n",false));
  }
  EXEC(reg<4> r1;);
  {
    EXEC(execute_if(r1==0,[&](){r1=3;}));
    r1.print("r1=","\n\n",false);
  }
  {
    EXEC(val<4> x=11);
    EXEC(arr<val<8>,4> pp = execute_if(x,[&](u64 i){return val<8>{x}<<i;}));
    EXEC(pp.fold_add().print("x*x=","\n\n",false));
  }
  {
    EXEC(val<12> x=0b111100000010);
    EXEC(val<12> y=x.fo1());
    y.printb("y=","\n",false);
    EXEC(y.fanout(hard<4>{}));
    EXEC(y.make_array(val<4>{}).printb("","\n",false));
    EXEC(y.reverse().printb("reverse: ","\n",false));
    EXEC(y.rotate_left(1).printb("rotate left: ","\n",false));
    EXEC(y.rotate_left(-1).printb("rotate right: ","\n",false));
    EXEC(y.one_hot().printb("one hot: ","\n",false));
    EXEC(y.replicate(hard<3>{}).printb("","\n",false));
    EXEC(y.ones().print("bit count: ","\n",false));
    EXEC(val<4> z=3;);
    EXEC(z.decode().concat().printb("","\n\n",false));
  }
  {
    hsu.next_cycle();
    EXEC(r1 = r1+1);
    r1.print("r1=","\n\n",false);
  }
  EXEC(int b[]={3,2,1});
  EXEC(arr<reg<4>,3> B = b);
  B.print("B","\n",false);
  {
    EXEC(arr<val<4>,3> A = [](u64 i){return (1<<(i+1))-1;});
    EXEC(B=A.fo1());
    B.printb("B","\n",false);
    EXEC(B.fanout(hard<4>{}));
    EXEC(B.select(val<2>{1}).printb("B[1]=","\n",false));
    EXEC(B.concat().printb("concat: ","\n",false));
    EXEC(B.append(-B[0]).printb("","\n",false));
    EXEC(B.truncate(hard<2>{}).printb("","\n",false));
    EXEC(B.make_array(val<3>{}).printb("","\n",false));
    EXEC(B.shift_left(val<1>{1}).printb("","\n",false));
    EXEC(B.shift_right(val<1>{1}).printb("","\n",false));
    EXEC(B.fold_or().printb("or: ","\n",false));
    EXEC(B.fold_and().printb("and: ","\n",false));
    EXEC(B.fold_xor().printb("xor: ","\n",false));
    EXEC(B.fold_nor().printb("nor: ","\n",false));
    EXEC(B.fold_nand().printb("nand: ","\n",false));
    EXEC(B.fold_xnor().printb("xnor: ","\n",false));
    EXEC(B.fold_add().print("add: ","\n\n",false));
  }
  {
    EXEC(rom<val<3>,16> bitcount = [](u64 i){return std::popcount(i);});
    EXEC(bitcount(val<4>{14}).print("bit count: ","\n\n",false));
  }
  EXEC(ram<val<4>,1024> M1);
  EXEC(ram<arr<val<4>,3>,1024> M2);
  EXEC(ram<val<4>,256> M3[4]);
  panel.make_floorplan();
  {
    EXEC(val<10> x=31);
    EXEC(M1.write(x,13));
    hsu.next_cycle();
    EXEC(M2.write(x*hard<31>{},M1.read(x).replicate(hard<3>{})));
    hsu.next_cycle();
    EXEC(M2.read(x*x).print("","\n",false));
    hsu.next_cycle();
    EXEC((M1.read(x)+M2.read(x*x)[1]).print());
    hsu.next_cycle();
    EXEC((M1.read(x)+M2.read(x*x)[1].connect(M1)).print());
    EXEC(val<4>{5}.distribute(M3).print());
    EXEC(M2.reset());
    hsu.next_cycle();
    EXEC(M2.read(x*x).concat().printb("","\n\n",false));
  }
  {
    EXEC(val<8,i64> x=-100);
    EXEC(absolute_value(x).print("|x|=","\n\n",false));
  }
  {
    EXEC(val<8> x=0b10010000);
    EXEC(encode(x.one_hot()).print("rightmost=","\n\n",false));
  }
  {
    EXEC(auto max = [](val<4> x,val<4> y){return select(x>y,x,y);});
    EXEC(arr<val<4>,4> A = {8,2,13,7});
    EXEC(fold(A,max).print("max=","\n\n",false));
  }
  {
    EXEC(auto add = [](val<7> x,val<7> y){return x+y;});
    EXEC(arr<val<7>,8> A = {10,3,4,9,2,3,1,11});
    EXEC(scan(A,add).print("","  ",false));
    std::cout << "\n\n";
  }
  {
    EXEC(static_loop<10>([]<int I>(){std::cout<<I;}););
    std::cout << "\n";
  }
  {
#ifdef CHEATING_MODE
    std::cout << "\n";
    EXEC(val<4> x=3);
    EXEC(assert(x==3));
#endif
  }
}
