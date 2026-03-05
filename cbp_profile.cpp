// cbp_profile.cpp
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>
#include <charconv>
#include <streambuf>
#include <chrono>

#include "trace_reader.hpp"
#include "harcom.hpp"
#include "cbp.hpp"
#include "branch_predictor.hpp"

using namespace hcm;

namespace cbpp {

// ========================== CLI / Modes ==========================
enum class RunMode { ACCURACY, ANALYZE };
static RunMode g_mode = RunMode::ANALYZE;
static bool g_profile = false;        // enable per-function profiling in analyze mode
static bool g_print_score = true;     // print score breakdown (captures CSV line)
static bool g_host_timing = false;    // host wall-clock timing per function (optional)

// ========================== Fixed-buffer val.print() ==========================
namespace fastio {

struct fixed_buf : std::streambuf {
  static constexpr size_t CAP = 128;
  char data[CAP];

  fixed_buf() { reset(); }
  void reset() { setp(data, data + CAP - 1); }

  std::string_view view() {
    *pptr() = '\0';
    return std::string_view(data, (size_t)(pptr() - pbase()));
  }

  int overflow(int ch) override {
    if (ch == EOF) return 0;
    if (pptr() == epptr()) return ch; // drop extra
    *pptr() = (char)ch;
    pbump(1);
    return ch;
  }
};

struct fixed_ostream {
  fixed_buf buf;
  std::ostream os;
  fixed_ostream() : os(&buf) {}
  void reset() { buf.reset(); }
  std::string_view view() { return buf.view(); }
};

static thread_local fixed_ostream F;

template <typename V>
static inline bool print_val(const V& x, bool show_timing) {
  F.reset();
  if constexpr (requires { x.print("", "", show_timing, F.os); }) {
    x.print("", "", show_timing, F.os);
    return true;
  } else if constexpr (requires { x.print("", "\n", show_timing, F.os); }) {
    x.print("", "\n", show_timing, F.os);
    return true;
  } else {
    return false;
  }
}

static inline bool parse_leading_bit(std::string_view s, bool& ok) {
  ok = false;
  size_t i = 0;
  while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) i++;
  if (i >= s.size()) return false;

  if (s[i] == '0') { ok = true; return false; }
  if (s[i] == '1') { ok = true; return true; }

  size_t j = i;
  while (j < s.size() && (s[j] >= '0' && s[j] <= '9')) j++;
  if (j == i) return false;

  uint64_t v = 0;
  auto res = std::from_chars(s.data() + i, s.data() + j, v);
  if (res.ec != std::errc()) return false;
  ok = true;
  return (v != 0);
}

static inline uint64_t parse_t_ps(std::string_view s) {
  size_t tpos = s.find('t');
  if (tpos == std::string_view::npos) return 0;
  size_t j = tpos;
  while (j < s.size() && !(s[j] >= '0' && s[j] <= '9')) j++;
  size_t t0 = j;
  while (j < s.size() && (s[j] >= '0' && s[j] <= '9')) j++;
  if (j == t0) return 0;
  uint64_t t = 0;
  auto res = std::from_chars(s.data() + t0, s.data() + j, t);
  if (res.ec != std::errc()) return 0;
  return t;
}

template <auto N>
static inline bool bit_of_val(const val<N>& x, bool show_timing, uint64_t* t_ps_out, bool& ok) {
  ok = false;
  if (!print_val(x, show_timing)) return false;
  auto sv = F.view();
  bool bit = parse_leading_bit(sv, ok);
  if (t_ps_out) *t_ps_out = show_timing ? parse_t_ps(sv) : 0;
  return bit;
}

} // namespace fastio

// ========================== CSV helpers ==========================
namespace csv {

static inline std::string last_nonempty_line(const std::string& s) {
  size_t end = s.size();
  while (end && (s[end - 1] == '\n' || s[end - 1] == '\r')) end--;
  if (!end) return {};
  size_t start = s.rfind('\n', end - 1);
  start = (start == std::string::npos) ? 0 : start + 1;
  return s.substr(start, end - start);
}

static inline std::vector<std::string_view> split(const std::string& line) {
  std::vector<std::string_view> out;
  std::string_view sv(line);
  size_t i = 0;
  while (i <= sv.size()) {
    size_t j = sv.find(',', i);
    if (j == std::string_view::npos) j = sv.size();
    out.emplace_back(sv.substr(i, j - i));
    i = j + 1;
    if (j == sv.size()) break;
  }
  return out;
}

static inline bool parse_u64(std::string_view sv, uint64_t& out) {
  while (!sv.empty() && (sv.front() == ' ' || sv.front() == '\t')) sv.remove_prefix(1);
  while (!sv.empty() && (sv.back() == ' ' || sv.back() == '\t')) sv.remove_suffix(1);
  if (sv.empty()) return false;
  uint64_t v = 0;
  auto res = std::from_chars(sv.data(), sv.data() + sv.size(), v);
  if (res.ec != std::errc()) return false;
  out = v;
  return true;
}

static inline bool parse_f64(std::string_view sv, double& out) {
  while (!sv.empty() && (sv.front() == ' ' || sv.front() == '\t')) sv.remove_prefix(1);
  while (!sv.empty() && (sv.back() == ' ' || sv.back() == '\t')) sv.remove_suffix(1);
  if (sv.empty()) return false;
  try { out = std::stod(std::string(sv)); return true; } catch (...) { return false; }
}

struct RunCSV {
  std::string name;
  uint64_t instr=0, branch=0, condbr=0, nblock=0;
  uint64_t extra=0, nshort=0, ncoincide=0, nlong=0;
  double p1_lat=0.0, p2_lat=0.0;
  double epi_fJ=0.0;
};

static inline RunCSV parse(const std::string& line) {
  RunCSV r;
  auto f = split(line);
  if (f.size() != 12) return r;

  r.name = std::string(f[0]);
  (void)parse_u64(f[1], r.instr);
  (void)parse_u64(f[2], r.branch);
  (void)parse_u64(f[3], r.condbr);
  (void)parse_u64(f[4], r.nblock);
  (void)parse_u64(f[5], r.extra);
  (void)parse_u64(f[6], r.nshort);
  (void)parse_u64(f[7], r.ncoincide);
  (void)parse_u64(f[8], r.nlong);
  (void)parse_f64(f[9], r.p1_lat);
  (void)parse_f64(f[10], r.p2_lat);
  (void)parse_f64(f[11], r.epi_fJ);
  return r;
}

} // namespace csv

// ========================== Score breakdown (predictor_metrics.py for one trace) ==========================
static inline void print_score_breakdown(std::ostream& os, const csv::RunCSV& r) {
  if (!r.instr) return;
  constexpr int misprediction_penalty = 8;
  const int L1 = (int)std::ceil(r.p1_lat);
  const int L2 = (int)std::ceil(r.p2_lat);

  double Tcp = 0.0;
  if (L2 <= L1) {
    Tcp = (double)r.nblock * (double)std::max(1, L2);
  } else {
    Tcp = (double)r.nblock * (double)std::max(1, L1)
        + (double)r.nshort * (double)L2
        - (double)r.ncoincide * (double)std::max(1, L1);
  }
  Tcp += (double)r.extra;

  const double IPCcbp = (double)r.instr / Tcp;
  const double MPI    = (double)r.nlong / (double)r.instr;
  const double CPIcbp = MPI * (double)(misprediction_penalty + L2);
  const double EPIcbp = r.epi_fJ;

  const double WPI = IPCcbp * CPIcbp;
  const double IPC = 1.0 / (1.0 / IPCcbp + CPIcbp);

  os << "\n=== Score breakdown (from CBP CSV counters) ===\n";
  os << "Trace: " << r.name << "\n";
  os << "Ncp(instr)=" << r.instr
     << "  Nblock=" << r.nblock
     << "  AvgBlockLen=" << std::fixed << std::setprecision(2) << (double)r.instr / (double)r.nblock
     << "  Textra=" << r.extra << "\n";
  os << "CondBr=" << r.condbr
     << "  Nshort=" << r.nshort
     << "  Ncoincide=" << r.ncoincide
     << "  Nlong=" << r.nlong
     << "  LongRate=" << std::fixed << std::setprecision(3) << (100.0*(double)r.nlong/(double)r.condbr) << "%\n";
  os << "Latency(max): p1=" << std::fixed << std::setprecision(3) << r.p1_lat
     << "cy  p2=" << r.p2_lat << "cy  =>  L1=" << L1 << "  L2=" << L2 << "\n";
  os << "IPCcbp=" << std::fixed << std::setprecision(6) << IPCcbp
     << "  CPIcbp=" << std::fixed << std::setprecision(6) << CPIcbp
     << "  EPIcbp=" << std::fixed << std::setprecision(1) << EPIcbp << " fJ/instr\n";
  os << "Derived: MPI=" << std::fixed << std::setprecision(6) << MPI
     << "  WPI=" << std::fixed << std::setprecision(6) << WPI
     << "  IPC=" << std::fixed << std::setprecision(6) << IPC << "\n";
  os << "=== end ===\n";
}

// ========================== ACC mode: exact P1/P2 conditional accuracy ==========================
namespace acc {

static uint64_t warmup_instr = 0;
static bool warmed_up = false;
static uint64_t ninstr = 0;

static bool last_p1_valid = false;
static bool last_p2_valid = false;
static bool last_p1_bit = false;
static bool last_p2_bit = false;

static uint64_t cond = 0;
static uint64_t p1_ok = 0;
static uint64_t p2_ok = 0;

static inline void reset_stats() {
  cond = 0; p1_ok = 0; p2_ok = 0;
  last_p1_valid = last_p2_valid = false;
}

static inline void on_instruction() {
  ninstr++;
  if (!warmed_up && ninstr > warmup_instr) {
    warmed_up = true;
    reset_stats();
    ninstr = 0;
  }
}

static inline bool collecting() { return warmed_up; }

static inline void record_p1(const val<1>& out) {
  if (!collecting()) return;
  bool ok=false;
  last_p1_bit = fastio::bit_of_val(out, false, nullptr, ok);
  last_p1_valid = ok;
}

static inline void record_p2(const val<1>& out) {
  if (!collecting()) return;
  bool ok=false;
  last_p2_bit = fastio::bit_of_val(out, false, nullptr, ok);
  last_p2_valid = ok;
}

static inline void update_on_condbr(const val<1>& taken) {
  if (!collecting()) return;
  bool ok=false;
  bool gt = fastio::bit_of_val(taken, false, nullptr, ok);
  if (!ok) return;
  if (!last_p1_valid || !last_p2_valid) return;

  cond++;
  if (last_p1_bit == gt) p1_ok++;
  if (last_p2_bit == gt) p2_ok++;
}

static inline void print_accuracy(std::ostream& os) {
  double p1_acc = cond ? (double)p1_ok / (double)cond : 0.0;
  double p2_acc = cond ? (double)p2_ok / (double)cond : 0.0;

  os << "\n=== Accuracy (measurement window, conditional branches) ===\n";
  os << "P1_cond_acc=" << std::fixed << std::setprecision(6) << p1_acc
     << " (" << p1_ok << "/" << cond << ")\n";
  os << "P2_cond_acc=" << std::fixed << std::setprecision(6) << p2_acc
     << " (" << p2_ok << "/" << cond << ")\n";
  os << "=== end ===\n";
}

} // namespace acc

template <class P>
struct acc_predictor final : public P {
  val<1> predict1(val<64> pc) override {
    acc::on_instruction();
    val<1> out = P::predict1(pc);
    acc::record_p1(out);
    return out;
  }
  val<1> reuse_predict1(val<64> pc) override {
    acc::on_instruction();
    val<1> out = P::reuse_predict1(pc);
    acc::record_p1(out);
    return out;
  }
  val<1> predict2(val<64> pc) override {
    val<1> out = P::predict2(pc);
    acc::record_p2(out);
    return out;
  }
  val<1> reuse_predict2(val<64> pc) override {
    val<1> out = P::reuse_predict2(pc);
    acc::record_p2(out);
    return out;
  }
  void update_condbr(val<64> pc, val<1> taken, val<64> next_pc) override {
    P::update_condbr(pc, taken, next_pc);
    acc::update_on_condbr(taken);
  }
  void update_cycle(instruction_info& info) override { P::update_cycle(info); }
};

// ========================== ANALYZE mode: per-function avg energy + latency ==========================
namespace prof {

static bool enable = false;
static uint64_t warmup_instr = 0;
static bool warmed_up = false;
static uint64_t ninstr = 0;

// CBP time model tracking
static uint64_t time_ps = 0;
static uint64_t next_time_ps = 0;

enum class Fn : uint8_t { PRED1, RPRED1, PRED2, RPRED2, UPD_CONDBR, UPD_CYCLE, N };

static Fn   last_p1_fn = Fn::PRED1;
static Fn   last_p2_fn = Fn::PRED2;
static bool last_p1_bit = false;
static bool last_p2_bit = false;

static inline uint64_t now_ns() {
  return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Stats {
  uint64_t calls = 0;

  double   e_sum_fJ = 0.0;
  double   e_max_fJ = 0.0;

  // model latency from val timing (predict/reuse only)
  uint64_t lat_ps_sum = 0;
  uint64_t lat_ps_max = 0;
  uint64_t lat_count  = 0;

  // conditional accuracy attribution (predict/reuse only)
  uint64_t cond = 0;
  uint64_t cond_ok = 0;

  // optional host timing for all functions
  uint64_t host_ns_sum = 0;
  uint64_t host_ns_max = 0;
};

static std::array<Stats, (size_t)Fn::N> S{};

static inline const char* name(Fn f) {
  switch (f) {
    case Fn::PRED1: return "predict1";
    case Fn::RPRED1: return "reuse_predict1";
    case Fn::PRED2: return "predict2";
    case Fn::RPRED2: return "reuse_predict2";
    case Fn::UPD_CONDBR: return "update_condbr";
    case Fn::UPD_CYCLE: return "update_cycle";
    default: return "unknown";
  }
}

static inline void reset_stats() { for (auto& x : S) x = Stats{}; }

static inline void on_instruction() {
  ninstr++;
  if (!warmed_up && ninstr > warmup_instr) {
    warmed_up = true;
    reset_stats();
    ninstr = 0;
  }
}

static inline bool collecting() { return enable && warmed_up; }

static inline void add_energy(Fn f, double de) {
  if (!collecting()) return;
  auto& st = S[(size_t)f];
  st.calls++;
  st.e_sum_fJ += de;
  st.e_max_fJ = std::max(st.e_max_fJ, de);
}

static inline void add_latency(Fn f, uint64_t out_t_ps) {
  if (!collecting()) return;
  auto& st = S[(size_t)f];
  uint64_t lat_ps = (out_t_ps > time_ps) ? (out_t_ps - time_ps) : 0;
  st.lat_ps_sum += lat_ps;
  st.lat_ps_max = std::max(st.lat_ps_max, lat_ps);
  st.lat_count++;
}

static inline void add_acc(Fn f, bool correct) {
  if (!collecting()) return;
  auto& st = S[(size_t)f];
  st.cond++;
  if (correct) st.cond_ok++;
}

static inline void add_host(Fn f, uint64_t ns) {
  if (!collecting()) return;
  auto& st = S[(size_t)f];
  st.host_ns_sum += ns;
  st.host_ns_max = std::max(st.host_ns_max, ns);
}

static inline uint64_t measured_instr() {
  return S[(size_t)Fn::PRED1].calls + S[(size_t)Fn::RPRED1].calls;
}
static inline uint64_t measured_blocks() { return S[(size_t)Fn::UPD_CYCLE].calls; }
static inline uint64_t measured_condbr() { return S[(size_t)Fn::UPD_CONDBR].calls; }

static inline double total_energy() {
  double tot = 0.0;
  for (auto& st : S) tot += st.e_sum_fJ;
  return tot;
}

static void dump(std::ostream& os) {
  const uint64_t instr  = measured_instr();
  const uint64_t blocks = measured_blocks();
  const uint64_t condbr = measured_condbr();

  const double Etot = total_energy();
  const double EPI  = instr ? (Etot / (double)instr) : 0.0;
  const double avg_blk_len = blocks ? (double)instr / (double)blocks : 0.0;
  const double reuse_rate  = instr ? (double)S[(size_t)Fn::RPRED1].calls / (double)instr : 0.0;

  // aggregate P1/P2 acc across predict+reuse
  const uint64_t p1_cond = S[(size_t)Fn::PRED1].cond + S[(size_t)Fn::RPRED1].cond;
  const uint64_t p1_ok   = S[(size_t)Fn::PRED1].cond_ok + S[(size_t)Fn::RPRED1].cond_ok;
  const uint64_t p2_cond = S[(size_t)Fn::PRED2].cond + S[(size_t)Fn::RPRED2].cond;
  const uint64_t p2_ok   = S[(size_t)Fn::PRED2].cond_ok + S[(size_t)Fn::RPRED2].cond_ok;

  const double p1_acc = p1_cond ? (double)p1_ok / (double)p1_cond : 0.0;
  const double p2_acc = p2_cond ? (double)p2_ok / (double)p2_cond : 0.0;

  os << "\n=== Analyze: per-function avg energy + latency (measurement window) ===\n";
  os << "Instr=" << instr
     << "  Blocks=" << blocks
     << "  CondBr=" << condbr
     << "  AvgBlockLen=" << std::fixed << std::setprecision(2) << avg_blk_len
     << "  ReuseRate=" << std::fixed << std::setprecision(1) << (reuse_rate * 100.0) << "%\n";
  os << "EPI(from per-call deltas)=" << std::fixed << std::setprecision(1) << EPI
     << " fJ/instr | CondAcc(P1)=" << std::fixed << std::setprecision(4) << p1_acc
     << " CondAcc(P2)=" << std::fixed << std::setprecision(4) << p2_acc << "\n\n";

  // Table header
  os << std::left  << std::setw(16) << "Function"
     << std::right << std::setw(12) << "Calls"
     << std::setw(14) << "E_avg(fJ)"
     << std::setw(14) << "E_sum(fJ)"
     << std::setw(12) << "LatAvg(cy)"
     << std::setw(12) << "LatMax(cy)";
  if (cbpp::g_host_timing) {
    os << std::setw(14) << "HostAvg(us)"
       << std::setw(14) << "HostMax(us)";
  }
  os << std::setw(18) << "CondAcc(ok/total)"
     << "\n";

  os << std::string(16+12+14+14+12+12 + (cbpp::g_host_timing?28:0) + 18, '-') << "\n";

  for (size_t i = 0; i < (size_t)Fn::N; i++) {
    const auto& st = S[i];
    const bool is_pred = (i <= (size_t)Fn::RPRED2);

    const double e_avg = st.calls ? st.e_sum_fJ / (double)st.calls : 0.0;

    double lat_avg_cy = 0.0, lat_max_cy = 0.0;
    if (is_pred && st.lat_count) {
      const double lat_avg_ps = (double)st.lat_ps_sum / (double)st.lat_count;
      lat_avg_cy = lat_avg_ps / (double)cycle_ps;
      lat_max_cy = (double)st.lat_ps_max / (double)cycle_ps;
    }

    os << std::left  << std::setw(16) << name((Fn)i)
       << std::right << std::setw(12) << st.calls
       << std::setw(14) << std::fixed << std::setprecision(3) << e_avg
       << std::setw(14) << std::fixed << std::setprecision(1) << st.e_sum_fJ;

    if (is_pred) {
      os << std::setw(12) << std::fixed << std::setprecision(5) << lat_avg_cy
         << std::setw(12) << std::fixed << std::setprecision(5) << lat_max_cy;
    } else {
      os << std::setw(12) << "-" << std::setw(12) << "-";
    }

    if (cbpp::g_host_timing) {
      double host_avg_us = st.calls ? ((double)st.host_ns_sum / (double)st.calls) / 1000.0 : 0.0;
      double host_max_us = (double)st.host_ns_max / 1000.0;
      os << std::setw(14) << std::fixed << std::setprecision(3) << host_avg_us
         << std::setw(14) << std::fixed << std::setprecision(3) << host_max_us;
    }

    if (is_pred && st.cond) {
      double acc = (double)st.cond_ok / (double)st.cond;
      std::ostringstream accs;
      accs << std::fixed << std::setprecision(4) << acc
           << " (" << st.cond_ok << "/" << st.cond << ")";
      os << std::setw(18) << accs.str();
    } else {
      os << std::setw(18) << "-";
    }
    os << "\n";
  }

  os << "\nNotes:\n"
     << "  • E_avg is average HARCOM energy delta per call (all 6 functions).\n"
     << "  • LatAvg/LatMax are CBP/HARCOM model latencies from val<1> timing (predict/reuse only).\n";
  if (cbpp::g_host_timing) {
    os << "  • HostAvg/HostMax are wall-clock times (useful for optimizing code, not CBP timing model).\n";
  }
  os << "=== end ===\n";
}

} // namespace prof

template <class P>
struct profiled_predictor final : public P {
  // No member state (avoid HARCOM NHC surprises)

  val<1> predict1(val<64> pc) override {
    prof::on_instruction();
    prof::last_p1_fn = prof::Fn::PRED1;

    const bool do_meas = prof::collecting();
    const uint64_t t0 = (cbpp::g_host_timing && do_meas) ? prof::now_ns() : 0;

    double e0 = 0.0;
    if (do_meas) e0 = (double)panel.energy_fJ();

    val<1> out = P::predict1(pc);

    double e1 = 0.0;
    if (do_meas) e1 = (double)panel.energy_fJ();
    if (do_meas) prof::add_energy(prof::Fn::PRED1, e1 - e0);

    // default advance
    prof::next_time_ps = prof::time_ps + cycle_ps;

    bool ok=false;
    uint64_t out_t_ps=0;
    bool bit = fastio::bit_of_val(out, true, &out_t_ps, ok);
    if (ok) {
      prof::last_p1_bit = bit;
      if (do_meas) prof::add_latency(prof::Fn::PRED1, out_t_ps);

      // match cbp.hpp ceil-to-cycles next_time
      uint64_t next_time = prof::time_ps;
      if (out_t_ps > prof::time_ps) {
        uint64_t lat_ps = out_t_ps - prof::time_ps;
        uint64_t lat_cycles = (lat_ps + cycle_ps - 1) / cycle_ps;
        next_time += lat_cycles * cycle_ps;
      } else {
        next_time += cycle_ps;
      }
      prof::next_time_ps = next_time;
    }

    if (cbpp::g_host_timing && do_meas) prof::add_host(prof::Fn::PRED1, prof::now_ns() - t0);
    return out;
  }

  val<1> reuse_predict1(val<64> pc) override {
    prof::on_instruction();
    prof::last_p1_fn = prof::Fn::RPRED1;

    const bool do_meas = prof::collecting();
    const uint64_t t0 = (cbpp::g_host_timing && do_meas) ? prof::now_ns() : 0;

    double e0 = 0.0;
    if (do_meas) e0 = (double)panel.energy_fJ();

    val<1> out = P::reuse_predict1(pc);

    double e1 = 0.0;
    if (do_meas) e1 = (double)panel.energy_fJ();
    if (do_meas) prof::add_energy(prof::Fn::RPRED1, e1 - e0);

    prof::next_time_ps = prof::time_ps + cycle_ps;

    bool ok=false;
    uint64_t out_t_ps=0;
    bool bit = fastio::bit_of_val(out, true, &out_t_ps, ok);
    if (ok) {
      prof::last_p1_bit = bit;
      if (do_meas) prof::add_latency(prof::Fn::RPRED1, out_t_ps);

      uint64_t next_time = prof::time_ps;
      if (out_t_ps > prof::time_ps) {
        uint64_t lat_ps = out_t_ps - prof::time_ps;
        uint64_t lat_cycles = (lat_ps + cycle_ps - 1) / cycle_ps;
        next_time += lat_cycles * cycle_ps;
      } else {
        next_time += cycle_ps;
      }
      prof::next_time_ps = next_time;
    }

    if (cbpp::g_host_timing && do_meas) prof::add_host(prof::Fn::RPRED1, prof::now_ns() - t0);
    return out;
  }

  val<1> predict2(val<64> pc) override {
    prof::last_p2_fn = prof::Fn::PRED2;

    const bool do_meas = prof::collecting();
    const uint64_t t0 = (cbpp::g_host_timing && do_meas) ? prof::now_ns() : 0;

    double e0 = 0.0;
    if (do_meas) e0 = (double)panel.energy_fJ();

    val<1> out = P::predict2(pc);

    double e1 = 0.0;
    if (do_meas) e1 = (double)panel.energy_fJ();
    if (do_meas) prof::add_energy(prof::Fn::PRED2, e1 - e0);

    if (do_meas) {
      bool ok=false;
      uint64_t out_t_ps=0;
      bool bit = fastio::bit_of_val(out, true, &out_t_ps, ok);
      if (ok) {
        prof::last_p2_bit = bit;
        prof::add_latency(prof::Fn::PRED2, out_t_ps);
      }
    }

    if (cbpp::g_host_timing && do_meas) prof::add_host(prof::Fn::PRED2, prof::now_ns() - t0);
    return out;
  }

  val<1> reuse_predict2(val<64> pc) override {
    prof::last_p2_fn = prof::Fn::RPRED2;

    const bool do_meas = prof::collecting();
    const uint64_t t0 = (cbpp::g_host_timing && do_meas) ? prof::now_ns() : 0;

    double e0 = 0.0;
    if (do_meas) e0 = (double)panel.energy_fJ();

    val<1> out = P::reuse_predict2(pc);

    double e1 = 0.0;
    if (do_meas) e1 = (double)panel.energy_fJ();
    if (do_meas) prof::add_energy(prof::Fn::RPRED2, e1 - e0);

    if (do_meas) {
      bool ok=false;
      uint64_t out_t_ps=0;
      bool bit = fastio::bit_of_val(out, true, &out_t_ps, ok);
      if (ok) {
        prof::last_p2_bit = bit;
        prof::add_latency(prof::Fn::RPRED2, out_t_ps);
      }
    }

    if (cbpp::g_host_timing && do_meas) prof::add_host(prof::Fn::RPRED2, prof::now_ns() - t0);
    return out;
  }

  void update_condbr(val<64> pc, val<1> taken, val<64> next_pc) override {
    const bool do_meas = prof::collecting();
    const uint64_t t0 = (cbpp::g_host_timing && do_meas) ? prof::now_ns() : 0;

    double e0 = 0.0;
    if (do_meas) e0 = (double)panel.energy_fJ();

    P::update_condbr(pc, taken, next_pc);

    double e1 = 0.0;
    if (do_meas) e1 = (double)panel.energy_fJ();
    if (do_meas) prof::add_energy(prof::Fn::UPD_CONDBR, e1 - e0);

    if (do_meas) {
      bool ok=false;
      bool gt = fastio::bit_of_val(taken, false, nullptr, ok);
      if (ok) {
        prof::add_acc(prof::last_p1_fn, prof::last_p1_bit == gt);
        prof::add_acc(prof::last_p2_fn, prof::last_p2_bit == gt);
      }
    }

    if (cbpp::g_host_timing && do_meas) prof::add_host(prof::Fn::UPD_CONDBR, prof::now_ns() - t0);
  }

  void update_cycle(instruction_info& info) override {
    const bool do_meas = prof::collecting();
    const uint64_t t0 = (cbpp::g_host_timing && do_meas) ? prof::now_ns() : 0;

    double e0 = 0.0;
    if (do_meas) e0 = (double)panel.energy_fJ();

    P::update_cycle(info);

    double e1 = 0.0;
    if (do_meas) e1 = (double)panel.energy_fJ();
    if (do_meas) prof::add_energy(prof::Fn::UPD_CYCLE, e1 - e0);

    // advance CBP time base at end-of-block
    prof::time_ps = prof::next_time_ps;

    if (cbpp::g_host_timing && do_meas) prof::add_host(prof::Fn::UPD_CYCLE, prof::now_ns() - t0);
  }
};

// ========================== Driver ==========================
static void usage(const char* name) {
  std::cerr
    << "Usage:\n  " << name
    << " --format csv|human --mode acc|analyze [--profile] [--host-timing] [--no-score]\n"
    << "   <trace.gz> <trace_name> <warmup_instr> <meas_instr>\n";
  std::exit(1);
}

} // namespace cbpp

int main(int argc, char* argv[]) {
  using namespace cbpp;

  if (argc < 5) usage(argv[0]);

  bool human = false;
  std::vector<std::string> pos;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "-f" || arg == "--format") {
      if (++i >= argc) usage(argv[0]);
      std::string fmt = argv[i];
      if (fmt == "human") human = true;
      else if (fmt == "csv") human = false;
      else usage(argv[0]);
    } else if (arg == "--mode") {
      if (++i >= argc) usage(argv[0]);
      std::string m = argv[i];
      if (m == "acc" || m == "accuracy") g_mode = RunMode::ACCURACY;
      else if (m == "analyze") g_mode = RunMode::ANALYZE;
      else usage(argv[0]);
    } else if (arg == "--profile") {
      g_profile = true;
    } else if (arg == "--host-timing") {
      g_host_timing = true;
    } else if (arg == "--no-score") {
      g_print_score = false;
    } else if (arg == "-h" || arg == "--help") {
      usage(argv[0]);
    } else {
      pos.emplace_back(arg);
    }
  }

  if (pos.size() != 4) usage(argv[0]);

  const std::string& trace_path = pos[0];
  const std::string& trace_name = pos[1];
  (void)trace_name;
  uint64_t warmup_instr = std::stoull(pos[2]);
  uint64_t meas_instr   = std::stoull(pos[3]);

  trace_reader reader(trace_path, trace_name);
  using base_pred_t = branch_predictor;

  std::unique_ptr<predictor> pred;
  if (g_mode == RunMode::ACCURACY) {
    acc::warmup_instr = warmup_instr;
    pred = std::make_unique<acc_predictor<base_pred_t>>();
  } else {
    if (g_profile) {
      prof::enable = true;
      prof::warmup_instr = warmup_instr;
      pred = std::make_unique<profiled_predictor<base_pred_t>>();
    } else {
      pred = std::make_unique<base_pred_t>();
    }
  }

  const bool need_capture = (!human) && g_print_score;

  if (need_capture) {
    std::ostringstream cap;
    auto* old = std::cout.rdbuf(cap.rdbuf());
    {
      harcom_superuser sim(reader, false);
      sim.run(*pred, warmup_instr, meas_instr);
    }
    std::cout.rdbuf(old);

    const std::string csvline = csv::last_nonempty_line(cap.str());
    if (!csvline.empty()) {
      std::cout << csvline << "\n";
      auto r = csv::parse(csvline);
      print_score_breakdown(std::cerr, r);
    } else {
      std::cerr << "warning: failed to capture CBP CSV line\n";
    }
  } else {
    harcom_superuser sim(reader, human);
    sim.run(*pred, warmup_instr, meas_instr);
  }

  if (g_mode == RunMode::ACCURACY) {
    acc::print_accuracy(std::cerr);
  } else if (g_profile) {
    prof::dump(std::cerr);
  }

  return 0;
}