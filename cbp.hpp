#include <cstdint>

#include <iostream>
#include <functional>
#include "trace_reader.hpp"
#include "harcom.hpp"

#pragma once

using namespace hcm;

static constexpr uint64_t cycle_ps = 300;

/*
 * `predictor` specifies the interface each branch predictor will implement so
 * that the simulator can exercise it. This interface allows for implementing
 * up to two distinct prediction stages (named 1 and 2), as well as re-using
 * previous results for the same stage (to allow, for example, predictors to
 * look up structures once per cacheline rather than once per instruction).
 *
 * The predict methods collectively (predict1, reuse_predict1, predict2,
 * reuse_predict2) are called with the program counter of each right-path
 * instruction in a given trace. One method in each level (i.e. 1 or 2) is
 * called for every instruction, but which method within that level is
 * determined by the predictor itself.
 *
 * The predictor may call the `reuse_prediction()` with a HARCOM `val<1>`.
 * If the last call to `reuse_prediction()` had a value of 1, the simulator will call
 * `reuse_predict1` and `reuse_predict2` for the next instruction instead of
 * `predict1` and `predict2`.
 *
 * Instructions that are predicted in the same clock cycle form a "prediction block".
 * A prediction block ends if the current instruction is an unconditional branch,
 * or a conditional branch mispredicted by the level 1 or level 2 predictor,
 * or if reuse_prediction() is called with value 0.
 * When the prediction block ends, the simulator resets "reuse prediction" to 0
 * so that predict1 and predict2 are called in the next cycle.
 *
 * You do not need to implement both levels of prediction. You may effectively
 * omit level 1 by always returning a static value of 0 or 1. Or you may omit
 * level 2 by saving the previous level 1 return value (i.e. via a HARCOM
 * `reg`) and returning the same thing for level 2.
 *
 * After the appropriate *predict* calls are made for an instruction which is a
 * conditional branch, `update` will be called with the program counter of that
 * branch and its direction.
 */

using callback =  std::function<void(val<1>)>;

struct predictor {
    friend class harcom_superuser;
    /*
     * predict1 or reuse_predict1 are the first prediction methods called for an instruction.
     */
    virtual val<1> predict1(val<64> inst_pc) = 0;
    virtual val<1> reuse_predict1(val<64> inst_pc) = 0;

    virtual val<1> predict2(val<64> inst_pc) = 0;
    virtual val<1> reuse_predict2(val<64> inst_pc) = 0;

    virtual void update_condbr(val<64> branch_pc, val<1> taken, val<64> next_pc) = 0;

    virtual void update_cycle(val<1> mispredict, val<64> next_pc) = 0;

    void reuse_prediction(val<1> reuse_next) {
        reuse_prediction_callback(reuse_next.fo1());
    }

    void need_extra_cycle(val<1> yes) {
        need_extra_cycle_callback(yes.fo1());
    }

    virtual ~predictor()
    {
#ifdef VERBOSE
        panel.print(std::cerr);
#endif
    }

private:
    callback need_extra_cycle_callback;
    callback reuse_prediction_callback;
};


class harcom_superuser {
    friend class predictor;
    uint64_t ninstr = 0;
    uint64_t nbranch = 0;
    uint64_t ncondbr = 0;
    uint64_t npred = 0; // number of prediction cycles on the correct path
    uint64_t mispredictions = 0;
    uint64_t p1_p2_disagreements = 0;
    uint64_t p1_p2_disagreements_at_block_end = 0;
    uint64_t extra_cycles = 0;
    double warmup_energy_fJ = 0;
    bool warmed_up = false;

    uint64_t time = 0; // for measuring predictor latencies
    uint64_t max_p1_lat_ps = 0;
    uint64_t max_p2_lat_ps = 0;

    trace_reader &reader;

    auto next_instruction()
    {
        // traces have discontinuities in their control without any branch;
        // we must correct taken_branch and next_pc;
        // treat discontinuities as if they are unconditional branches
        static instruction instr;
        static instruction next_instr;
        instr = next_instr;
        next_instr = reader.next_instruction();
        ninstr++;
        if (instr.branch) {
            nbranch++;
            if (instr.inst_class == INST_CLASS::BR_COND)
                ncondbr++;
        }
        instr.next_pc = next_instr.pc;
        instr.taken_branch = (next_instr.pc != instr.pc+4);
        return instr;
    }

    void clear_stats()
    {
        ninstr = 0;
        nbranch = 0;
        ncondbr = 0;
        npred = 0;
        mispredictions = 0;
        p1_p2_disagreements = 0;
        p1_p2_disagreements_at_block_end = 0;
        extra_cycles = 0;
        warmup_energy_fJ = panel.energy_fJ();
    }

public:

    harcom_superuser(trace_reader &reader): reader(reader)
    {
        panel.clock_cycle_ps = cycle_ps;
        panel.make_floorplan();
    }

    void run(predictor &p, uint64_t warmup_instructions=0, uint64_t measurement_instructions=10)
    {
        warmed_up = (warmup_instructions==0);

        // Note: reuse_prediction changes values based on callbacks made from
        // within the predictors, so we query and store its value below rather
        // than rely upon it to remain constant
        bool reuse_prediction = false;
        callback rcb = [&reuse_prediction](val<1> reuse_me) {
            reuse_prediction = reuse_me.fo1().get();
        };
        p.reuse_prediction_callback = rcb;

        p.need_extra_cycle_callback = [&](val<1> yes) {
            // ignore timing of value "yes" (FIXME?)
            if (yes.fo1().get()) {
                panel.next_cycle();
                extra_cycles++;
            }
        };

        try {
            while (!warmed_up || ninstr < measurement_instructions) {
                auto instruction = next_instruction();
                bool conditional_branch = (instruction.inst_class == INST_CLASS::BR_COND);
                //bool unconditional_branch = instruction.branch && !conditional_branch;

                // First, make the first and second level predictions
                val<1> p1_result;
                val<1> p2_result;
                bool reuse_this_prediction = reuse_prediction;
                if (reuse_this_prediction) {
                    p1_result = p.reuse_predict1({instruction.pc, time});
                    p2_result = p.reuse_predict2({instruction.pc, time});
                } else {
                    p1_result = p.predict1({instruction.pc, time});
                    p2_result = p.predict2({instruction.pc, time});
                }

                auto [prediction1, p1_time] = p1_result.fo1().get_vt();
                auto [prediction2, p2_time] = p2_result.fo1().get_vt();

                uint64_t next_time = time;
                if (p1_time > time) {
                    uint64_t p1_lat_ps = p1_time - time;
                    max_p1_lat_ps = std::max(max_p1_lat_ps, p1_lat_ps);
                    uint64_t p1_lat_cycles = (p1_lat_ps + cycle_ps-1) / cycle_ps;
                    next_time += p1_lat_cycles * cycle_ps;
                } else {
                    next_time += cycle_ps;
                }
                if (p2_time > time) {
                    max_p2_lat_ps = std::max(max_p2_lat_ps, p2_time-time);
                }

                // Update the predictor if this was a conditional branch
                if (conditional_branch) {
                    p.update_condbr({instruction.pc, next_time},
                                    {instruction.taken_branch, next_time},
                                    {instruction.next_pc, next_time});
                }

                bool reuse_next_prediction = reuse_prediction;
                bool p2_misprediction = conditional_branch && (prediction2 != instruction.taken_branch);
                // FIXME: in reality, P1 predicts all instructions, not just branches
                bool p1_p2_disagreement = conditional_branch && (prediction2 != prediction1);
                bool end_of_block = instruction.taken_branch || p2_misprediction || !reuse_next_prediction;
                if (p2_misprediction) {
                    mispredictions++;
                } else if (p1_p2_disagreement) {
                    p1_p2_disagreements++;
                    if (end_of_block) {
                        p1_p2_disagreements_at_block_end++;
                    }
                }

                // One block predicted per cycle.
                // The predicted block ends in the following cases:
                //   @ jump;
                //   @ level 2 misprediction;
                //   @ the predictor asks to stop here.
                if (end_of_block) {
                    p.update_cycle({p2_misprediction,next_time},{instruction.next_pc,next_time});
                    panel.next_cycle();
                    time = next_time;
                    npred++;
                    // Override what the predictor said about re-using this prediction
                    // (in case it wanted to continue).
                    reuse_prediction = false;
                }

                if (!warmed_up && ninstr > warmup_instructions) {
                    warmed_up = true;
                    clear_stats();
                }
            }
        } catch (const out_of_instructions &e) { }
    }

    ~harcom_superuser()
    {
        if (!warmed_up) {
            std::cout << reader.name();
            std::cout << ",0,0,0,0,0,0,0,0,0" << std::endl;
            return;
        }

        double p1_latency_cycles = double(max_p1_lat_ps) / cycle_ps;
        double p2_latency_cycles = double(max_p2_lat_ps) / cycle_ps;
        double energy_fJ = panel.energy_fJ() - warmup_energy_fJ; // total correct-path dynamic energy
        double epi_fJ = energy_fJ / ninstr; // dynamic energy per correct-path instruction
        std::cout << reader.name();
        std::cout << "," << ninstr;
        std::cout << "," << nbranch;
        std::cout << "," << ncondbr;
        std::cout << "," << npred;
        std::cout << "," << extra_cycles;
        std::cout << "," << p1_p2_disagreements;
        std::cout << "," << p1_p2_disagreements_at_block_end;
        std::cout << "," << mispredictions;
        std::cout << "," << p1_latency_cycles;
        std::cout << "," << p2_latency_cycles;
        std::cout << "," << int64_t(epi_fJ);
        std::cout << std::endl;
    }
};
