#include <iostream>
#include "trace_reader.hpp"
#include "seznec_cbp2025.h"


CBP2025 PRED;

int main(int argc, char * argv[])
{
    if (argc != 5) {
        printf("Expected three arguments: the instruction stream, warmup instructions, and measurement instructions\n");
        return -1;
    }
    trace_reader reader(argv[1], argv[2]);
    uint64_t warmup_instructions = atoi(argv[3]);
    uint64_t measurement_instructions = atoi(argv[4]);
    uint64_t total_instructions = warmup_instructions + measurement_instructions;

    uint64_t ninstr = 0;
    uint64_t nbranch = 0;
    uint64_t ncondbr = 0;
    uint64_t nmisp = 0;
    uint64_t nblock = 0;

    bool warmed_up = (warmup_instructions==0);

    try {

        while (ninstr < total_instructions) {

            auto inst = reader.next_instruction();
            ninstr++;

            if (!inst.branch) continue; // not a branch, go to next instruction
            nbranch++;

            int brtype = 0;
            switch (inst.inst_class) {
            case INST_CLASS::BR_UNCOND_DIRECT:
            case INST_CLASS::BR_CALL_DIRECT:
                brtype = 0;
                break;
            case INST_CLASS::BR_COND:
                brtype = 1;
                break;
            case INST_CLASS::BR_UNCOND_INDIRECT:
            case INST_CLASS::BR_CALL_INDIRECT:
            case INST_CLASS::BR_RETURN:
                brtype = 2;
                break;
            default:
                assert(0);
            };

            bool actual_dir = inst.taken_branch;
            bool misprediction = false;

            if (inst.inst_class == INST_CLASS::BR_COND) {
                // conditional branch
                ncondbr++;
                bool prediction = PRED.predict(nbranch,0,inst.pc);
                misprediction = (prediction != actual_dir);
                if (misprediction) nmisp++;
                PRED.history_update(nbranch, 0, inst.pc, brtype, actual_dir, inst.next_pc);
                PRED.update(nbranch, 0, inst.pc, actual_dir, prediction, inst.next_pc);
            } else {
                PRED.TrackOtherInst(inst.pc, brtype, actual_dir, inst.next_pc);
            }

            if (inst.taken_branch || misprediction) nblock++;

            if (!warmed_up && ninstr > warmup_instructions) {
                warmed_up = true;
                ninstr = 0;
                nbranch = 0;
                ncondbr = 0;
                nmisp = 0;
                nblock = 0;
            }
        }
    } catch (const out_of_instructions &e) { }

    std::cout << reader.name();
    std::cout << "," << ninstr;
    std::cout << "," << nbranch;
    std::cout << "," << ncondbr;
    std::cout << "," << nblock;//npred;
    std::cout << "," << 0;//p1_p2_disagreements;
    std::cout << "," << 0;//p1_p2_disagreements_at_block_end;
    std::cout << "," << nmisp;//mispredictions;
    std::cout << "," << 0;//p1_latency_cycles;
    std::cout << "," << 0;//p2_latency_cycles;
    std::cout << "," << 0;//int64_t(epi_fJ);
    std::cout << std::endl;
}
