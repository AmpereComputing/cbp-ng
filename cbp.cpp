#include "trace_reader.hpp"
#include "harcom.hpp"
#include "cbp.hpp"
#include "branch_predictor.hpp"

branch_predictor pred;

int main(int argc, char * argv[])
{
    if (argc != 5) {
        printf("Expected four arguments: the instruction stream, the trace name, warmup instructions, and measurement instructions\n");
        return -1;
    }
    trace_reader reader(argv[1], argv[2]);
    harcom_superuser sim(reader);
    uint64_t warmup_instructions = std::atoll(argv[3]);
    uint64_t measurement_instructions = std::atoll(argv[4]);
    sim.run(pred, warmup_instructions, warmup_instructions + measurement_instructions);
}
