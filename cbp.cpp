#include "trace_reader.hpp"
#include "harcom.hpp"
#include "cbp.hpp"
#include "branch_predictor.hpp"

branch_predictor pred;

static void usage(const char* name) {
    printf("Usage:\n  %s <path to trace> <name of trace> <warmup instructions> <measurement instructions> [--format csv|human]\n", name);
    std::exit(1);
}

int main(int argc, char* argv[])
{
    if (argc < 5) {
        printf("Too few arguments!\n\n");
        usage(argv[0]);
    }

    std::vector<std::string> positional_args;
    bool human_readable_output = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-f" || arg == "--format") {
            i++;
            if (i == argc) {
                printf("Argument must follow --format!\n\n");
                usage(argv[0]);
            }
            std::string format = argv[i];
            if (format == "human") {
                human_readable_output = true;
            } else if (format == "csv") {
                human_readable_output = false;
            } else {
                printf("Invalid argument to --format: %s\n\n", argv[i]);
                usage(argv[0]);
            }
        } else if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
        } else {
            positional_args.emplace_back(arg);
        }
    }

    if (positional_args.size() != 4) {
        printf("Too few positional arguments!\n\n");
        usage(argv[0]);
    }

    trace_reader reader(positional_args[0], positional_args[1]);
    harcom_superuser sim(reader, human_readable_output);
    uint64_t warmup_instructions = std::stoll(positional_args[2]);
    uint64_t measurement_instructions = std::stoll(positional_args[3]);
    sim.run(pred, warmup_instructions, measurement_instructions);
}
