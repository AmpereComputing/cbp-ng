#include <sstream>
#include <string>
#include <map>

#include "../../harcom.hpp"

using namespace hcm;

struct energy_monitor {
    /*
     * This class and the below macro provide a simple way to monitor HARCOM
     * energy usage between different call sites in a HARCOM design. The energy
     * consumption differential between two consecutive calls to `record()` is
     * recorded and the average energy usage across all pairings of those two
     * call sites is reported whenever `report()` is called.
     *
     * Note that the results of this are only meaningful for pairs of call
     * sites whose relationship you understand. For example, if two call sites 
     */

    struct count {
        f64 total_energy{0.0};
        u64 count{0};
    };
    std::map<std::string, struct count> counts;

    std::string last_file = "reset";
    u64 last_line = 0;
    f64 last_energy_fJ = 0.0;

    void reset() { record("reset", 0); }

    void record(const char *file, const int line) {
        std::string this_file = file;
        u64 this_line = line;
        f64 this_energy_fJ = panel.energy_fJ();
        std::ostringstream oss;
        oss << last_file << ":" << last_line << " - " << this_file << ":" << this_line;
        std::string key = oss.str();

        f64 difference = this_energy_fJ - last_energy_fJ;
        if (counts.contains(key)) {
            counts[key].total_energy += difference;
            counts[key].count++;
        } else {
            counts[key] = {
                .total_energy = difference,
                .count = 1
            };
        }
        last_file = this_file;
        last_line = this_line;
        last_energy_fJ = this_energy_fJ;
    }

    void report() {
        printf("\nMeasurement start/end points: average energy difference (fJ)\n");
        printf("-------------------------------------------------------------\n");
        for (const auto& [key, count] : counts) {
            printf("%s: %.2f fJ\n", key.c_str(), count.total_energy / count.count);
        }
    }
};

#define energy_checkpoint(monitor) monitor.record(__FILE__, __LINE__)
