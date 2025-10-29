#include "predictors/bimodal.hpp"
#include "predictors/gshare.hpp"
#include "predictors/tage.hpp"

#ifdef PREDICTOR
using branch_predictor = PREDICTOR;
#else
//using branch_predictor = bimodal<>;
//using branch_predictor = gshare<>;
using branch_predictor = tage<>;
#endif
