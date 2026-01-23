#include "predictors/always_taken.hpp"
#include "predictors/bimodal.hpp"
#include "predictors/gshare.hpp"
#include "predictors/never_taken.hpp"
#include "predictors/tage.hpp"
#include "predictors/bimodalN.hpp"
#include "predictors/gshareN.hpp"

#ifdef PREDICTOR
using branch_predictor = PREDICTOR;
#else
//using branch_predictor = bimodal<>;
//using branch_predictor = gshare<>;
using branch_predictor = tage<>;
#endif
