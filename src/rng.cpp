#include "rng.hpp"
#if defined(_OPENMP)
#include <omp.h>
#endif

namespace ptsm {

#if defined(_OPENMP)
    thread_local RNG rng;
#else
    RNG rng;
#endif

// NOTE(mgnb): this initialises one RNG per OpenMP thread. They get predictable seeds, which is not enough in general
// for predictable output of MCMC algorithms, but pretty close.
void RNG::initialise() {
    Rcpp::NumericVector draws;
    #pragma omp parallel
    {
        #pragma omp single
        {
            #if defined(_OPENMP)
                draws = Rcpp::runif(omp_get_num_threads(), 0, 1);
            #else
                draws = Rcpp::runif(1, 0, 1);
            #endif
        }

        #if defined(_OPENMP)
            rng = RNG(static_cast<uint_fast64_t>(UINT_FAST64_MAX * draws[omp_get_thread_num()]));
        #else
            rng = RNG(static_cast<uint_fast64_t>(UINT_FAST64_MAX * draws[0]));
        #endif
    }
}

}
