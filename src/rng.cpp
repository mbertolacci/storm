#include "rng.hpp"

namespace ptsm {

#if defined(__clang__)
    RNG rng;
#else
    thread_local RNG rng;
#endif

// NOTE(mgnb): this initialises one RNG per OpenMP thread. They get predictable seeds, which is not enough in general
// for predictable output of MCMC algorithms, but pretty close.
void RNG::initialise() {
    Rcpp::NumericVector draws;
    #pragma omp parallel
    {
        #pragma omp single
        {
            #if defined(omp_get_num_threads)
                draws = Rcpp::runif(omp_get_num_threads(), 0, 1);
            #else
                draws = Rcpp::runif(1, 0, 1);
            #endif
        }

        #if defined(omp_get_thread_num)
            rng = RNG(static_cast<uint_fast64_t>(UINT_FAST64_MAX * draws[omp_get_thread_num()]));
        #else
            rng = RNG(static_cast<uint_fast64_t>(UINT_FAST64_MAX * draws[0]));
        #endif
    }
}

}
