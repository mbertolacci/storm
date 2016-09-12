#ifndef SRC_PROGRESS_HPP_
#define SRC_PROGRESS_HPP_

#include <chrono>
#include <cstdint>
#include <RcppArmadillo.h>

class ProgressBar {
 public:
    explicit ProgressBar(uint64_t nSteps)
        : nSteps_(nSteps),
          currentStep_(0),
          lastCheckStep_(0),
          lastCheckStepTime_(0) {
        lastCheckTime_ = std::chrono::system_clock::now();
    }

    uint64_t operator+=(uint64_t increment) {
        currentStep_ += increment;
        output();
        return currentStep_;
    }

    uint64_t operator++() {
        currentStep_ += 1;
        output();
        return currentStep_;
    }

 private:
    uint64_t nSteps_;
    uint64_t currentStep_;

    uint64_t lastCheckStep_;
    std::chrono::time_point<std::chrono::system_clock> lastCheckTime_;
    double lastCheckStepTime_;

    void output() {
        unsigned int percent = 100 * currentStep_ / nSteps_;

        if (currentStep_ == 1 || currentStep_ % 10 == 0) {
            std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
            std::chrono::duration<double> elapsedSeconds = now - lastCheckTime_;
            lastCheckStepTime_ = 1000 * elapsedSeconds.count() / (currentStep_ - lastCheckStep_);

            lastCheckStep_ = currentStep_;
            lastCheckTime_ = now;
        }

        Rcpp::Rcout << "\r"
          << currentStep_ << "/" << nSteps_ << " (" << percent << "%)"
          << " " << lastCheckStepTime_ << "ms/iteration"
          << " (" << ((nSteps_ - currentStep_) * lastCheckStepTime_) / 1000 << "s remaining)";

        if (currentStep_ == nSteps_) {
            Rcpp::Rcout << "\n";
        }

        Rcpp::Rcout.flush();
    }
};

#endif  // SRC_PROGRESS_HPP_
