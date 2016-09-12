#ifndef SRC_LOGGING_HPP_
#define SRC_LOGGING_HPP_

#include <string>
#include <RcppArmadillo.h>

namespace ptsm {

class Logger {
 public:
    Logger(std::string name)
        : name_(name) {
        futileLogger_ = Rcpp::Environment::namespace_env("futile.logger");
    }

    template<typename ... Args>
    void trace(const std::string& format, Args ... args) {
        Rcpp::Function flogTrace = futileLogger_["flog.trace"];
        flogTrace(format, args..., Rcpp::_["name"]=name_);
    }

    template<typename ... Args>
    void debug(const std::string& format, Args ... args) {
        Rcpp::Function flogDebug = futileLogger_["flog.debug"];
        flogDebug(format, args..., Rcpp::_["name"]=name_);
    }

    template<typename ... Args>
    void info(const std::string& format, Args ... args) {
        Rcpp::Function flogInfo = futileLogger_["flog.info"];
        flogInfo(format, args..., Rcpp::_["name"]=name_);
    }

    template<typename ... Args>
    void warn(const std::string& format, Args ... args) {
        Rcpp::Function flogWarn = futileLogger_["flog.warn"];
        flogWarn(format, args..., Rcpp::_["name"]=name_);
    }

    template<typename ... Args>
    void error(const std::string& format, Args ... args) {
        Rcpp::Function flogError = futileLogger_["flog.error"];
        flogError(format, args..., Rcpp::_["name"]=name_);
    }

    template<typename ... Args>
    void fatal(const std::string& format, Args ... args) {
        Rcpp::Function flogFatal = futileLogger_["flog.fatal"];
        flogFatal(format, args..., Rcpp::_["name"]=name_);
    }

 private:
    std::string name_;
    Rcpp::Environment futileLogger_;
};

}  // namespace ptsm

#endif  // SRC_LOGGING_HPP_
