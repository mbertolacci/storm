#' @useDynLib positivemixtures
#' @importFrom Rcpp evalCpp

.onUnload <- function (libpath) {
    library.dynam.unload('positivemixtures', libpath)
}
