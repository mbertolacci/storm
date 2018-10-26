#' @useDynLib storm
#' @importFrom Rcpp evalCpp

.onUnload <- function (libpath) {
    library.dynam.unload('storm', libpath)
}
