# Generated by using Rcpp::compileAttributes() -> do not edit by hand
# Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

hasOMP <- function() {
    .Call(`_CautiousLearning_hasOMP`)
}

setOMPThreads <- function(nthreads) {
    invisible(.Call(`_CautiousLearning_setOMPThreads`, nthreads))
}

setSITMOSeeds <- function(seed) {
    invisible(.Call(`_CautiousLearning_setSITMOSeeds`, seed))
}

mkChart <- function(chart, m, A, B, arl0, Linf, alpha = 0.10, beta = 0.05, H = 200L, Ninit = 1000L, Nfinal = 30000L) {
    .Call(`_CautiousLearning_mkChart`, chart, m, A, B, arl0, Linf, alpha, beta, H, Ninit, Nfinal)
}

ruv <- function(n, m) {
    .Call(`_CautiousLearning_ruv`, n, m)
}

rcrl <- function(n, chart, u, v, tau, delta, omega, maxrl = 1000000L) {
    .Call(`_CautiousLearning_rcrl`, n, chart, u, v, tau, delta, omega, maxrl)
}

applyChart <- function(chart, x, mu0, s0) {
    .Call(`_CautiousLearning_applyChart`, chart, x, mu0, s0)
}

