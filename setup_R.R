#!/usr/bin/env Rscript
# Install or check the exact R package versions used for the baseline.
# Usage: Rscript setup_R.R [--check]

required <- c(
  Rcpp = "1.1.0",
  lattice = "0.22-7",
  nlme = "3.1-168",
  zoo = "1.8-14",
  numDeriv = "2016.8-1.1",
  Formula = "1.2-5",
  stringmagic = "1.2.0",
  dreamerr = "1.5.0",
  sandwich = "3.1-1",
  fixest = "0.13.2",
  data.table = "1.17.8"
)

matches <- function(package, version) {
  requireNamespace(package, quietly = TRUE) &&
    utils::packageVersion(package) == package_version(version)
}

install_exact <- function(package, version, library) {
  archive <- tempfile(fileext = ".tar.gz")
  on.exit(unlink(archive), add = TRUE)
  filename <- sprintf("%s_%s.tar.gz", package, version)
  urls <- c(
    sprintf("https://cran.r-project.org/src/contrib/%s", filename),
    sprintf("https://cran.r-project.org/src/contrib/Archive/%s/%s", package, filename)
  )
  downloaded <- FALSE
  for (url in urls) {
    status <- tryCatch(
      suppressWarnings(utils::download.file(url, archive, mode = "wb", quiet = TRUE)),
      error = function(e) 1L
    )
    if (identical(status, 0L) && file.exists(archive) && file.size(archive) > 0) {
      downloaded <- TRUE
      break
    }
  }
  if (!downloaded) stop("Cannot download ", filename)
  utils::install.packages(archive, repos = NULL, type = "source", lib = library)
  if (!matches(package, version)) stop("Version check failed: ", package)
}

check_only <- "--check" %in% commandArgs(trailingOnly = TRUE)
library <- .libPaths()[1]
if (!check_only && file.access(library, 2L) != 0L) {
  stop("Set R_LIBS_USER to a writable library and restart R before installing.")
}
for (package in names(required)) {
  version <- required[[package]]
  if (!matches(package, version)) {
    if (check_only) stop("Missing or mismatched package: ", package, " == ", version)
    install_exact(package, version, library)
  }
  cat(package, "==", version, "\n")
}
cat("Verified R dependencies. Reference R version: 4.5.1.\n")
