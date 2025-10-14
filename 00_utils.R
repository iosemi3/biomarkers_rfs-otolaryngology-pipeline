# 00_utils.R — helpers for path args, I/O, and safe plotting

parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  kv <- list()
  if (length(args) > 0) {
    for (i in seq(1, length(args), by = 2)) {
      key <- gsub("^--", "", args[i])
      val <- args[i + 1]
      kv[[key]] <- val
    }
  }
  if (is.null(kv$in)) kv$in <- "data/italproj_full.csv"
  if (is.null(kv$outdir)) kv$outdir <- "results"
  if (is.null(kv$figdir)) kv$figdir <- "figures"
  kv
}

ensure_dirs <- function(paths) {
  for (p in paths) if (!dir.exists(p)) dir.create(p, recursive = TRUE, showWarnings = FALSE)
}

saveRDS2 <- function(obj, file) {
  dir.create(dirname(file), recursive = TRUE, showWarnings = FALSE)
  saveRDS(obj, file = file)
}

ggsave2 <- function(filename, plot, path = ".", width = 7, height = 5) {
  dir.create(path, recursive = TRUE, showWarnings = FALSE)
  ggplot2::ggsave(filename = filename, plot = plot, path = path, width = width, height = height)
}
