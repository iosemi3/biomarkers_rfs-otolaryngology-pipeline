# 02_assoc_maps.R — cross-type association matrices and heatmaps
suppressPackageStartupMessages({
  library(Hmisc)
  library(reshape2)
  library(ggplot2)
})

source("00_utils.R")
args <- parse_args()
ensure_dirs(c(args$outdir, args$figdir))

# Load panels
numeric_data <- readRDS(file.path(args$outdir, "numeric_data.rds"))
binary_nonnumeric <- readRDS(file.path(args$outdir, "binary_nonnumeric.rds"))
ordinal_nn_mapped <- readRDS(file.path(args$outdir, "ordinal_nn_mapped.rds"))
categoric_nonnumeric_mapped <- readRDS(file.path(args$outdir, "categoric_nonnumeric_mapped.rds"))

# ---- Numeric vs Numeric (Spearman) ----
res_num <- rcorr(as.matrix(numeric_data), type = "spearman")
corr <- res_num$r; pmat <- res_num$P
sig <- corr; sig[pmat >= 0.05] <- NA
write.csv(sig, file.path(args$outdir, "significant_spearman_numeric_allvsall.csv"))
m <- melt(sig); m <- m[!is.na(m$value), ]
if (nrow(m)) {
  p <- ggplot(m, aes(Var1, Var2, fill = value)) + geom_tile() +
    geom_text(aes(label = sprintf("%.2f", value))) +
    scale_fill_gradient2(limits=c(-1,1), midpoint=0) +
    theme_minimal() + theme(axis.text.x = element_text(angle=45,hjust=1))
  ggsave2("spearman_numeric.pdf", p, path = args$figdir)
}

# ---- Numeric vs Ordinal ----
res_no <- rcorr(as.matrix(numeric_data), as.matrix(ordinal_nn_mapped), type = "spearman")
sig_no <- res_no$r; sig_no[res_no$P >= 0.05] <- NA
write.csv(sig_no, file.path(args$outdir, "significant_spearman_numeric_vs_ordinal.csv"))
m <- melt(sig_no); m <- m[!is.na(m$value), ]
if (nrow(m)) {
  p <- ggplot(m, aes(Var1, Var2, fill = value)) + geom_tile() +
    geom_text(aes(label = sprintf("%.2f", value))) +
    scale_fill_gradient2(limits=c(-1,1), midpoint=0) +
    theme_minimal() + theme(axis.text.x = element_text(angle=45,hjust=1))
  ggsave2("spearman_num_vs_ord.pdf", p, path = args$figdir)
}

# ---- Numeric vs Categoric (encoded) ----
res_nc <- rcorr(as.matrix(numeric_data), as.matrix(categoric_nonnumeric_mapped), type = "spearman")
sig_nc <- res_nc$r; sig_nc[res_nc$P >= 0.05] <- NA
write.csv(sig_nc, file.path(args$outdir, "significant_spearman_numeric_vs_categoric.csv"))
m <- melt(sig_nc); m <- m[!is.na(m$value), ]
if (nrow(m)) {
  p <- ggplot(m, aes(Var1, Var2, fill = value)) + geom_tile() +
    geom_text(aes(label = sprintf("%.2f", value))) +
    scale_fill_gradient2(limits=c(-1,1), midpoint=0) +
    theme_minimal() + theme(axis.text.x = element_text(angle=45,hjust=1))
  ggsave2("spearman_num_vs_cat.pdf", p, path = args$figdir)
}

# ---- Binary vs Binary (chi-square + Cramer's V) ----
if (ncol(binary_nonnumeric) > 1) {
  p_values <- matrix(NA, ncol(binary_nonnumeric), ncol(binary_nonnumeric),
                     dimnames = list(names(binary_nonnumeric), names(binary_nonnumeric)))
  cramers_v <- p_values
  for (i in 1:ncol(binary_nonnumeric)) {
    for (j in 1:ncol(binary_nonnumeric)) {
      if (i == j) next
      test <- suppressWarnings(chisq.test(binary_nonnumeric[[i]], binary_nonnumeric[[j]], correct = FALSE))
      p_values[i, j] <- test$p.value
      if (!is.na(test$p.value) && test$p.value < 0.05) {
        cramers_v[i, j] <- sqrt(test$statistic / (nrow(binary_nonnumeric) * (min(dim(table(binary_nonnumeric[[i]], binary_nonnumeric[[j]]))) - 1)))
      }
    }
  }
  write.csv(p_values, file.path(args$outdir, "binary_chisq_pvals.csv"))
  write.csv(cramers_v, file.path(args$outdir, "binary_cramersV.csv"))
  m <- melt(cramers_v); m <- m[!is.na(m$value), ]
  if (nrow(m)) {
    p <- ggplot(m, aes(Var1, Var2, fill = value)) + geom_tile() +
      geom_text(aes(label = sprintf("%.2f", value))) +
      scale_fill_gradient(limits=c(0,1)) + theme_minimal() +
      theme(axis.text.x = element_text(angle=45,hjust=1)) +
      labs(title="Cramér's V (binary vs binary)")
    ggsave2("cramersV_binary.pdf", p, path = args$figdir)
  }
}

# ---- Binary vs Ordinal (point-biserial via Pearson after binarization) ----
if (ncol(binary_nonnumeric) > 0 && ncol(ordinal_nn_mapped) > 0) {
  cor_mat <- matrix(NA, nrow = ncol(ordinal_nn_mapped), ncol = ncol(binary_nonnumeric),
                    dimnames = list(colnames(ordinal_nn_mapped), colnames(binary_nonnumeric)))
  p_mat <- cor_mat
  for (i in 1:ncol(ordinal_nn_mapped)) {
    for (j in 1:ncol(binary_nonnumeric)) {
      b <- binary_nonnumeric[[j]]
      if (length(unique(b)) == 2) {
        u <- sort(unique(b)); b <- ifelse(b == u[1], 0, 1)
        ct <- suppressWarnings(cor.test(ordinal_nn_mapped[[i]], b, method="pearson"))
        p_mat[i,j] <- ct$p.value
        if (!is.na(ct$p.value) && ct$p.value < 0.05) cor_mat[i,j] <- unname(ct$estimate)
      }
    }
  }
  write.csv(p_mat, file.path(args$outdir, "binary_vs_ordinal_pointbiserial_pvals.csv"))
  write.csv(cor_mat, file.path(args$outdir, "binary_vs_ordinal_pointbiserial_sig.csv"))
  m <- melt(cor_mat); m <- m[!is.na(m$value), ]
  if (nrow(m)) {
    p <- ggplot(m, aes(Var1, Var2, fill = value)) + geom_tile() +
      geom_text(aes(label = sprintf("%.2f", value))) +
      scale_fill_gradient2(limits=c(-1,1), midpoint=0) + theme_minimal() +
      theme(axis.text.x = element_text(angle=45,hjust=1))
    ggsave2("pointbiserial_bin_vs_ord.pdf", p, path = args$figdir)
  }
}

# ---- Ordinal vs Categoric (Kruskal + eta^2 proxy) ----
if (ncol(ordinal_nn_mapped) > 0 && ncol(categoric_nonnumeric_mapped) > 0) {
  eta2 <- matrix(NA, nrow=ncol(ordinal_nn_mapped), ncol=ncol(categoric_nonnumeric_mapped),
                 dimnames=list(colnames(ordinal_nn_mapped), colnames(categoric_nonnumeric_mapped)))
  for (i in 1:ncol(ordinal_nn_mapped)) {
    for (j in 1:ncol(categoric_nonnumeric_mapped)) {
      kr <- suppressWarnings(kruskal.test(ordinal_nn_mapped[[i]], categoric_nonnumeric_mapped[[j]]))
      if (!is.na(kr$p.value) && kr$p.value < 0.05) {
        n <- length(ordinal_nn_mapped[[i]])
        eta2[i,j] <- as.numeric(kr$statistic) / (n - 1)
      }
    }
  }
  write.csv(eta2, file.path(args$outdir, "ordinal_vs_categoric_eta2_sig.csv"))
  m <- melt(eta2); m <- m[!is.na(m$value), ]
  if (nrow(m)) {
    p <- ggplot(m, aes(Var1, Var2, fill = value)) + geom_tile() +
      geom_text(aes(label = sprintf("%.2f", value))) +
      scale_fill_gradient(limits=c(0,1)) + theme_minimal() +
      theme(axis.text.x = element_text(angle=45,hjust=1)) +
      labs(title="Kruskal–Wallis η² (ordinal vs categoric)")
    ggsave2("kruskal_eta2_ord_vs_cat.pdf", p, path = args$figdir)
  }
}
