# 03_models.R — LM, multinomial, ordinal (polr), random forest
suppressPackageStartupMessages({
  library(nnet)
  library(MASS)
  library(randomForest)
  library(ggplot2)
})

source("00_utils.R")
args <- parse_args()
ensure_dirs(c(args$outdir, args$figdir))

numeric_data <- readRDS(file.path(args$outdir, "numeric_data.rds"))
categoric_nonnumeric <- readRDS(file.path(args$outdir, "categoric_nonnumeric.rds"))
categoric_nonnumeric_mapped <- readRDS(file.path(args$outdir, "categoric_nonnumeric_mapped.rds"))
ordinal_nn <- readRDS(file.path(args$outdir, "ordinal_nn.rds"))
ordinal_nn_mapped <- readRDS(file.path(args$outdir, "ordinal_nn_mapped.rds"))

# Merge for convenience (row-wise alignment assumed from 01 script)
merged_data <- cbind(numeric_data, categoric_nonnumeric, ordinal_nn)

# ---- LM: numeric Y ~ single categoric X (loop) ----
out_lm <- data.frame()
for (y_col in colnames(numeric_data)) {
  for (x_col in colnames(categoric_nonnumeric)) {
    form <- as.formula(paste0("`", y_col, "` ~ `", x_col, "`"))
    fit <- tryCatch(lm(form, data = merged_data), error = function(e) NULL)
    if (is.null(fit)) next
    coefs <- summary(fit)$coefficients
    coefs <- coefs[setdiff(rownames(coefs), "(Intercept)"), , drop = FALSE]
    if (nrow(coefs) == 0) next
    sig <- coefs[coefs[,4] < 0.05, , drop = FALSE]
    if (nrow(sig) == 0) next
    df <- data.frame(Outcome = y_col,
                     Predictor = x_col,
                     Variable = rownames(sig),
                     Estimate = sig[,1],
                     P_Value = sig[,4],
                     Adj_R2 = summary(fit)$adj.r.squared)
    out_lm <- rbind(out_lm, df)
  }
}
if (nrow(out_lm)) write.csv(out_lm, file.path(args$outdir, "lm_numericY_categoricX_significant.csv"), row.names = FALSE)

# ---- Multinomial: categoric Y ~ numeric predictors ----
mn_results <- data.frame()
if (ncol(categoric_nonnumeric) > 0) {
  rhs <- paste(sprintf("`%s`", colnames(numeric_data)), collapse = " + ")
  for (y_col in colnames(categoric_nonnumeric)) {
    form <- as.formula(paste0("`", y_col, "` ~ ", rhs))
    df <- cbind(categoric_nonnumeric[, y_col, drop=FALSE], numeric_data)
    df <- na.omit(df)
    if (!is.factor(df[[1]])) df[[1]] <- as.factor(df[[1]])
    fit <- tryCatch(multinom(form, data = df, maxit = 200, trace = FALSE), error = function(e) NULL)
    if (is.null(fit)) next
    coef_m <- summary(fit)$coefficients
    se_m <- summary(fit)$standard.errors
    z <- coef_m / se_m; p <- 2 * (1 - pnorm(abs(z)))
    probs <- predict(fit, type = "probs")
    tjur_R2 <- max(colMeans(probs)) - min(colMeans(probs))
    cur <- data.frame(Outcome = y_col,
                      Category = rep(rownames(coef_m), times = ncol(coef_m)),
                      Predictor = rep(colnames(coef_m), each = nrow(coef_m)),
                      Coef = as.vector(coef_m),
                      P = as.vector(p),
                      Tjur_R2 = tjur_R2)
    cur <- subset(cur, P < 0.05 & Coef != 0)
    mn_results <- rbind(mn_results, cur)
  }
}
if (nrow(mn_results)) write.csv(mn_results, file.path(args$outdir, "multinom_categoricY_numericX_significant.csv"), row.names = FALSE)

# ---- Ordinal (polr): ordinal Y ~ numeric X (univariate loops) ----
polr_results <- data.frame()
if (ncol(ordinal_nn) > 0) {
  for (y_col in colnames(ordinal_nn)) {
    y <- as.ordered(ordinal_nn[[y_col]])
    for (x_col in colnames(numeric_data)) {
      df <- data.frame(y = y, x = numeric_data[[x_col]])
      fit <- tryCatch(polr(y ~ x, data = df, Hess = TRUE), error = function(e) NULL)
      if (is.null(fit)) next
      coefs <- coef(summary(fit))
      p <- pnorm(abs(coefs[, "t value"]), lower.tail = FALSE) * 2
      if (p[1] < 0.05) {
        polr_results <- rbind(polr_results, data.frame(Outcome = y_col, Predictor = x_col, Estimate = coefs[1,1], P = p[1]))
      }
    }
  }
}
if (nrow(polr_results)) write.csv(polr_results, file.path(args$outdir, "polr_ordinalY_numericX_significant.csv"), row.names = FALSE)

# ---- Random forest: ordinal Y (importance, per-class accuracy) ----
rf_results <- data.frame()
set.seed(42)
x <- data.frame(categoric_nonnumeric_mapped, numeric_data)
for (y_col in colnames(ordinal_nn)) {
  y <- as.ordered(ordinal_nn[[y_col]])
  df <- na.omit(data.frame(x, y = y))
  if (nrow(df) < 10) next
  fit <- tryCatch(randomForest(x = df[, setdiff(names(df), "y")], y = df$y, ntree = 500, importance = TRUE), error = function(e) NULL)
  if (is.null(fit)) next
  imp <- importance(fit, type = 1)  # MeanDecreaseAccuracy
  imp <- sort(imp[,1], decreasing = TRUE)
  top <- head(imp, 10)
  acc <- mean(predict(fit, df[, setdiff(names(df), "y")]) == df$y)
  rf_results <- rbind(rf_results, data.frame(Outcome = y_col, Predictor = names(top), Importance = as.numeric(top), Overall_Accuracy = acc))
}
if (nrow(rf_results)) write.csv(rf_results, file.path(args$outdir, "rf_ordinalY_importance.csv"), row.names = FALSE)
