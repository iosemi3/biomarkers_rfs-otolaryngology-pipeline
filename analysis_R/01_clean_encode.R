# 01_clean_encode.R — load, clean, engineer features, split cohorts, core stats
suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(reshape2)
  library(car)
})

source("00_utils.R")
args <- parse_args()
ensure_dirs(c(args$outdir, args$figdir))

# ---- Load ----
data <- read.csv(args$in, header = TRUE, sep = ";", stringsAsFactors = FALSE)

# Global replace: "" -> NA -> 0 (use with caution)
data[data == ""] <- NA
data[is.na(data)] <- 0

# RFS -> numeric ratio a/b
split_data <- strsplit(as.character(data$RFS), " / ")
data$RFS_numeric <- sapply(split_data, function(x) as.numeric(x[1]) / as.numeric(x[2]))
rm(split_data)

# Harmonize factors / numeric coercions
data$Istologico <- gsub("^K squamoso.*$", "K squamoso", data$Istologico)
num_cols <- c("F0..Hz.","Intensità..dB.","Ampiezza.della.vibrazione..DX.","Ampiezza.della.vibrazione..Sx.",
              "Onda.mucosa..Dx.","Onda.mucosa..Sx.","Porzioni.non.vibranti..Dx.","Porzioni.non.vibranti..Sx.")
for (nm in intersect(num_cols, names(data))) data[[nm]] <- as.numeric(as.character(data[[nm]]))

# Extract numeric content from Regolarità... columns
target_columns <- c("Regolarità....di.tempo.vibrazione.regolare...Dx.",
                    "Regolarità....di.tempo.vibrazione.regolare...Sx.",
                    "Regolarità....di.tempo.vibrazione.regolare...Bilaterale.")
for (col_name in intersect(target_columns, names(data))) {
  new_col <- paste0(col_name, ".extracted")
  data[[new_col]] <- as.numeric(gsub("[^0-9.]", "", as.character(data[[col_name]])))
}
data <- data %>% select(-any_of(target_columns))

# Helper: collapse repeated ordinal blocks to maximum numeric value
merge_and_keep_highest <- function(df, stem, exclude_regex = NULL) {
  cols <- grep(paste0("^", gsub("\\.", "\\\\.", stem)), names(df))
  if (length(cols) == 0) return(df)
  apply_fun <- function(x) {
    if (!is.null(exclude_regex)) x <- ifelse(grepl(exclude_regex, x), NA, x)
    numeric_values <- suppressWarnings(as.numeric(gsub("[^0-9]", "", x)))
    hv <- max(numeric_values, na.rm = TRUE)
    if (is.infinite(hv)) hv <- 0
    as.character(hv)
  }
  df[[stem]] <- apply(df[, cols, drop = FALSE], 1, apply_fun)
  df <- df[, -grep(paste0("^", gsub("\\.", "\\\\.", stem), "\\."), names(df)), drop = FALSE]
  df
}

merge_and_keep_highest2 <- function(df, stem) merge_and_keep_highest(df, stem, exclude_regex = NULL)

# Remove all-zero / all-NA columns after coercions
nonzero <- function(x) any(x != 0 & !is.na(x))
data_cleaned <- data[, vapply(data, nonzero, logical(1))]

# Specific merges (if present)
for (stem in c("Edema.delle.CVV","Obliterazione.ventricolare","Eritema...Iperemia",
               "Edema.diffuso.laringeo","Ipertrofia.della.commissura.posteriore")) {
  data_cleaned <- merge_and_keep_highest(data_cleaned, stem)
}

# Cohort split
data_control <- subset(data_cleaned, CONTROLLI == "ASSENZA DI PATOLOGIA RISCONTRABILE ENDOSCOPICAMENTE")
data_disease <- subset(data_cleaned, CONTROLLI != "ASSENZA DI PATOLOGIA RISCONTRABILE ENDOSCOPICAMENTE")

# ---- Core stats: RFS control vs disease ----
data_control$group <- 'control'; data_disease$group <- 'disease'
combined_data <- rbind(data_control, data_disease)

# Levene & Wilcoxon
suppressWarnings(leveneTest(RFS_numeric ~ group, data = combined_data))
wilcox_res <- wilcox.test(RFS_numeric ~ group, data = combined_data, exact = FALSE)

# Boxplot with p-value annotation
p_txt <- if (wilcox_res$p.value < 0.05) sprintf("* p = %.2e", wilcox_res$p.value) else sprintf("ns p = %.2e", wilcox_res$p.value)
max_val <- max(combined_data$RFS_numeric, na.rm = TRUE)
plt <- ggplot(combined_data, aes(x = group, y = RFS_numeric, fill = group)) +
  geom_boxplot() + theme_minimal() +
  labs(title = "RFS by group", x = "Group", y = "RFS (numeric)") +
  annotate("text", x = 1.5, y = max_val, label = p_txt, size = 4.5)
ggsave2("rfs_group_boxplot.pdf", plt, path = args$figdir)

# ---- Construct analysis panels ----
numeric_vars <- sapply(data_cleaned, is.numeric)
numeric_data <- data_cleaned[, numeric_vars, drop = FALSE]
numeric_data[is.na(numeric_data)] <- 0
numeric_data <- numeric_data[, colSums(numeric_data) != 0, drop = FALSE]

# Non-numeric
non_numeric_vars <- !numeric_vars
non_numeric_data <- data_cleaned[, non_numeric_vars, drop = FALSE]

# Binary set (keep exactly 2 unique values)
binary_cols <- c()
for (nm in intersect(c("Edema.sottoglottico","Granuloma","Muco.endolaringeo.spesso","Varici","Sesso.biologico"), names(non_numeric_data))) {
  if (length(unique(non_numeric_data[[nm]])) == 2) binary_cols <- c(binary_cols, nm)
}
binary_nonnumeric <- non_numeric_data[, binary_cols, drop = FALSE]
# recodes
if ("Sesso.biologico" %in% names(binary_nonnumeric)) {
  binary_nonnumeric$Sesso.biologico[binary_nonnumeric$Sesso.biologico == "Maschio"] <- 0
  binary_nonnumeric$Sesso.biologico[binary_nonnumeric$Sesso.biologico == "Femmina"] <- 2
}
for (col in names(binary_nonnumeric)) {
  binary_nonnumeric[[col]] <- as.character(binary_nonnumeric[[col]])
  binary_nonnumeric[[col]][grepl("Pres", binary_nonnumeric[[col]])] <- "2"
  binary_nonnumeric[[col]] <- as.numeric(binary_nonnumeric[[col]])
}

# Ordinal block (example selection; adapt to your columns)
ordinal_keep <- c('Obliterazione.ventricolare','Eritema...Iperemia','Edema.delle.CVV','Edema.diffuso.laringeo',
                  'Ipertrofia.della.commissura.posteriore','Spessore.del.muco','Accumulo','Localizzazione',
                  'Arresti.vibratori..Dx.','Arresti.vibratori..Sx.','Periodicità.della.vibrazione.glottica..Dx.',
                  'Periodicità.della.vibrazione.glottica..Sx.','Phase.Closure')
ordinal_keep <- intersect(ordinal_keep, names(data_cleaned))
ordinal_nn <- data_cleaned[, ordinal_keep, drop = FALSE]
ordinal_nn_mapped <- as.data.frame(lapply(ordinal_nn, function(x) as.numeric(as.factor(x))))

# Categoric (non-binary, non-ordinal leftovers)
categoric_nonnumeric <- non_numeric_data[, setdiff(names(non_numeric_data), c(names(binary_nonnumeric), names(ordinal_nn))), drop = FALSE]
categoric_nonnumeric_mapped <- as.data.frame(lapply(categoric_nonnumeric, function(x) as.numeric(as.factor(x))))

# Save panels for downstream scripts
saveRDS2(numeric_data, file.path(args$outdir, "numeric_data.rds"))
saveRDS2(binary_nonnumeric, file.path(args$outdir, "binary_nonnumeric.rds"))
saveRDS2(ordinal_nn, file.path(args$outdir, "ordinal_nn.rds"))
saveRDS2(ordinal_nn_mapped, file.path(args$outdir, "ordinal_nn_mapped.rds"))
saveRDS2(categoric_nonnumeric, file.path(args$outdir, "categoric_nonnumeric.rds"))
saveRDS2(categoric_nonnumeric_mapped, file.path(args$outdir, "categoric_nonnumeric_mapped.rds"))
