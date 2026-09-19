# 04_freq_tables_plots.R — frequency tables and stacked bars
suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(ggplot2)
})

source("00_utils.R")
args <- parse_args()
ensure_dirs(c(args$outdir, args$figdir))

binary_nonnumeric <- readRDS(file.path(args$outdir, "binary_nonnumeric.rds"))
categoric_nonnumeric <- readRDS(file.path(args$outdir, "categoric_nonnumeric.rds"))
ordinal_nn <- readRDS(file.path(args$outdir, "ordinal_nn.rds"))

# ---- Binary ----
binary_long <- tidyr::pivot_longer(binary_nonnumeric, everything(), names_to = "Variable")
counts_bin <- binary_long %>% group_by(Variable, value) %>% summarise(count = n(), .groups="drop")
total_bin <- counts_bin %>% group_by(Variable) %>% summarise(total = sum(count), .groups="drop")
counts_bin <- left_join(counts_bin, total_bin, by="Variable")
write.csv(counts_bin, file.path(args$outdir, "freqtable_binary.csv"), row.names = FALSE)
p <- ggplot(counts_bin, aes(x = Variable, y = count, fill = as.factor(value))) +
  geom_bar(stat = "identity") + geom_text(aes(label = count), position = position_stack(vjust = 0.5)) +
  theme_minimal() + theme(axis.text.x = element_text(angle=45,hjust=1)) +
  labs(fill="Value", title="Binary counts")
ggsave2("binary_counts.pdf", p, path = args$figdir, width=9, height=5)

# ---- Categoric ----
categoric_long <- tidyr::pivot_longer(categoric_nonnumeric, everything(), names_to = "Variable")
counts_cat <- categoric_long %>% group_by(Variable, value) %>% summarise(count = n(), .groups="drop")
total_cat <- counts_cat %>% group_by(Variable) %>% summarise(total = sum(count), .groups="drop")
counts_cat <- left_join(counts_cat, total_cat, by="Variable")
write.csv(counts_cat, file.path(args$outdir, "freqtable_categorics.csv"), row.names = FALSE)
p <- ggplot(counts_cat, aes(x = Variable, y = count, fill = value)) +
  geom_bar(stat = "identity") + geom_text(aes(label = count), position = position_stack(vjust = 0.5)) +
  theme_minimal() + theme(axis.text.x = element_text(angle=45,hjust=1)) +
  labs(fill="Category", title="Categoric counts")
ggsave2("categoric_counts.pdf", p, path = args$figdir, width=10, height=5)

# ---- Ordinal ----
ord_chr <- as.data.frame(lapply(ordinal_nn, as.character))
ordinal_long <- tidyr::pivot_longer(ord_chr, everything(), names_to = "Variable")
counts_ord <- ordinal_long %>% group_by(Variable, value) %>% summarise(count = n(), .groups="drop")
total_ord <- counts_ord %>% group_by(Variable) %>% summarise(total = sum(count), .groups="drop")
counts_ord <- left_join(counts_ord, total_ord, by="Variable")
write.csv(counts_ord, file.path(args$outdir, "freqtable_ordinal.csv"), row.names = FALSE)
p <- ggplot(counts_ord, aes(x = Variable, y = count, fill = value)) +
  geom_bar(stat = "identity") + geom_text(aes(label = count), position = position_stack(vjust = 0.5)) +
  theme_minimal() + theme(axis.text.x = element_text(angle=45,hjust=1)) +
  labs(fill="Level", title="Ordinal counts")
ggsave2("ordinal_counts.pdf", p, path = args$figdir, width=10, height=5)
