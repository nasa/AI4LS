# Smoke test for go_enrichment.R, run inside the bioinformatics container:
#   docker compose exec bioinformatics_service Rscript /app/r_scripts/test_go_enrichment.R
# Uses real mouse genes: a skeletal-muscle gene list inside a 2,000-gene background.
# Expected: muscle-related GO:BP terms (e.g. "muscle contraction") from both ORA and GSEA.

source("/app/r_scripts/go_enrichment.R")
suppressPackageStartupMessages(library(org.Mm.eg.db))
set.seed(1)

muscle <- c("Acta1", "Tnnc2", "Tnnt3", "Tnni2", "Ckm", "Myh1", "Myh2", "Myh4", "Mylpf", "Myl1",
            "Atp2a1", "Pvalb", "Actn3", "Tcap", "Casq1", "Ryr1", "Des", "Ttn", "Neb", "Myoz1",
            "Myot", "Ldb3", "Tnnt1", "Tnnc1", "Myl2", "Myh7", "Cav3", "Mb", "Smpx", "Xirp2")
all_symbols <- keys(org.Mm.eg.db, keytype = "SYMBOL")
universe <- unique(c(muscle, sample(setdiff(all_symbols, muscle), 1970)))
universe <- c(universe, "ENSMUSG00000029580.3", "Mup11_1")   # an Ensembl ID and a suffixed symbol (cleaned in Python normally)

dir <- file.path(tempdir(), "go_test"); dir.create(dir, showWarnings = FALSE)
write.csv(data.frame(name = "muscle_list", gene = muscle), file.path(dir, "lists.csv"), row.names = FALSE)
scores <- rnorm(length(universe))
scores[universe %in% muscle] <- scores[universe %in% muscle] + 4        # muscle genes ranked high
write.csv(rbind(
  data.frame(name = "signed", gene = universe, score = scores, score_type = "std"),
  data.frame(name = "positive", gene = universe, score = rank(scores) / length(scores), score_type = "pos")),
  file.path(dir, "rankings.csv"), row.names = FALSE)
write.csv(data.frame(gene = universe), file.path(dir, "universe.csv"), row.names = FALSE)

t0 <- Sys.time()
n <- run_go_enrichment(file.path(dir, "lists.csv"), file.path(dir, "rankings.csv"),
                       file.path(dir, "universe.csv"), dir, "mmu", c("BP"), "BH",
                       0.05, 0.2, 10L, 500L, TRUE, 0.7)
cat(sprintf("\nFinished in %.0f s with %d terms\n\n", as.numeric(difftime(Sys.time(), t0, units = "secs")), n))

res <- read.csv(file.path(dir, "go_enrichment_results.csv"))
print(read.csv(file.path(dir, "go_enrichment_summary.csv")))
for (a in unique(res$analysis)) {
  cat(sprintf("\nTop terms for %s:\n", a))
  r <- res[res$analysis == a, ]
  print(head(r[order(r$padj), c("method", "term_id", "description", "nes", "padj", "count")], 5), row.names = FALSE)
}
ok <- n > 0 && any(grepl("muscle", res$description, ignore.case = TRUE))
cat(if (ok) "\nPASS: muscle terms found\n" else "\nCHECK: no muscle terms found; see the messages above\n")
