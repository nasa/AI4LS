# r_scripts/go_enrichment.R
# GO and KEGG enrichment with clusterProfiler: over-representation (ORA) on gene lists and
# GSEA on ranked genes. GO works offline (org.Mm.eg.db / org.Hs.eg.db).
#
# ontologies: any of "BP", "MF", "CC" (GO) and "KEGG" (KEGG pathways). KEGG gene sets are
# downloaded from rest.kegg.jp once per species and kept for KEGG_CACHE_DAYS in this R session
# (the bioinformatics service keeps one R session); if the download fails, the KEGG rows of the
# summary say why and GO still runs. Genes are converted to Entrez IDs first, which for mouse and
# human are also KEGG's gene IDs.
#
# Inputs (written by src/go_enrichment.py):
#   lists.csv     columns: name, gene                 (one row per gene per list)
#   rankings.csv  columns: name, gene, score, score_type ("std" = signed, "pos" = positive only)
#   universe.csv  column:  gene                       (the genes the pipeline analysed)
# Genes may be symbols or Ensembl IDs (versions like .4 are fine).
#
# Outputs in output_dir:
#   go_enrichment_results.csv   one row per enriched term
#   go_enrichment_summary.csv   one row per (analysis, method, ontology), incl. mapping counts
# Returns the number of enriched terms.

suppressPackageStartupMessages({
  library(clusterProfiler)
  library(BiocParallel)
})
# Run single-process. gseGO -> fgsea otherwise uses BiocParallel's default MulticoreParam, which
# forks one worker per core. Forking the multi-threaded gRPC server process can deadlock (the call
# never returns) and, on many-core machines, multiplies memory use.
BiocParallel::register(BiocParallel::SerialParam(), default = TRUE)

.go_orgdb <- function(organism) {
  db <- switch(tolower(organism),
               mmu = "org.Mm.eg.db", mouse = "org.Mm.eg.db",
               hsa = "org.Hs.eg.db", human = "org.Hs.eg.db",
               stop(paste("Unsupported organism:", organism, "(use mmu or hsa)")))
  suppressPackageStartupMessages(library(db, character.only = TRUE))
  get(db)
}

# Map symbols and Ensembl IDs to Entrez IDs. Returns data.frame(input, ENTREZID), one row per input.
.to_entrez <- function(genes, orgdb) {
  genes <- unique(as.character(genes[!is.na(genes) & genes != ""]))
  out <- data.frame(input = character(), ENTREZID = character(), stringsAsFactors = FALSE)
  if (length(genes) == 0) return(out)

  is_ens <- grepl("^ENS[A-Z]*G[0-9]+", genes)
  sym <- genes[!is_ens]
  ens <- genes[is_ens]

  safe_bitr <- function(ids, from) {
    if (length(ids) == 0) return(NULL)
    tryCatch(suppressWarnings(suppressMessages(
      bitr(ids, fromType = from, toType = "ENTREZID", OrgDb = orgdb))),
      error = function(e) NULL)          # bitr errors when nothing maps
  }

  s <- safe_bitr(sym, "SYMBOL")
  if (!is.null(s) && nrow(s) > 0) {
    out <- rbind(out, data.frame(input = s$SYMBOL, ENTREZID = s$ENTREZID, stringsAsFactors = FALSE))
  }
  if (length(ens) > 0) {
    unversioned <- sub("\\.[0-9]+$", "", ens)
    e <- safe_bitr(unique(unversioned), "ENSEMBL")
    if (!is.null(e) && nrow(e) > 0) {
      back <- data.frame(input = ens, ENSEMBL = unversioned, stringsAsFactors = FALSE)
      e <- merge(back, e, by = "ENSEMBL")
      out <- rbind(out, data.frame(input = e$input, ENTREZID = e$ENTREZID, stringsAsFactors = FALSE))
    }
  }
  out[!duplicated(out$input), , drop = FALSE]   # one Entrez ID per input gene
}

# ── KEGG pathway gene sets ─────────────────────────────────────────────────────
KEGG_CACHE_DAYS <- 7
.kegg_cache <- new.env()

.kegg_code <- function(organism) {
  switch(tolower(organism), mmu = "mmu", mouse = "mmu", hsa = "hsa", human = "hsa",
         stop(paste("Unsupported organism for KEGG:", organism, "(use mmu or hsa)")))
}

# list(t2g = data.frame(term, gene = Entrez ID), t2n = data.frame(term, name)); errors if KEGG is unreachable
.kegg_sets <- function(organism) {
  code <- .kegg_code(organism)
  hit <- .kegg_cache[[code]]
  if (!is.null(hit) && difftime(Sys.time(), hit$time, units = "days") < KEGG_CACHE_DAYS) return(hit)
  download <- getFromNamespace("download_KEGG", "clusterProfiler")
  k <- suppressMessages(download(code, keggType = "KEGG", keyType = "kegg"))
  t2g <- data.frame(term = as.character(k$KEGGPATHID2EXTID[[1]]),
                    gene = sub("^[a-z]+:", "", as.character(k$KEGGPATHID2EXTID[[2]])),
                    stringsAsFactors = FALSE)
  t2n <- data.frame(term = as.character(k$KEGGPATHID2NAME[[1]]),
                    # "Glycolysis / Gluconeogenesis - Mus musculus (house mouse)" -> "Glycolysis / Gluconeogenesis"
                    name = sub(" - [A-Z][a-z]+ [a-z]+ \\([^)]*\\)$", "", as.character(k$KEGGPATHID2NAME[[2]])),
                    stringsAsFactors = FALSE)
  if (nrow(t2g) == 0) stop("KEGG returned no pathway genes for ", code)
  hit <- list(t2g = t2g, t2n = t2n, time = Sys.time())
  assign(code, hit, envir = .kegg_cache)
  cat(sprintf("KEGG %s: %d pathways, %d genes (downloaded)\n", code, length(unique(t2g$term)),
              length(unique(t2g$gene))))
  hit
}

.ont_label <- function(ont) if (ont == "KEGG") "KEGG" else paste0("GO:", ont)

.empty_terms <- function() {
  data.frame(analysis = character(), method = character(), ontology = character(),
             term_id = character(), description = character(), set_size = integer(),
             count = integer(), gene_ratio = character(), bg_ratio = character(),
             enrichment_score = numeric(), nes = numeric(), pvalue = numeric(),
             padj = numeric(), qvalue = numeric(), genes = character(),
             stringsAsFactors = FALSE)
}

.col <- function(df, name, default) if (name %in% names(df)) df[[name]] else rep(default, nrow(df))

.maybe_simplify <- function(x, do_simplify, cutoff, label) {
  if (!do_simplify || is.null(x) || nrow(as.data.frame(x)) < 2) return(x)
  tryCatch(clusterProfiler::simplify(x, cutoff = cutoff, by = "p.adjust", select_fun = min),
           error = function(e) {
             message(sprintf("simplify() skipped for %s: %s", label, conditionMessage(e)))
             x
           })
}

run_go_enrichment <- function(lists_path, rankings_path, universe_path, output_dir, organism,
                              ontologies, p_adjust_method, padj_cutoff, qvalue_cutoff,
                              min_gs_size, max_gs_size, do_simplify, simplify_cutoff) {
  orgdb <- .go_orgdb(organism)
  ontologies <- as.character(ontologies)
  min_gs_size <- as.integer(min_gs_size); max_gs_size <- as.integer(max_gs_size)
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

  universe <- read.csv(universe_path, stringsAsFactors = FALSE)$gene
  uni_map <- .to_entrez(universe, orgdb)
  universe_entrez <- unique(uni_map$ENTREZID)
  cat(sprintf("Universe: %d genes, %d mapped to Entrez\n", length(unique(universe)), length(universe_entrez)))

  kegg <- NULL; kegg_error <- ""
  if ("KEGG" %in% ontologies) {
    kegg <- tryCatch(.kegg_sets(organism), error = function(e) {
      kegg_error <<- paste("KEGG download failed:", conditionMessage(e)); NULL })
    if (is.null(kegg)) message(kegg_error)
  }

  terms <- list(); sums <- list()
  add_summary <- function(analysis, method, ont, n_input, n_mapped, n_terms, note = "") {
    sums[[length(sums) + 1]] <<- data.frame(
      analysis = analysis, method = method, ontology = ont, n_input = n_input,
      n_mapped = n_mapped, n_terms = n_terms, note = note, stringsAsFactors = FALSE)
  }

  # ── Over-representation on gene lists ─────────────────────────────────────
  lists <- if (file.exists(lists_path)) read.csv(lists_path, stringsAsFactors = FALSE) else NULL
  if (!is.null(lists) && nrow(lists) > 0) {
    for (nm in unique(lists$name)) {
      genes <- lists$gene[lists$name == nm]
      m <- .to_entrez(genes, orgdb)
      entrez <- intersect(unique(m$ENTREZID), universe_entrez)
      for (ont in ontologies) {
        label <- sprintf("ORA %s %s", nm, .ont_label(ont))
        if (length(entrez) < 3) {
          add_summary(nm, "ora", ont, length(unique(genes)), length(entrez), 0L,
                      "fewer than 3 genes mapped; skipped")
          next
        }
        if (ont == "KEGG" && is.null(kegg)) {
          add_summary(nm, "ora", ont, length(unique(genes)), length(entrez), 0L, kegg_error)
          next
        }
        note <- ""
        fail <- function(e) { note <<- paste("failed:", conditionMessage(e)); message(label, " ", note); NULL }
        res <- if (ont == "KEGG") {
          tryCatch(enricher(gene = entrez, universe = universe_entrez,
                            TERM2GENE = kegg$t2g, TERM2NAME = kegg$t2n,
                            pAdjustMethod = p_adjust_method, pvalueCutoff = padj_cutoff,
                            qvalueCutoff = qvalue_cutoff, minGSSize = min_gs_size,
                            maxGSSize = max_gs_size), error = fail)
        } else {
          tryCatch(enrichGO(gene = entrez, universe = universe_entrez, OrgDb = orgdb,
                            keyType = "ENTREZID", ont = ont,
                            pAdjustMethod = p_adjust_method, pvalueCutoff = padj_cutoff,
                            qvalueCutoff = qvalue_cutoff, minGSSize = min_gs_size,
                            maxGSSize = max_gs_size, readable = TRUE), error = fail)
        }
        if (ont == "KEGG" && !is.null(res)) {   # Entrez IDs -> symbols in the gene column
          res <- tryCatch(setReadable(res, OrgDb = orgdb, keyType = "ENTREZID"), error = function(e) res)
        }
        if (ont != "KEGG") res <- .maybe_simplify(res, do_simplify, simplify_cutoff, label)   # GO only
        df <- if (is.null(res)) data.frame() else as.data.frame(res)
        if (nrow(df) > 0) {
          df <- df[df$p.adjust <= padj_cutoff, , drop = FALSE]
          q <- .col(df, "qvalue", NA_real_)
          df <- df[is.na(q) | q <= qvalue_cutoff, , drop = FALSE]
        }
        if (nrow(df) > 0) {
          set_size <- suppressWarnings(as.integer(sub("/.*$", "", df$BgRatio)))
          terms[[length(terms) + 1]] <- data.frame(
            analysis = nm, method = "ora", ontology = ont, term_id = df$ID,
            description = df$Description, set_size = set_size, count = as.integer(df$Count),
            gene_ratio = df$GeneRatio, bg_ratio = df$BgRatio,
            enrichment_score = NA_real_, nes = NA_real_, pvalue = df$pvalue,
            padj = df$p.adjust, qvalue = .col(df, "qvalue", NA_real_), genes = df$geneID,
            stringsAsFactors = FALSE)
        }
        add_summary(nm, "ora", ont, length(unique(genes)), length(entrez), nrow(df), note)
        cat(sprintf("%s: %d terms\n", label, nrow(df)))
      }
    }
  }

  # ── GSEA on rankings ───────────────────────────────────────────────────────
  rankings <- if (file.exists(rankings_path)) read.csv(rankings_path, stringsAsFactors = FALSE) else NULL
  if (!is.null(rankings) && nrow(rankings) > 0) {
    for (nm in unique(rankings$name)) {
      r <- rankings[rankings$name == nm, , drop = FALSE]
      score_type <- r$score_type[1]
      m <- .to_entrez(r$gene, orgdb)
      r <- merge(r, m, by.x = "gene", by.y = "input")
      # Several genes can map to one Entrez ID: keep the strongest score
      r <- r[order(-abs(r$score)), , drop = FALSE]
      r <- r[!duplicated(r$ENTREZID), , drop = FALSE]
      gl <- setNames(r$score, r$ENTREZID)
      gl <- sort(gl[is.finite(gl)], decreasing = TRUE)
      for (ont in ontologies) {
        label <- sprintf("GSEA %s %s", nm, .ont_label(ont))
        n_in <- length(unique(rankings$gene[rankings$name == nm]))
        if (length(gl) < 15) {
          add_summary(nm, "gsea", ont, n_in, length(gl), 0L, "fewer than 15 ranked genes mapped; skipped")
          next
        }
        if (ont == "KEGG" && is.null(kegg)) {
          add_summary(nm, "gsea", ont, n_in, length(gl), 0L, kegg_error)
          next
        }
        note <- ""
        run <- function(extra) {
          set.seed(42)
          common <- list(geneList = gl, minGSSize = min_gs_size, maxGSSize = max_gs_size,
                         pvalueCutoff = padj_cutoff, pAdjustMethod = p_adjust_method,
                         eps = 0, verbose = FALSE, seed = TRUE)
          if (ont == "KEGG") {
            do.call(GSEA, c(common, list(TERM2GENE = kegg$t2g, TERM2NAME = kegg$t2n), extra))
          } else {
            do.call(gseGO, c(common, list(ont = ont, OrgDb = orgdb, keyType = "ENTREZID"), extra))
          }
        }
        res <- tryCatch(run(list(scoreType = score_type)), error = function(e) {
          # Older clusterProfiler versions don't pass scoreType through to fgsea
          message(label, ": retrying without scoreType (", conditionMessage(e), ")")
          tryCatch(run(list()), error = function(e2) {
            note <<- paste("failed:", conditionMessage(e2)); message(label, " ", note); NULL })
        })
        if (!is.null(res)) res <- tryCatch(setReadable(res, OrgDb = orgdb, keyType = "ENTREZID"),
                                           error = function(e) res)
        if (ont != "KEGG") res <- .maybe_simplify(res, do_simplify, simplify_cutoff, label)   # GO only
        df <- if (is.null(res)) data.frame() else as.data.frame(res)
        if (nrow(df) > 0) df <- df[df$p.adjust <= padj_cutoff, , drop = FALSE]
        if (nrow(df) > 0) {
          q <- if ("qvalue" %in% names(df)) df$qvalue else .col(df, "qvalues", NA_real_)
          terms[[length(terms) + 1]] <- data.frame(
            analysis = nm, method = "gsea", ontology = ont, term_id = df$ID,
            description = df$Description, set_size = as.integer(df$setSize),
            count = as.integer(lengths(strsplit(df$core_enrichment, "/"))),
            gene_ratio = NA_character_, bg_ratio = NA_character_,
            enrichment_score = df$enrichmentScore, nes = df$NES, pvalue = df$pvalue,
            padj = df$p.adjust, qvalue = q, genes = df$core_enrichment,
            stringsAsFactors = FALSE)
        }
        add_summary(nm, "gsea", ont, n_in, length(gl), nrow(df), note)
        cat(sprintf("%s: %d terms\n", label, nrow(df)))
      }
    }
  }

  all_terms <- if (length(terms)) do.call(rbind, terms) else .empty_terms()
  all_summary <- if (length(sums)) do.call(rbind, sums) else
    data.frame(analysis = character(), method = character(), ontology = character(),
               n_input = integer(), n_mapped = integer(), n_terms = integer(), note = character())
  write.csv(all_terms, file.path(output_dir, "go_enrichment_results.csv"), row.names = FALSE)
  write.csv(all_summary, file.path(output_dir, "go_enrichment_summary.csv"), row.names = FALSE)
  cat(sprintf("Enrichment complete: %d terms\n", nrow(all_terms)))
  nrow(all_terms)
}
