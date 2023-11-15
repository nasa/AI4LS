BiocManager::install("STRINGdb")
BiocManager::install("org.Mm.eg.db")
BiocManager::install("org.Hs.eg.db")
library(STRINGdb)
#library(org.Mm.eg.db)
library(org.Hs.eg.db)

args <- commandArgs(trailingOnly = TRUE)
print(args)
exprFile <- args[1]
#exprFile <- '/Users/jcasalet/Desktop/NASA/RADIO_GWAS/MY_TASKS/gene-summary-per-chrom.txt'
outFile <- args[2]
#outFile <- '/Users/jcasalet/Desktop/NASA/RADIO_GWAS/MY_TASKS/gene-names.txt'

expr <- read.csv(exprFile, header=TRUE, row.names=1, stringsAsFactors=TRUE, check.names=FALSE)

# get ENSEMBL:symbol mapping from org.Mm.eg.db database
# drop any genes that don't have a gene symbol
mapped <- na.omit(as.data.frame(mapIds(org.Mm.eg.db, keys=rownames(expr),
                                       keytype='ENSEMBL', column='SYMBOL', multiVals='first')))
colnames(mapped) <- 'symbol'

# # Convert and write out  data
symbolize_and_save <- function(df, fileName, takeLog){
  temp <- merge(df, mapped, by=0, all=FALSE, no.dups=FALSE) # merge gene symbols into expression df
  .rowNamesDF(temp, make.names=TRUE) <- temp$symbol # make gene symbols row names
  temp <- within(temp, rm(Row.names)) # remove residual columns
  temp <- within(temp, rm(symbol))
  if(takeLog == TRUE) {
    temp <- log2(temp+1)
  }
  write.csv(temp, fileName, row.names=TRUE, quote=FALSE) # write out
  return(temp)
}

all <- symbolize_and_save(expr, outFile, FALSE)

