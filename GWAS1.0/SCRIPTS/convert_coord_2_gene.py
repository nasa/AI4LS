import sys
from pyensembl import EnsemblRelease

def main():
    chrom=int(sys.argv[1])
    pos=int(sys.argv[2])
    data=EnsemblRelease(75)
    genes=data.gene_names_at_locus(contig=chrom, position=pos)
    print(genes)
    return genes

if __name__ == "__main__":
    main()
