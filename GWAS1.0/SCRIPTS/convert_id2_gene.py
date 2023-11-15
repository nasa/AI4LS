import mygene
import sys

def main():
    mg = mygene.MyGeneInfo()
    ens=[sys.argv[1]]
    ginfo = mg.querymany(ens, scopes='ensembl.gene')
    return ginfo[0]['symbol']


if __name__ == "__main__":
    main()
