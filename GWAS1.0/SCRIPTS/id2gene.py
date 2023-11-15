import mygene 

mg = mygene.MyGeneInfo()
ens = ['ENSG00000148795', 'ENSG00000165359', 'ENSG00000150676']
gList = mg.querymany(ens, scopes='ensembl.gene')

# [{'query': 'ENSG00000148795', '_id': '1586', '_score': 26.06642, 'entrezgene': '1586', 'name': 'cytochrome P450 family 17 subfamily A member 1', 'symbol': 'CYP17A1', 'taxid': 9606}, {'query': 'ENSG00000165359', '_id': '203522', '_score': 26.073044, 'entrezgene': '203522', 'name': 'integrator complex subunit 6 like', 'symbol': 'INTS6L', 'taxid': 9606}, {'query': 'ENSG00000150676', '_id': '220047', '_score': 26.067537, 'entrezgene': '220047', 'name': 'coiled-coil domain containing 83', 'symbol': 'CCDC83', 'taxid': 9606}]

print('ens_id', 'ens_symbol', 'ens_name')
for g in gList:
    # {'query': 'ENSG00000148795', '_id': '1586', '_score': 26.073044, 'entrezgene': '1586', 'name': 'cytochrome P450 family 17 subfamily A member 1', 'symbol': 'CYP17A1', 'taxid': 9606}
    ens_id=g['query']
    ens_name=g['name']
    ens_symbol=g['symbol']
    print(ens_id, '\t', ens_symbol, '\t', ens_name)

