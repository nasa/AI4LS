import gseapy as gp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px

def gsea(geneList, gmtFile, pcutoff, nPathways, 
         figSize, labelSize, title, filename, streamlit_mode = False):
    '''
    Performs gene set enrichment analysis on a given gene list using a GMT pathway file to search. 
    Accesses ENRICHR through GSEAPY: https://gseapy.readthedocs.io/en/latest/gseapy_example.html#2.-Enrichr-Example
    Several GMT files are available for download here: https://www.gsea-msigdb.org/gsea/downloads.jsp#msigdb
    
    geneList = list of genes to investigate
    gmtFile = path to gmt (pathways) file to search (string)
    pcutoff = pvalue cutoff to use for plotting (integer)
    nPathways = top number of pathways to plot (eg 10)
    figSize = figure size of the bar plot (tuple) (eg (3,7))
    labelSize = font size of the y and x axis (eg 20)
    title = string of the plot title
    filename = string of where to save plot
    
    # how to get genes associated with a particular pathway in ENRICHR
    list(enr.results.sub.sort[enr.results.sub.sort['Term'] == 'PATHWAY_NAME']['Genes'])[0].split(';')
    '''
    enr = gp.enrichr(gene_list=geneList,
                 gene_sets=gmtFile,
                 background='hsapiens_gene_ensembl', #'mmusculus_gene_ensembl'
                 cutoff=0, 
                 verbose=False)
    
    print('Enrichr pathway results dataframe:')
    print(enr.results.shape)
    
    # Subset to pathways within a certain Pvalue cutoff 
    enr.results.sub = enr.results[(enr.results['P-value'] < pcutoff)]
    print('Enrichr pathway results dataframe after filtering:')
    print(enr.results.sub.shape)
    
    # Create new column with -log of pvalue
    enr.results.sub['pvalue(-log10)'] = -np.log(enr.results.sub['P-value'])
    
    # Sort by -log pvalue (highest on top)
    enr.results.sub.sort = enr.results.sub.sort_values(by=['pvalue(-log10)'], ascending=False)
    
    ### PLOT
    f, ax = plt.subplots(figsize=figSize)
    plt.rc('xtick', labelsize=labelSize)  
    plt.rc('ytick', labelsize=labelSize)  

    plot = px.bar(enr.results.sub.sort[:nPathways], x='pvalue(-log10)', y='Term')

    ax.set_ylabel('')
    ax.set_xlabel('-log10(pvalue)',fontsize=labelSize)

    ax.set_title(title,fontsize=17)
    
    if streamlit_mode == True:
        st.write(plot)
    
    
    plt.savefig(filename, dpi=500, bbox_inches='tight')
    
    # Return results df used in plotting
    return enr.results.sub.sort
    
    

# EXAMPLE FUNCTION CALL 

# myGeneList = ['SERPINE1','MAPK13','SLIT2','TENM4','HMHB1','GRAMD1C','VPREB3','MS4A1','IL32','TCL1A','MDK','IFITM1','ALAS2','CA2']

# gsea( myGeneList, 
#      './gmt/c5.bp.v7.1.symbols.gmt', 
#      pcutoff=0.05, nPathways=10, figSize=(3,5), labelSize=20,
#      title='HSPC up', filename='')





