#!/bin/bash


#./parse_log_R_barplot.py outcome_pdb_graphs_combined plots/pdb Proteins YF 7500
#./parse_log_R_barplot.py outcome_citation_disconnected_monthly plots/citation Citation
#./parse_log_R_barplot.py outcome_dblp_1990_2010_5 plots/dblp DBLP
#./parse_log_R_barplot.py nci_v1.log plots/nci NCI YI 20000
#./parse_log_R_barplot.py outcome_fsg_500K plots/syn_500K "Syn 500K" YI
#./parse_log_R_barplot.py outcome_fsg_1000K plots/syn_1000K "Syn 1000K" YI

export step=2

./parse_log_R_barplot.py outcome_pdb_6875_KDD_2014 plots/pdb_6875 Proteins $step YF 6000
./parse_log_R_barplot.py outcome_citation_KDD_2014 plots/citation Citation $step
./parse_log_R_barplot.py outcome_dblp_KDD_2014 plots/dblp DBLP 1 N 13
./parse_log_R_barplot.py outcome_nci_KDD_2014 plots/nci NCI $step YI 20000
./parse_log_R_barplot.py outcome_fsg_500K_KDD_2014 plots/syn_500K "Syn 500K" 1 YI
./parse_log_R_barplot.py outcome_fsg_1000K_KDD_2014 plots/syn_1000K "Syn 1000K" 1 YI
