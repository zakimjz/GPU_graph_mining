STYLE_FILES:=algorithmic.sty algorithm.sty grph_db.sty gspan.sty 
TEX_FILES:=gspan_cuda.tex   basic_notion.tex embedding_storage.tex computing_extensions.tex support_computation.tex grph_db_example.tex intro.tex  graph_mining.tex

all: gspan_cuda.pdf gspan_cuda_kdd.pdf

#gspan_cuda_kdd: gspan_cuda_kdd.bbl gspan_cuda_kdd.tex
#	pdflatex gspan_cuda_kdd.tex


#gspan_cuda.pdf: gspan_cuda.bbl  gspan_cuda.tex
#	pdflatex gspan_cuda.tex
#	pdflatex gspan_cuda.tex

gspan_cuda.pdf: gspan_cuda.bbl  gspan_cuda.tex $(STYLE_FILES) $(TEX_FILES) 
	pdflatex gspan_cuda.tex 
	pdflatex gspan_cuda.tex


gspan_cuda_kdd.pdf: $(STYLE_FILES) $(TEX_FILES) gspan_cuda_kdd.tex gspan_cuda_kdd.bbl 
	pdflatex gspan_cuda_kdd.tex 
	pdflatex gspan_cuda_kdd.tex 

gspan_cuda.bbl: papers.bib 
	pdflatex gspan_cuda.tex
	bibtex gspan_cuda

gspan_cuda_kdd.bbl: papers.bib
	pdflatex gspan_cuda_kdd.tex
	bibtex gspan_cuda_kdd


clean:
	rm -rf gspan_cuda.pdf *.aux *.log *.tex.aux *~ *.bbl *.blg gspan_cuda_kdd.pdf