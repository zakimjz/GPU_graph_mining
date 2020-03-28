#!/usr/bin/python
import sys
import os
import math
import pdb
import glob
import re
from  collections import defaultdict

metrics_to_plot = ["time", "sm_efficiency", "branch_efficiency", "achieved_occupancy", "gld_transactions_per_request", "gst_transactions_per_request", "gst_throughput", "gld_throughput", "gld_efficiency", "gst_efficiency", "dram_utilization"]
metrics_factor = [1, 1, 1, 100, 1, 1, 1, 1, 1, 1, 1]
metrics_unit = ["Percentage", "Percentage", "Percentage", "Percentage", "Number of transactions", "Number of transactions", "GB/sec", "GB/sec", "Percentage", "Percentage", "Unit"]
ylims_ext = dict()
dataset_labels = dict(
	pdb='Protein',
	dblp='DBLP',
	citation='Citation',
	nci='NCI',           
)

#kernel_list = ["store_embedding_extension", "store_mapped_db_flag_multiple_dfs", "store_validity", "store_graph_id"]
kernel_list = ["store_embedding_ext", "store_mapped_db_flag", "store_validity", "store_graph_id"]
#kernel_plot_labels = ["update-EXT$_k$", "update-$\mathbf{F}$", "update-$\mathbf{V}$", "update-$\mathbf{B}$"]
kernel_plot_labels = ["expression(paste(\"update-\",bold(EXT)[k]))", "expression(paste(\"update-\",bold(F)))", "expression(paste(\"update-\",bold(V)))", "expression(paste(\"update-\",bold(B)))"]
kernel_string = ""


values = defaultdict(lambda: defaultdict(dict))
max_val = dict()

#for i in range(0,len(kernel_list)):
#	if i < len(kernel_list) - 1:
#		kernel_string += "\"" + kernel_list[i] + "\" ,"
#	else:
#		kernel_string += "\"" + kernel_list[i] + "\""
for i in range(0,len(kernel_plot_labels)):
	if i < len(kernel_plot_labels) - 1:
		kernel_string +=  kernel_plot_labels[i] + " , "
	else:
		kernel_string +=  kernel_plot_labels[i]

for m in metrics_to_plot:	
	if "request" in m or "utilization" in m:
		ylims_ext[m] = "3"
	else:
		ylims_ext[m] = "23"
		 

# get time
def get_time(infile,dataset):

	f = open(infile,"r")

	for line in f:
		for k in kernel_list:
			if k in line:
				w = line.split()
				#print k, w[0]
				val = w[0][0:len(w[0])-1]
				values["time"][dataset][k] = val
				#print k, values["time"][dataset][k]
				#update max_val 
				if "time" not in max_val.keys() or max_val["time"] < float(val):
					max_val["time"] = float(val)
				break

	f.close()

# get metrics
def get_metrics(infile,dataset):

	f = open(infile+"_metrics","r")
	
	in_kernel_desc = False
	for line in f:

		if "Kernel" in line:
			for k in kernel_list:
				if k in line:
					in_kernel_desc = True
					print k
					break
				else:
					in_kernel_desc = False
			continue

		if in_kernel_desc == True:
			#print line			
			w = line.split()
 			metric = w[1] 
			pat = re.compile('[\d\.]+')
			val = pat.findall(w[len(w)-1])[0]
			#if val[0] == "(":
			#	val = val[1:]
			#if val[len(val)-1] == ")" or val[len(val)-1] == "%":
			#	val = val[:len(val) -1]
			#print k, w[1], val
			values[metric][dataset][k] = str( float(val) * metrics_factor[metrics_to_plot.index(metric)] )
			#print k, w[1], values[w[1]][dataset][k]
			if metric not in max_val.keys() or max_val[metric] < float(values[metric][dataset][k]):
				max_val[metric] = float(values[metric][dataset][k])

	f.close()

def do_bar_plot_metric(metric,file_prefix):
	
	# Write the R script
	f = open(file_prefix+".r","w")

	f.write("#!/usr/bin/Rscript\n")
	f.write("data <- read.table(\""+file_prefix+".dat\", header=T, sep=\"\\t\") \n")
	#f.write("max_y <- max(data$seq,data$par) \n")
	f.write("max_y <- "+str(max_val[metric])+"\n")
	f.write("plot_colors <- c(\"red\",\"green\",\"blue\",\"cyan\")\n")
	f.write("postscript(\""+file_prefix+".eps\", bg=\"white\", paper=\"letter\") \n")
	f.write("par(mar=c(4.2,5.2,4,4.2)+.1)\n") 
	f.write("barplot(t(as.matrix(data[,2:5])), ylim=c(0,max_y+"+ylims_ext[metric] +"), names.arg=data$dataset, cex.names=2.5, cex.axis=2.75, col=plot_colors, beside=TRUE) \n")
	f.write("par(new=TRUE) \n")
	f.write("title(main=\""+metric+"\", col.main=\"blue\", font.main=4, cex.main=3.5) \n")
	f.write("title(xlab=\"Datasets\", line=3 , font.lab=4, col.lab=rgb(0,0.5,0), cex.lab=3.0 ) \n")
	#f.write("title(ylab=\"percentage \", line=3, font.lab=4, col.lab=rgb(0,0.5,0), cex.lab=3.25) \n")
	f.write("title(ylab=\""+ metrics_unit[metrics_to_plot.index(metric)] +"\", line=3, font.lab=4, col.lab=rgb(0,0.5,0), cex.lab=3.0) \n")
	#f.write("legend(\"topright\", substring(rownames(t(data))[2:4],12), cex=2, fill=plot_colors) \n")
	f.write("legend(\"top\",c( "+ kernel_string +" ), cex=2.0, ncol=2, inset=c(0.2,0), fill=plot_colors, bty=\"n\") \n")
	f.write("box() \n")
	# Turn off device driver (to flush output to eps)
	f.write("dev.off() \n")

	f.close()
	# change the permission of the R script to executable
	os.chmod(file_prefix+".r",0755)
	# run the R script and produce the eps file 
	os.system("./"+file_prefix+".r")


if __name__ == "__main__":

	if len(sys.argv) < 2:
		print "usage :" +sys.argv[0] + " <input-file prefix (IN DOUBLE QUOTES)> "
		sys.exit()

	infile_pattern = sys.argv[1]
	files = glob.glob(infile_pattern)
	print files
	for inf in files:
		print inf
		w = inf.split("-")[1][1:] # dataset has the format ...-D<dataset>...
		dataset = re.split("[\._]*",w)[0]  
		print dataset
		get_time(inf, dataset_labels[dataset])
		get_metrics(inf, dataset_labels[dataset])


	for m in metrics_to_plot:
		of = open("kernels_"+m+".dat","w")
		of.write("dataset")
		for k in kernel_list:
			of.write("\t"+k)
		of.write("\n")

		for dataset in values[m].keys():
			of.write(dataset.title())
			for k in kernel_list:
				of.write("\t"+values[m][dataset][k])
			of.write("\n")
		of.close()	
			
		do_bar_plot_metric(m,"kernels_"+m)


