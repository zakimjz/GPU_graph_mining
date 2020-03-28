#!/usr/bin/python
import sys
import os
import math
import pdb
from  collections import defaultdict

if len(sys.argv) < 4:
	print "usage : parse_output_gnu_plot.py <input-file> <output-dir and file prefix> <dataset name> [<support step size>] [<labels_as_K> (YI YF or N)= (F=float I=int) ] [<max support value>]"
	sys.exit()

infile = sys.argv[1]
outfile = sys.argv[2]
dataset = sys.argv[3]
support_max = -1 
labels_as_K = ""

if len(sys.argv) >4:
	step_size = int(sys.argv[4])

if len(sys.argv) > 5: 
	labels_as_K = sys.argv[5].upper()

if len(sys.argv) > 6:
	support_max = int(sys.argv[6]) 

support_all  = defaultdict(list)
seq_time_all = defaultdict(dict)
par_time_all = defaultdict(dict)

plot_colors = ["red", "blue", "green", "orange", "black"]
density = [10, 10, 10, 10, 10]
angle = [30, 90, 0, 150, 60]
marker = [16, 18, 15, 19]

f = open(infile,"r")
readFlag = False
for line in f:
	line.strip()
	w = line.split()
	if len(w) == 0:
		if readFlag == True:
			readFlag = False
		continue
	if readFlag:
		support_all[gspan_cuda_version].append(int(w[0]))
		seq_time_all[gspan_cuda_version][w[0]] = float(w[1])
		par_time_all[gspan_cuda_version][w[0]] = float(w[2])

	if w[0] == "GSPAN" and w[1] == "CUDA" and w[2] == "version:":
		gspan_cuda_version = w[3] 
		print gspan_cuda_version	
	if w[0] == "support":
		readFlag = True
f.close()


def do_bar_plot_seq_vs_par(support,seq_time,par_time, outfile, filltype, ver):

	support = sorted(support)

	f = open(infile+"_seq_vs_par"+ver+".dat","w")
	f.write("support\tseq\tpar\tspeedup\n")
	for i in xrange(0,len(support),step_size):
		s = support[i]
		if support_max < 0 or s <= support_max:
			if labels_as_K == "YF":	
				f.write(str(s*1.0/1000)+"K\t"+str(seq_time[str(s)])+"\t"+str(par_time[str(s)])+"\t"+str((seq_time[str(s)]*1.0)/par_time[str(s)])+"\n")
			elif labels_as_K == "YI":	
				f.write(str(s/1000)+"K\t"+str(seq_time[str(s)])+"\t"+str(par_time[str(s)])+"\t"+str((seq_time[str(s)]*1.0)/par_time[str(s)])+"\n")
			else:
				f.write(str(s)+"\t"+str(seq_time[str(s)])+"\t"+str(par_time[str(s)])+"\t"+str((seq_time[str(s)]*1.0)/par_time[str(s)])+"\n")

	f.close()

	max_y = max( max([v for (k,v) in seq_time.items()] ), max([v for (k,v) in par_time.items()]) )
	#print max_y
	if max_y < 10:
		ytick = math.ceil(max_y/5)
	else:
		ytick = math.ceil(max_y/50)*10

	# Write the R script
	f = open(outfile+".r","w")

	f.write("#!/usr/bin/Rscript\n")
	f.write("data <- read.table(\""+infile+"_seq_vs_par.dat\", header=T, sep=\"\\t\") \n")
	f.write("max_y <- max(data$seq,data$par) \n")
	f.write("plot_colors <- c(\"red\",\"green\")\n")
	f.write("plot_density <- c(14,10)\n")
	f.write("plot_angle <- c(30,90)\n")
	f.write("postscript(\""+outfile+".eps\", bg=\"white\", paper=\"letter\") \n")
	f.write("par(mar=c(4.2,5.2,4,4.2)+.1)\n") 
	if filltype == "color":
		f.write("barplot(t(as.matrix(data[,2:3])), ylim=c(0,max_y+2.0), names.arg=data$support, cex.names=2.5, cex.axis=2.75, col=plot_colors, beside=TRUE) \n")
	elif filltype == "pattern":
		f.write("barplot(t(as.matrix(data[,2:3])), ylim=c(0,max_y+2.0), names.arg=data$support, cex.names=2.5, cex.axis=2.75, density=plot_density, angle=plot_angle, beside=TRUE) \n")
	else:
		print "wrong filltype"
		sys.exit()

	f.write("par(new=TRUE) \n")

	if filltype == "color":
		f.write("plot(data$speedup, type=\"o\", lwd=3.0, col=\"blue\", ylim=c(0,max(data$speedup)+1.8), cex=4.0, axes=FALSE, ann=FALSE)\n")
		#f.write("title(main=\"Seq vs GPU graph mining\", col.main=\"blue\", font.main=4, cex.main=3.5) \n")
		f.write("title(main=\"Seq vs GPU ("+dataset+" Dataset)\", col.main=\"blue\", font.main=4, cex.main=3.5) \n")
		f.write("title(xlab=\"Support\", line=3 , font.lab=4, col.lab=rgb(0,0.5,0), cex.lab=3.25 ) \n")
		f.write("title(ylab=\"Time in seconds\", line=3 ,font.lab=4, col.lab=rgb(0,0.5,0), cex.lab=3.25) \n")
		f.write("axis(4, col.lab=rgb(0,0.5,0), cex.axis=2.75) \n")  #put the axis on the right side
		f.write("mtext(\"Speedup\", side=4, line=3, font=4, col=rgb(0,0.5,0), cex=3.25 ) \n")
		f.write("legend(\"top\",c(\"Seq\", \"GPU\"), cex=3, fill=plot_colors, horiz=TRUE) \n")
	elif filltype == "pattern":
		f.write("plot(data$speedup, type=\"o\", lwd=3.0, ylim=c(0,max(data$speedup)+1.8), cex=4.0, axes=FALSE, ann=FALSE)\n")
		f.write("title(main=\"Seq vs GPU ("+dataset+" Dataset)\", font.main=4, cex.main=3.5) \n")
		f.write("title(xlab=\"Support\", line=3 , font.lab=4, cex.lab=3.25 ) \n")
		f.write("title(ylab=\"Time in seconds\", line=3 ,font.lab=4, cex.lab=3.25) \n")
		f.write("axis(4, cex.axis=2.75) \n")  #put the axis on the right side
		f.write("mtext(\"Speedup\", side=4, line=3, font=4, cex=3.25 ) \n")
		f.write("legend(\"top\",c(\"Seq\", \"GPU\"), cex=3, density=c(14,10), angle=c(30,90), horiz=TRUE) \n")
	else:
		print "wrong filltype"
		sys.exit()


	f.write("box() \n")
	# Turn off device driver (to flush output to eps)
	f.write("dev.off() \n")

	f.close()
	# change the permission of the R script to executable
	os.chmod(outfile+".r",0755)
	# run the R script and produce the eps file 
	os.system("./"+outfile+".r")

def do_bar_plot_seq_vs_par2(supports, seq_time, par_time, par_versions, par_labels, outfile,filltype):
	support = []
	for k in supports.keys():
		support = supports[k]	# get any of the support lists
		break

	support = sorted(support)

	f = open(infile+"_seq_vs_par2.dat","w")

	f.write("support\tseq")
	for v in par_versions:
		f.write("\t"+v)
	for v in par_versions:
		f.write("\tspeedup_"+v)
	f.write("\n")

	for i in xrange(0,len(support),step_size):
		s = support[i]
		if support_max < 0 or s <= support_max:
			if labels_as_K == "YF":	
				f.write(str(s*1.0/1000)+"K\t"+str(seq_time[str(s)]))
			elif labels_as_K == "YI":	
				f.write(str(s/1000)+"K\t"+str(seq_time[str(s)]))
			else:
				f.write(str(s)+"\t"+str(seq_time[str(s)]))

			for v in par_versions:
				f.write("\t"+str(par_time[v][str(s)]))
			for v in par_versions:
				f.write("\t"+str((seq_time[str(s)]*1.0)/par_time[v][str(s)]))
			f.write("\n")

	f.close()

	max_y = 0
	for ver in par_versions:
		y = max( max([v for (k,v) in seq_time.items()] ), max([v for (k,v) in par_time[ver].items()]) )
		if y > max_y:
			max_y = y
	#print max_y
	if max_y < 10:
		ytick = math.ceil(max_y/5)
	else:
		ytick = math.ceil(max_y/50)*10

	# Write the R script
	f = open(outfile+".r","w")

	f.write("#!/usr/bin/Rscript\n")
	f.write("data <- read.table(\""+infile+"_seq_vs_par2.dat\", header=T, sep=\"\\t\") \n")
	f.write("max_y <- max(data$seq")
	for v in par_versions:
        	f.write(",data$"+v)
        f.write(") \n")

	if filltype == "color":
		f.write("plot_colors <- c(")
		#length = 1 + len(par_versions)
		for i in range(len(par_versions)):
			f.write("\""+plot_colors[i]+"\",")
		f.write("\""+plot_colors[len(par_versions)]+"\")\n")
	elif filltype == "pattern":
		f.write("plot_density <- c(")
		for i in range(len(par_versions)):
			f.write(""+str(density[i])+",")
		f.write(""+str(density[len(par_versions)])+")\n")
		f.write("plot_angle <- c(")
		for i in range(len(par_versions)):
			f.write(""+str(angle[i])+",")
		f.write(""+str(angle[len(par_versions)])+")\n")
	else:
		print "wrong filltype"
		sys.exit()
	

	f.write("postscript(\""+outfile+".eps\", bg=\"white\", paper=\"letter\") \n")
	f.write("par(mar=c(4.2,5.2,4,4.2)+.1)\n") 
	
	if filltype == "color":
		f.write("barplot(t(as.matrix(data[,2:"+str(len(par_versions)+2)+"])), ylim=c(0,max_y+2.0), names.arg=data$support, cex.names=2.5, cex.axis=2.75, col=plot_colors, beside=TRUE) \n")
	elif filltype == "pattern":
		f.write("barplot(t(as.matrix(data[,2:"+str(len(par_versions)+2)+"])), ylim=c(0,max_y+2.0), names.arg=data$support, cex.names=2.5, cex.axis=2.75, density=plot_density, angle=plot_angle, beside=TRUE) \n")
	else:
		print "wrong filltype"
		sys.exit()

	f.write("par(new=TRUE) \n")

	f.write("max_y <- max(")
	for i in range(len(par_versions)-1):
        	f.write("data$speedup_"+par_versions[i]+",")
       	f.write("data$speedup_"+par_versions[len(par_versions)-1]+") \n")


	if filltype == "color":
		f.write("plot(data$speedup_"+par_versions[0]+", type=\"o\", lwd=3.0, col=\""+plot_colors[1] +"\", ylim=c(0,max_y+3), cex=3.5, axes=FALSE, ann=FALSE)\n")
		for i in range(1,len(par_versions)):
			f.write("lines(data$speedup_"+par_versions[i]+", type=\"o\", lwd=3.0, col=\""+plot_colors[i+1]+"\", cex=3.5) \n")
			#f.write("lines(data$speedup"+par_versions[i]+", type=\"o\", pch=22, lty=2, lwd=3.0, col=\""+plot_colors[i]+"\", cex=3.5) \n")
			#f.write("plot(data$speedup_"+v+", type=\"o\", lwd=3.0, col=\"blue\", ylim=c(0,max(data$speedup_"+v+")+1.8), cex=4.0, axes=FALSE, ann=FALSE)\n")
		#f.write("title(main=\"Seq vs GPU graph mining\", col.main=\"blue\", font.main=4, cex.main=3.5) \n")
		f.write("title(main=\"Seq vs GPU ("+dataset+" Dataset)\", col.main=\"blue\", font.main=4, cex.main=3.5) \n")
		f.write("title(xlab=\"Support\", line=3 , font.lab=4, col.lab=rgb(0,0.5,0), cex.lab=3.25 ) \n")
		f.write("title(ylab=\"Time in seconds\", line=3 ,font.lab=4, col.lab=rgb(0,0.5,0), cex.lab=3.25) \n")
		f.write("axis(4, col.lab=rgb(0,0.5,0), cex.axis=2.75) \n")  #put the axis on the right side
		f.write("mtext(\"Speedup\", side=4, line=3, font=4, col=rgb(0,0.5,0), cex=3.25 ) \n")

		f.write("legend(\"top\",c(\"Seq\"") 
		for v in par_versions:
			f.write(",\"GPU( "+par_labels[v]+" )\"")
		#f.write("), cex=3, fill=plot_colors, horiz=TRUE) \n")
		f.write("), cex=2.5, inset=c(0.1,0), fill=plot_colors, bty=\"n\") \n")
	elif filltype == "pattern":
		f.write("plot(data$speedup_"+par_versions[0]+", pch="+str(marker[0])+", type=\"b\", lty=1, lwd=3.0, ylim=c(0,max_y+3), cex=3.5, axes=FALSE, ann=FALSE)\n")
		for i in range(1,len(par_versions)):
			f.write("lines(data$speedup_"+par_versions[i]+", pch="+str(marker[i])+", type=\"b\", lty=1, lwd=3.0, cex=3.5) \n")
		
		f.write("title(main=\"Seq vs GPU ("+dataset+" Dataset)\", font.main=4, cex.main=3.5) \n")
		f.write("title(xlab=\"Support\", line=3 , font.lab=4, cex.lab=3.25 ) \n")
		f.write("title(ylab=\"Time in seconds\", line=3 ,font.lab=4, cex.lab=3.25) \n")
		f.write("axis(4, cex.axis=2.75) \n")  #put the axis on the right side
		f.write("mtext(\"Speedup\", side=4, line=3, font=4, cex=3.25 ) \n")

		
		for v in par_versions:
			f.write("plot_density <- append(plot_density, NA) \n")
			f.write("plot_angle <- append(plot_angle, NA) \n")

		f.write("plot_markers <- c(NA")
		for i in range(len(par_versions)):
			f.write(", NA")
		for i in range(len(par_versions)):
			f.write(", "+str(marker[i]))
		f.write(") \n")
		

		f.write("legend(\"top\",c(\"seq\"") 
		for v in par_versions:
			f.write(",\"GPU( "+par_labels[v]+" )\"")
		for v in par_versions:
			f.write(",\"GPU( "+par_labels[v]+" )\"")
		f.write("), cex=2.5, inset=c(0.1,0), density=plot_density, angle=plot_angle, pch=plot_markers, bty=\"n\") \n")
	else:
		print "wrong filltype"
		sys.exit()

	f.write("box() \n")
	# Turn off device driver (to flush output to eps)
	f.write("dev.off() \n")

	f.close()
	# change the permission of the R script to executable
	os.chmod(outfile+".r",0755)
	# run the R script and produce the eps file 
	os.system("./"+outfile+".r")


def do_bar_plot_par_versions(supports,par_times, outfile):
	
	cuda_versions = dict()
	cuda_versions["gspan_cuda_no_sort"] = "single-ext"
	cuda_versions["gspan_cuda_no_sort_block"] = "single-seg"
	cuda_versions["gspan_cuda_mult_block"] = "multiple-seg"	

	support = []
	for k in supports.keys():
		support = supports[k]	# get any of the support lists
		break

	support = sorted(support)

	f = open(infile+"_par_versions.dat","w")
	f.write("support")
	for ver in cuda_versions.keys():
		f.write("\t"+ver)
	f.write("\n")

	versions_list = ""
	comma_flag = False
	for ver in cuda_versions.keys():
		if comma_flag == True:
			versions_list += ","
		versions_list += "\""+ cuda_versions[ver] +"\""
		comma_flag = True	

	for i in xrange(0,len(support),step_size):
		s = support[i]
		if support_max < 0 or s <= support_max:
			if labels_as_K == "YF":
				f.write(str(s*1.0/1000)+"K")
			elif labels_as_K == "YI":
				f.write(str(s/1000)+"K")
			else:
				f.write(str(s))

			for ver in cuda_versions.keys():
				f.write("\t"+str(par_times[ver][str(s)]))
				
			f.write("\n")
	f.close()
	
	max_y = 0
	for p in cuda_versions.keys():
		y = max([v for (k,v) in par_times[p].items()] )
		if y > max_y:
			max_y  = y
	#print max_y
	if max_y < 10:
		ytick = math.ceil(max_y/5)
	else:
		ytick = math.ceil(max_y/50)*10

	# Write the R script
	f = open(outfile+".r","w")

	f.write("#!/usr/bin/Rscript\n")
	f.write("data <- read.table(\""+infile+"_par_versions.dat\", header=T, sep=\"\\t\") \n")
	#f.write("max_y <- max(data$seq,data$par) \n")
	f.write("max_y <- "+str(max_y)+"\n")
	f.write("plot_colors <- c(\"red\",\"green\",\"blue\")\n")
	f.write("postscript(\""+outfile+".eps\", bg=\"white\", paper=\"letter\") \n")
	f.write("par(mar=c(4.2,5.2,4,4.2)+.1)\n") 
	f.write("barplot(t(as.matrix(data[,2:4])), ylim=c(0,max_y+2.0), names.arg=data$support, cex.names=2.5, cex.axis=2.75, col=plot_colors, beside=TRUE) \n")
	f.write("par(new=TRUE) \n")
	f.write("title(main=\"GPU times ("+dataset+" Dataset)\", col.main=\"blue\", font.main=4, cex.main=3.5) \n")
	f.write("title(xlab=\"Support\", line=3 , font.lab=4, col.lab=rgb(0,0.5,0), cex.lab=3.25 ) \n")
	f.write("title(ylab=\"Time in seconds\", line=3, font.lab=4, col.lab=rgb(0,0.5,0), cex.lab=3.25) \n")
	#f.write("legend(\"topright\", substring(rownames(t(data))[2:4],12), cex=2, fill=plot_colors) \n")
	f.write("legend(\"topright\",c("+ versions_list +"), cex=3, fill=plot_colors, bty=\"n\") \n")
	f.write("box() \n")
	# Turn off device driver (to flush output to eps)
	f.write("dev.off() \n")

	f.close()
	# change the permission of the R script to executable
	os.chmod(outfile+".r",0755)
	# run the R script and produce the eps file 
	os.system("./"+outfile+".r")

if __name__== "__main__":
	
	#filltype = "color"
	filltype = "pattern"

	# seq vs one par version
	for ver in support_all:	
		do_bar_plot_seq_vs_par(support_all[ver],seq_time_all[ver],par_time_all[ver],outfile+"_seq_vs_"+ver,filltype, ver)
	#seq vs multiple par version
	#par_versions = ["gspan_cuda", "gspan_cuda_lists", "gspan_cuda_no_sort_block"]
	par_versions = ["gspan_cuda_lists", "gspan_cuda_no_sort_block"]
	par_labels = { "gspan_cuda_no_sort_block":"single-seg" , "gspan_cuda_lists":"tid-list", "gspan_cuda":"dfs-sort"}	
	do_bar_plot_seq_vs_par2(support_all, seq_time_all["gspan_cuda_no_sort_block"], par_time_all, par_versions, par_labels, outfile+"_seq_vs_par",filltype)

	#par versions
	do_bar_plot_par_versions(support_all,par_time_all,outfile+"_par_versions")	

