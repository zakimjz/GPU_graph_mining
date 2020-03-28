#!/usr/bin/python
import sys
import os
import math

if len(sys.argv) < 2:
	print "usage : parse_output_gnu_plot.py <input-file> <output-file>"
	sys.exit()

infile = sys.argv[1]
outfile = sys.argv[2]

support = []
seq_time = dict()
par_time = dict()
freq_pat = dict()

f = open(infile,"r")
readFlag = False
for line in f:
	line.strip()
	w = line.split()
	if len(w) == 0:
		continue
	if readFlag:
		support.append(int(w[0]))
		seq_time[w[0]] = float(w[1])
		par_time[w[0]] = float(w[2])
		freq_pat[w[0]] = float(w[3])
	if w[0] == "support":
		readFlag = True
f.close()

support = sorted(support)

f = open(infile+".dat","w")
f.write("support\tseq\tpar\tspeedup\n")
for s in support:
	f.write(str(s/1000)+"K\t"+str(seq_time[str(s)])+"\t"+str(par_time[str(s)])+"\t"+str((par_time[str(s)]*1.0)/seq_time[str(s)])+"\n")

f.close()

max_y = max( max([v for (k,v) in seq_time.items()] ), max([v for (k,v) in par_time.items()]) )
#print max_y
if max_y < 10:
	ytick = math.ceil(max_y/5)
else:
	ytick = math.ceil(max_y/50)*10

f = open(outfile+".r","w")

f.write("#!/usr/bin/Rscript\n")
f.write("data <- read.table(\""+infile+".dat\", header=T, sep=\"\\t\") \n")
f.write("max_y <- max(data$seq,data$par) \n")
f.write("plot_colors <- c(\"blue\",\"red\")\n")
f.write("postscript(\""+outfile+".eps\", bg=\"white\", paper=\"letter\") \n")
f.write("par(mar=c(5.1,6.5,4.1,0.5))\n") ;
f.write("plot(data$seq, type=\"o\", lwd=3.0, col=plot_colors[1], ylim=c(0,max_y), cex=4.0,axes=FALSE, ann=FALSE) \n")
f.write("axis(1, at=1:nrow(data), lab=data$support, font=4, cex.axis=2.75) \n")
f.write("axis(2, las=2, at="+str(ytick)+"*0:"+str(ytick*5)+", font=4, cex.axis=2.5) \n")
f.write("box() \n")
# Graph trucks with red dashed line and square points
f.write("lines(data$par, type=\"o\", pch=22, lty=2, lwd=3.0, col=plot_colors[2], cex=3.25) \n")
# Create a title with a red, bold/italic font
f.write("title(main=\"Seq vs GPU graph mining\", col.main=\"blue\", font.main=4, cex.main=3.5) \n")

# Label the x and y axes with dark green text
f.write("title(xlab= \"Support\", line =3.5 , col.lab=rgb(0,0.5,0),  font.lab=4,  cex.lab=3.25 ) \n")
f.write("title(ylab= \"Time in seconds\", line=4.5 , col.lab=rgb(0,0.5,0), font.lab=4, cex.lab=3.25) \n")
# Create a legend at (1, max_y) that is slightly smaller
# (cex) and uses the same line colors and points used by
# the actual plots
f.write("legend(\"topright\",c(\"Seq\", \"GPU\"), cex=2.75, col=plot_colors, pch=21:23, lty=1:3) \n")
# Turn off device driver (to flush output to eps)
f.write("dev.off() \n")

f.close()
# change the permission of the R script to executable
os.chmod(outfile+".r",0755)
# run the R script and produce the eps file 
os.system("./"+outfile+".r")


