#Set snakemake
in_tex = snakemake.input[0]
in_dir = snakemake.params[0]
out_dir = snakemake.params[1]
in_txt = snakemake.params[2]
out_pdf = snakemake.output[0]


import os


#run doc (several times for bibtex)
os.chdir(in_dir)
os.system("pdflatex "+ in_txt)
os.system("bibtex "+ in_txt)
os.system("pdflatex "+ in_txt)
os.system("pdflatex "+ in_txt)
#unlink
os.unlink(in_txt+ ".aux")
os.unlink(in_txt+ ".bbl")
os.unlink(in_txt+ ".blg")
os.unlink(in_txt+ ".log")
os.unlink(in_txt+ ".out")

#move doc
os.chdir("..")
os.rename(in_dir+in_txt+".pdf", out_pdf)
