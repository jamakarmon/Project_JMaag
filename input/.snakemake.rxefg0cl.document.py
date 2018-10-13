
######## Snakemake header ########
import sys; sys.path.append("/anaconda3/lib/python3.6/site-packages"); import pickle; snakemake = pickle.loads(b'\x80\x03csnakemake.script\nSnakemake\nq\x00)\x81q\x01}q\x02(X\x05\x00\x00\x00inputq\x03csnakemake.io\nInputFiles\nq\x04)\x81q\x05X\x17\x00\x00\x00input/finalDocKopie.texq\x06a}q\x07X\x06\x00\x00\x00_namesq\x08}q\tsbX\x06\x00\x00\x00outputq\ncsnakemake.io\nOutputFiles\nq\x0b)\x81q\x0c}q\rh\x08}q\x0esbX\x06\x00\x00\x00paramsq\x0fcsnakemake.io\nParams\nq\x10)\x81q\x11}q\x12h\x08}q\x13sbX\t\x00\x00\x00wildcardsq\x14csnakemake.io\nWildcards\nq\x15)\x81q\x16}q\x17h\x08}q\x18sbX\x07\x00\x00\x00threadsq\x19K\x01X\t\x00\x00\x00resourcesq\x1acsnakemake.io\nResources\nq\x1b)\x81q\x1c(K\x01K\x01e}q\x1d(h\x08}q\x1e(X\x06\x00\x00\x00_coresq\x1fK\x00N\x86q X\x06\x00\x00\x00_nodesq!K\x01N\x86q"uh\x1fK\x01h!K\x01ubX\x03\x00\x00\x00logq#csnakemake.io\nLog\nq$)\x81q%}q&h\x08}q\'sbX\x06\x00\x00\x00configq(}q)X\x04\x00\x00\x00ruleq*X\x03\x00\x00\x00docq+ub.'); from snakemake.logging import logger; logger.printshellcmds = False
######## Original script #########
#Set snakemake
in_tex = snakemake.input[0]

import subprocess
import os


#subprocess.check_call(['pdflatex', in_tex])
os.system("cd input/")
os.system("pdflatex finalDocKopie.tex")
#os.unlink('document.aux')
#os.unlink('document.log')
