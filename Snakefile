
plottargets = "figACF_WN figACF figError figModel figOrig figPACF_WN figPACF figSeason figTrend".split()
tabletargets = "aicc_df error_table trend_table vectorcorr_df".split()

## rule all: runs over all rules except webscrape
rule all:
    input:
        "output/bundleData.csv",
        "output/data_allcantons.csv",
        "output/data_ZH.csv",
        "output/data_BE.csv",
        "output/finaldata_ZH.csv",
        "output/finaldata_BE.csv",
        "finalText.pdf",
        plotsZH = expand("output/ZH/plot/{iFile}.pdf", iFile= plottargets),
        tablesZH = expand("output/ZH/table/{iFile}.tex", iFile= tabletargets),
        plotsBE = expand("output/BE/plot/{iFile}.pdf", iFile= plottargets),
        tablesBE = expand("output/BE/table/{iFile}.tex", iFile= tabletargets)

## rule webscrape: scrapes data from Google Trends. This rule is only executed if called specifically (no output). Takes a long time, maybe better to split dates in half.
rule webscrape:
    input:
        script = "input/webscrapeData.py"
    params:
        "rawdata/"
    script:
        "input/webscrapeData.py"

## rule bundle: bundles the raw data into one file 'bundleData'
bundleTargets = glob_wildcards("rawdata/webscrapeData/{name}.csv").name
rule bundle:
    input:
        data = expand("rawdata/webscrapeData/{iFile}.csv", iFile= bundleTargets),
        script = "input/bundleData.py"
    params:
        "rawdata/webscrapeData/"
    output:
        "output/bundleData.csv"
    script:
        "input/bundleData.py"

## rule clean: cleans the bundleData and creates separate files for ZH and BE
rule clean:
    input:
        data = "output/bundleData.csv",
        script = "input/cleanData.py"
    params:
        "output/"
    output:
        all_cantons = "output/data_allcantons.csv",
        cantonZH = "output/data_ZH.csv",
        cantonBE = "output/data_BE.csv"
    script:
        "input/cleanData.py"

## rule tsaZH: full time series analysis for canton ZH
rule tsaZH:
    input:
        data = "output/data_ZH.csv",
        script = "input/TSA_ZH.py"
    params:
        "output/"
    output:
        finaldataZH = "output/finaldata_ZH.csv",
        plotsZH = expand("output/ZH/plot/{iFile}.pdf", iFile= plottargets),
        tablesZH = expand("output/ZH/table/{iFile}.tex", iFile= tabletargets)
    script:
        "input/TSA_ZH.py"

## rule tsaBE: full time series analysis for canton BE
rule tsaBE:
    input:
        data = "output/data_BE.csv",
        script = "input/TSA_BE.py"
    params:
        "output/"
    output:
        finaldataZH = "output/finaldata_BE.csv",
        plotsBE = expand("output/BE/plot/{iFile}.pdf", iFile= plottargets),
        tablesBE = expand("output/BE/table/{iFile}.tex", iFile= tabletargets)
    script:
        "input/TSA_BE.py"

## rule pdftext: this rule compiles the final tex file
rule pdftext:
    input:
        finalZH = "output/finaldata_ZH.csv",
        finalBE = "output/finaldata_BE.csv",
        file = "input/finalText.tex",
        script = "input/finalText.py",
        plotsZH = expand("output/ZH/plot/{iFile}.pdf", iFile= plottargets),
        tablesZH = expand("output/ZH/table/{iFile}.tex", iFile= tabletargets),
        plotsBE = expand("output/BE/plot/{iFile}.pdf", iFile= plottargets),
        tablesBE = expand("output/BE/table/{iFile}.tex", iFile= tabletargets)
    params:
        indir = "input/",
        outdir = "output/",
        intext = "finalText"
    output:
        "finalText.pdf"
    script:
        "input/finalText.py"

## rule help: this rule helps the reader to understand the rules
rule help:
    input:
        "Snakefile"
    shell:
        "sed -n 's/^##//p' {input}"
