./EthosightCLI.py summarize --label-affinity-scores shoplifting.affinities -o shoplifting_summary.txt
./EthosightCLI.py ask images/shoplifting.png --background-knowledge "the man is bill gates" --summary-file shoplifting_summary.txt --questions "what is his name?
what did he do?
which hair color does he have?
is he man or woman?
" --outfile shoplifting_answers.txt
