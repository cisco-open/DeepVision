echo 'EthosightCLI reason --prompt-type blank_slate --use-case "shoplifting, person in danger, medical incident in progress, violence of any type" -o shopliftingblankslate.labels' 
EthosightCLI reason --prompt-type blank_slate --use-case "shoplifting, person in danger, medical incident in progress, violence of any type" -o testoutput/shopliftingblankslate.labels 

echo 'EthosightCLI embed shopliftingblankslate.labels'
EthosightCLI embed testoutput/shopliftingblankslate.labels

echo 'EthosightCLI affinities ../images/shoplifting.png shopliftingblankslate.embeddings --output_filename testoutput/shopliftingblankslate.affinities'
EthosightCLI affinities ../images/shoplifting.png testoutput/shopliftingblankslate.embeddings --output_filename testoutput/shopliftingblankslate.affinities 

echo 'EthosightCLI reason --prompt-type iterative --label-affinity-scores shopliftingblankslate.affinities --use-case "retail loss prevention" -o shopliftingblankslate_iteration.labels'
EthosightCLI reason --prompt-type iterative --label-affinity-scores testoutput/shopliftingblankslate.affinities --use-case "retail loss prevention" -o testoutput/shopliftingblankslate_iteration.labels 

echo 'EthosightCLI embed shopliftingblankslate_iteration.labels'
EthosightCLI embed testoutput/shopliftingblankslate_iteration.labels

echo 'EthosightCLI affinities ../images/shoplifting.png shopliftingblankslate_iteration.embeddings --output_filename shopliftingblankslate_iteration1.affinities'
EthosightCLI affinities ../images/shoplifting.png testoutput/shopliftingblankslate_iteration.embeddings --output_filename testoutput/shopliftingblankslate_iteration1.affinities

echo 'EthosightCLI reason --prompt-type iterative --label-affinity-scores shopliftingblankslate_iteration1.affinities --use-case "retail loss prevention" -o shopliftingblankslate_iteration1.labels'
EthosightCLI reason --prompt-type iterative --label-affinity-scores testoutput/shopliftingblankslate_iteration1.affinities --use-case "retail loss prevention" -o testoutput/shopliftingblankslate_iteration1.labels

echo 'EthosightCLI embed shopliftingblankslate_iteration1.labels'
EthosightCLI embed testoutput/shopliftingblankslate_iteration1.labels

echo 'EthosightCLI affinities ../images/shoplifting.png testoutput/shopliftingblankslate_iteration1.embeddings --output_filename testoutput/shopliftingblankslate_iteration2.affinities'
EthosightCLI affinities ../images/shoplifting.png testoutput/shopliftingblankslate_iteration1.embeddings --output_filename testoutput/shopliftingblankslate_iteration2.affinities

