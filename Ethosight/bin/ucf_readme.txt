unzipper.sh
   will unzip all files in directory

extract_true_positives.sh
   will extract true positives based on temporal annotation file

extract_true_negatives.sh
   will extract true negatives based on "normal" videos

extract_true_negatives_hard.sh
   will extract true negatives based only on the temporal annotation file and does not include normal files

geneerate_dataset.sh
   will create a dataset based on serveral parameters including frame_set_count, true_positive_count, and true_negative_count 

   frame_set_count: the number of consecutive frames per sample (consecutive in the dataset e.g. _10, _20, _30

   true_positive_count, true_negative_count: the number of frame sets per video

 
