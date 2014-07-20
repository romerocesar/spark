import sys
import csv
from csv import reader

ROW_NUMBER_HEADER = "\"ROW_NUMBER\""
LABEL_HEADER = "\"LABEL\""

def create_rownum_to_result_map(inputfile):
    output_map = {}
    headers = inputfile.readline().strip().split(',')
    rownumidx = headers.index(ROW_NUMBER_HEADER)
    labelidx = headers.index(LABEL_HEADER)
     
    csv_reader = reader(inputfile)
    for line in csv_reader:
        rownum = line[rownumidx]
        label = line[labelidx]
        
        if rownum in output_map:
            raise Exception('row numbers repeated in ground truth file')
        output_map[rownum] = label
    return output_map

def check_results(expected_results, test_file):
    headers = test_file.readline().strip().split(',')
    rownumidx = headers.index(ROW_NUMBER_HEADER)
    labelidx = headers.index(LABEL_HEADER)

    tp_count = 0
    tn_count = 0
    fp_count = 0
    fn_count = 0
    csv_reader = reader(test_file)
    for line in csv_reader:
        rownum = line[rownumidx]
        label = line[labelidx]
        
        if rownum not in expected_results:
            raise Exception("output file contains a key not in input file or contains repeated keys!")

        if label == expected_results[rownum]:
            if label == 'POSITIVE':
                tp_count += 1
            elif label=='NEGATIVE':
                tn_count += 1
        else:
            if label == 'POSITIVE':
                fp_count += 1
            elif label == 'NEGATIVE':
                fn_count += 1
        del expected_results[rownum]

    row_count = tp_count + fp_count + tn_count + fn_count 
    if row_count < len(expected_results):
        raise Exception("Lesser number of rows in test file than in ground truth file!")
    return tp_count, fp_count, tn_count, fn_count

def get_eval_metrics(test_file, ground_truth_file):
    expected_results = create_rownum_to_result_map(ground_truth_file)
    
    tp_count, fp_count, tn_count, fn_count = check_results(expected_results, test_file)

    precision = float(tp_count)  / float(tp_count + fp_count)
    recall = float(tp_count)  / float(tp_count + fn_count)
    accuracy = float(tp_count + tn_count) / float(tp_count + tn_count + fp_count + fn_count)
    balanced_accuracy = float(0.5 * tp_count)/float(tp_count + fn_count)  + float(0.5*tn_count) / float(tn_count + fp_count);


    print 'precision : ', precision, '\n'
    print 'recall: ', recall, '\n'
    print 'accuracy: ', accuracy, '\n'
    print 'balanced_accuracy: ', balanced_accuracy, '\n'

if __name__ == '__main__':
    
    if len(sys.argv) != 3:
        raise Exception('usage: python eval.py <path/to/test_file> <path/to/ground_truth_file>')

    test_file = open(sys.argv[1])
    ground_truth_file = open(sys.argv[2])
    
    get_eval_metrics(test_file, ground_truth_file)

