import os
import itertools


def run_file(snv, indels, vcf, num_process=4):
    gold_file = "../svf_apply.py --snv_model " + snv + " --indel_model " + indels + " --vcf " + vcf
    new_file = "../svf_apply_parallel.py --snv_model " + snv + " --indel_model " + indels + " --vcf " + vcf + " --threads " + str(num_process)

    out_gold = os.system('python ' + gold_file)
    out_new = os.system('python ' + new_file)


def compare_output(gold, test, idx):
    with open(gold, 'r') as g, open(test, 'r') as tst:
        gold_lines = g.readlines()
        test_lines = tst.readlines()

        assert (len(gold_lines) == len(test_lines)), "Files are of different size!"

        for i in range(len(gold_lines)):
            assert (gold_lines[i] == test_lines[i]), "Line {} does not match! Test Failed!".format(i)

    os.remove(gold)
    os.remove(test)
    print("\t\tTEST  {}  SUCCESSFUL!".format(idx))


def get_output_file(file, par=False):
    filename = file.split('/').pop()
    basename = '.'.join(filename.split('.')[0:-1]) if filename.split('.')[-1] == 'vcf' else filename
    if par:
        out_name = basename + '.parallel.svf.vcf'
    else:
        out_name = basename + '.svf.vcf'
    return out_name


model_indels = ["../data/wes/6_features/HG001_NIST7035_dbsnp_indels.indel.sav",
                "../models/model_6_features_wgs_indel.sav"]
model_snvs = ["../data/wes/6_features/HG001_NIST7035_dbsnp_SNVs.snv.sav", "../models/model_6_features_wgs_snv.sav"]
vcfs = ["../data/wes/6_features/HG005_oslo_exome_chr20.vcf",
        "../data/wes/6_features/HG007.MOTHER.filtered.converted.medium.vcf"]
num_processes = [2, 4, 8, 16, 32]

for idx, example in enumerate(list(itertools.product(*[model_indels, model_snvs, num_processes, vcfs]))):
    print("TESTING {}\n\tSVN: {}\n\tINDELS: {}\n\tVCF: {}\n\tPROCESSES: {}".format(idx, example[1], example[0],
                                                                                   example[3], example[2]))
    run_file(example[1], example[0], example[3], example[2])
    compare_output(get_output_file(example[3]), get_output_file(example[3], True), idx)
