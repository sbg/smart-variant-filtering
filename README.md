# Overview  

Smart Variant Filtering (SVF) is and open-source Python framework for genomic variants filtering using machine learning.
Variant filtering process consists of selecting highly confident variants and removing the ones that are falsely called. Secondary genomic DNA analysis is mainly oriented toward alignment and variant calling assuming the accuracy of these two would provide major influence to the overall quality. Variant filtering step was mostly left out from deeper testing even though it can boost precision of variant calls significantly.

Smart Variant Filtering uses machine learning algorithms trained on features from the existing Genome In A Bottle (GIAB) variant-called samples (HG001-HG005) to perform variant filtering (classification). 

SVF is available on github and also as a Public tool on Seven Bridges Public App gallery. It is completely open sourced and free to use by any party or conditions. The comparison results obtained during deep, 3-stage testing has proven that it outperforms the solutions currently used within the most of the secondary DNA analysis. Smart Variant Filtering increases the precision of called SNVs (removes false positives) for up to 0.2% while keeping the overall f-score higher by 0.12-0.27% than in the existing solutions. Indel precision is increased up to 7.8%, while f-score increasement is in range 0.1-3.2%.


## Getting Started

Before starting SVF make sure that the following modules are installed: 

```
pip install pandas
pip install sklearn
```

## Running the example
After cloning the repository several examples of SVF are available for training a model:

```
python svf_train.py --table_indel data/wes/6_features/HG001_NIST7035_dbsnp_indels.table --table_snv data/wes/6_features/HG001_NIST7035_dbsnp_SNVs.table --alg_param_indel MLP,250,logistic,sgd --alg_param_snv MLP,500,tanh,adam --vcf data/wes/6_features/HG005_oslo_exome_chr20.vcf
    
python svf_train.py --table_indel data/wes/7_features/HG002_oslo_exome_dbsnp_indels.table --table_snv data/wes/7_features/HG002_oslo_exome_dbsnp_SNVs.table --alg_param_indel MLP,10,logistic,sgd --alg_param_snv MLP,10,logistic,sgd --vcf data/wes/7_features/HG001_NIST7035_raw.dbsnp.vcf --num_features 7

```
and for performing a filtering:

```
python svf_apply.py --snv_model data/wes/6_features/HG001_NIST7035_dbsnp_SNVs.snv.sav --indel_model data/wes/6_features/HG001_NIST7035_dbsnp_indels.indel.sav --vcf data/wes/6_features/HG005_oslo_exome_chr20.vcf --num_features 6
    
python svf_apply.py --snv_model data/wes/6_features/HG001_NIST7035_dbsnp_SNVs.snv.sav --indel_model data/wes/6_features/HG001_NIST7035_dbsnp_indels.indel.sav --vcf data/wes/6_features/HG005_oslo_exome_chr20.vcf --num_features 7 --discard_existing_filters

```
