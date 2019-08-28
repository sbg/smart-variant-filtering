# Overview  

Smart Variant Filtering (SVF) is and open-source Python framework for genomic variants filtering using machine learning.
Variant filtering process consists of selecting highly confident variants and removing the ones that are falsely called. Secondary genomic DNA analysis is mainly oriented toward alignment and variant calling assuming the accuracy of these two would provide major influence to the overall quality. Variant filtering step was mostly left out from deeper testing even though it can boost precision of variant calls significantly.

Smart Variant Filtering uses machine learning algorithms trained on features from the existing Genome In A Bottle (GIAB) variant-called samples (HG001-HG005) to perform variant filtering (classification). 

SVF is available on github and also as a Public tool on Seven Bridges Public App gallery. It is completely open sourced and free to use by any party or conditions. The comparison results obtained during deep, 3-stage testing has proven that it outperforms the solutions currently used within the most of the secondary DNA analysis. Smart Variant Filtering increases the precision of called SNVs (removes false positives) for up to 0.2% while keeping the overall f-score higher by 0.12-0.27% than in the existing solutions. Indel precision is increased up to 7.8%, while f-score increasement is in range 0.1-3.2%.

Detailed information about Smart Variant Filtering available in [white paper](https://www.sevenbridges.com/smart-variant-filtering).

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
    
python svf_train.py --table_indel data/wes/7_features/HG002_oslo_exome_dbsnp_indels.table --table_snv data/wes/7_features/HG002_oslo_exome_dbsnp_SNVs.table --alg_param_indel MLP,10,logistic,sgd --alg_param_snv MLP,10,logistic,sgd --vcf data/wes/7_features/HG001_NIST7035_raw.dbsnp.vcf --features_snv QD,MQ,FS,MQRankSum,ReadPosRankSum,SOR,dbSNPBuildID --features_indel QD,MQ,FS,MQRankSum,ReadPosRankSum,SOR,dbSNPBuildID

```
and for performing a filtering:

```
python svf_apply.py --snv_model data/wes/6_features/HG001_NIST7035_dbsnp_SNVs.snv.sav --indel_model data/wes/6_features/HG001_NIST7035_dbsnp_indels.indel.sav --vcf data/wes/6_features/HG005_oslo_exome_chr20.vcf
    
```

Performing the filtering step can be done in parallel with at most 64 threads with:
```
python svf_apply_parallel.py --snv_model data/wes/6_features/HG001_NIST7035_dbsnp_SNVs.snv.sav --indel_model data/wes/6_features/HG001_NIST7035_dbsnp_indels.indel.sav --vcf data/wes/6_features/HG005_oslo_exome_chr20.vcf --threads 32
```
Read more about recommended number of threads and methods used for parallel execution in the "Parallelisation" section.

## Parallelisation

### Idea - source of parallelism
In the sequential implementation, the input file is sequentially processed line by line and each line is filtered independently. This is a great and easily exploited source of parallelism, as the processing function used in sequential implementation can be rewritten to accommodate for parallel chunk processing, without need for complex synchronization between worker threads and avoiding shared memory variables.

###Implementation
Parallel version of the tool processes the header of the input file in the main thread, but  then splits the lines of the input file into chunks. To exploit the underlying parallelism of the Variant filtering problem, these chunks of total work are then delegated to worker threads created by using an  implementation of the [**Thread pool**](https://en.wikipedia.org/wiki/Thread_pool) design pattern. The number of created worker threads is equal to the amount of threads specified to be used for parallelisation.

After a worker thread finishes processing the designated chunk, the result is stored in a dedicated slot and the worker thread is joined to the main thread. After all the chunks are processed, the main thread continues executing and collects the partial  results from the dedicated slots, performs some simple aftercomputations and returns the result which is equal to the result produced by the sequential implementation, but reached substantially faster, as can be seen in the [benchmarks folder](https://github.com/sbg/smart-variant-filtering/tree/master/data/benchmarks).

### How many threads should you use?
In parallelisation problems, more threads don't always result in faster execution times. As stated by the [Amdahl's law](https://en.wikipedia.org/wiki/Amdahl%27s_law), the execution time of a program is bounded by the time needed for the execution of the sequential component. In other words, there is a limit to the speedup achievable with thread number increase and the limit is caused by the component of the program which is not or can not be parallelised. In case of SVF, this component that limits speedup is the task of processing the input file header. 

Independent of the size of input file, the effects of saturation start when using 16 threads, as can be seen in the [benchmarks folder](https://github.com/sbg/smart-variant-filtering/tree/master/data/benchmarks). Therefore, using more than 16 threads should not be expected to yield further improvements on execution time of SVF.

Furthermore, the number of threads to be used for parallelisation depends on the machine executing the script. Each CPU has a set number of logical cores, that number being a limit to how many different threads can be executed at the same time. This forms a bottleneck on the possible speedup due to multithreaded execution, resulting in drastic speedup saturation when using a number of threads greater than the number of logical cores of the CPU. 

To summarize, the recommended value of the ```threads_num``` parameter is ```min(16, logical_core_count)```, where ```logical_core_count``` denotes the number of logical cores of the CPU on which script is executed.  

## Testing
Provided python script tests parallel version of filtering with the sequential one. It uses two different vcf files, with two models and number of threads in set ```[1, 2, 4, 8, 16, 32]```.
Testing is done by running the python code:
``` python testing_parallel.py ``` 

## Benchmarks
The benchmarks folder contains graphs of execution speed comparison to the number of threads running on three dataset sizes. The benchmark has been executed on Cancer Genomics Cloud instance c3.8xlarge (32vCPU).
Dataset used: The smallest input vcf file is provided in the ```data/wes/6_features/``` directory. The largest can be downloaded from [here](https://studentetfbgacrs-my.sharepoint.com/:u:/g/personal/mm183066m_student_etf_bg_ac_rs/EfhhU5O-KxJLvAqDlpGxhz8BhqDfUU5YxWpe6aXYwj6sgw?e=V74xUh), and the medium sized file can be created by running the script:
``` ./make_medium_test.sh ```

## Models
This repository contains pre-trained models for SNVs and indels, for whole genome sequencing (WGS) and whole exome sequencing (WES) samples. 
7 features are used for training the models: QD,MQ,FS,MQRankSum,ReadPosRankSum,SOR and dbSNPBuildId.
6 features were used for training the WGS model (QD,MQ,FS,MQRankSum,ReadPosRankSum,SOR) with the following samples and library preps: HG001-NA12878-50x, HG003-60x, HG004-60x, HG005-60x, HG001-CEPH-30x, HG001-Robot-30x, HG001-ERR17432-150x.
7 features were used for training the WES model (QD,MQ,FS,MQRankSum,ReadPosRankSum,SOR,dbSNPBuildId) with the following samples and library preps: HG001-NA12878Rep1-S1-L00R-140x, HG003-oslo-190x, HG004-oslo-190x, HG005-oslo-190x.

## Slides and video
Presentation of work done on parallelizing the tool (slides and video): [link](https://www.dropbox.com/sh/9flk1ly3ehcjpnk/AAD_PctO8N-yC4UfHsJuetiua?dl=0)
