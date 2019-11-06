#OMP_NUM_THREADS=30 build/SCAMP DLS_DEBUG_LVL="2" --window=100 --max_tile_size=262144 --double_precision=true -output_pearson=true --aligned=true --input_a_file_name=test/SampleInput/randomlist2M.txt
#OMP_NUM_THREADS=30 ./SCAMP --window=100 --max_tile_size=262144 --double_precision=true -output_pearson=true --aligned=true --input_a_file_name=test/SampleInput/randomlist1M.txt

OMP_PLACES=threads OMP_PROC_BIND=true OMP_NUM_THREADS=3,76 CUDA_VISIBLE_DEVICES=0 time build/SCAMP DLS_DEBUG_LVL="2" --window=100 --max_tile_size=2000000 --double_precision=true -output_pearson=true --aligned=true --input_a_file_name=test/SampleInput/randomlist16K.txt
