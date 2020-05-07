# !/bin/bash
# Start the experiments
echo "Starting..."
#Run the original experiments
echo "RUN THE ORIGINAL EXPERIMENT "
n_epoch=15
echo "Number of epochs  = ${n_epoch}"
kernel_size=3
pool_2_size=4
echo "Kernel size  = ${kernel_size}"
echo "Pool 2 size  = ${pool_2_size}"
python -W ignore main.py -n_epoch ${n_epoch} -batch_size 100 -kernel_size ${kernel_size} -strides_conv1 1 -pool_1_size 2 -strides_conv2 3 -pool_2_size ${pool_2_size}

# Varying the parameters in the model to observe its effects on the result
echo "TESTING STRATEGY = VARYING ONLY ONE PARAMETER KERNEL SIZE OR POOL 2 SIZE AT A TIME "
n_epoch=5
echo "Set number of epochs  = ${n_epoch}"
# Keep all the paramenters the same, only varying kernel size
echo "I. ONLY VARYING KERNEL SIZE  "
pool_2_size=2
echo "Set Pool 2 size  = ${pool_2_size}"

for kernel_size in 2 3 4 5 6; do
  echo "RUN EXPERIMENT WITH KERNEL SIZE = ${kernel_size}  "
  python -W ignore main.py -n_epoch ${n_epoch} -batch_size 100 -kernel_size ${kernel_size} -strides_conv1 1 -pool_1_size 2 -strides_conv2 3 -pool_2_size ${pool_2_size}

done

# Keep all the paramenters the same, only varying pool 2 size
echo "II. ONLY VARYING POOL 2 SIZE "
n_epoch=5
kernel_size=3
echo "Set kernel size  = ${kernel_size}"

for pool_2_size in 3 2 1; do
  echo "RUN EXPERIMENT WITH POOL 2 SIZE = ${pool_2_size}  "
  python -W ignore main.py -n_epoch ${n_epoch} -batch_size 100 -kernel_size ${kernel_size} -strides_conv1 1 -pool_1_size 2 -strides_conv2 3 -pool_2_size ${pool_2_size}
done


echo "End !"
