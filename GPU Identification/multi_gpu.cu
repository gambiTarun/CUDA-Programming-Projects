#include <stdio.h>

// Based on https://cuda-programming.blogspot.com/2013/01/how-to-query-device-properties-and.html
int main() {
  int nDevices;

  cudaGetDeviceCount(&nDevices);
  printf("Number of GPU Devices: %d\n", nDevices);

  // You will need to track the minimum or maximum for one or more device properties, so initialize them here
  int minMem = INT_MAX;
  int maxMem = INT_MIN;
  int currentChosenDeviceNumber = -1; // Will not choose a device by default 
  cudaDeviceProp prop;

  for (int i = 0; i < nDevices; i++) {
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Device Compute Major: %d Minor: %d\n", prop.major, prop.minor);
    printf("  Max Thread Dimensions: [%d][%d][%d]\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("  Max Threads Per Block: %d\n", prop.maxThreadsPerBlock);
    printf("  Number of Multiprocessors: %d\n", prop.multiProcessorCount);
    printf("  Device Clock Rate (KHz): %d\n", prop.clockRate);
    printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
    printf("  Registers Per Block: %d\n", prop.regsPerBlock);
    printf("  Registers Per Multiprocessor: %d\n", prop.regsPerMultiprocessor);
    printf("  Shared Memory Per Block: %zu\n", prop.sharedMemPerBlock);
    printf("  Shared Memory Per Multiprocessor: %zu\n", prop.sharedMemPerMultiprocessor);
    printf("  Total Constant Memory (bytes): %zu\n", prop.totalConstMem);
    printf("  Total Global Memory (bytes): %zu\n", prop.totalGlobalMem);
    printf("  Warp Size: %d\n", prop.warpSize);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    // You can set the current chosen device property based on tracked min/max values
    if(prop.totalGlobalMem<minMem){
      minMem = prop.totalGlobalMem;
    }

    if(prop.totalGlobalMem>maxMem){
      maxMem = prop.totalGlobalMem;
      currentChosenDeviceNumber = i;
    }

  }

  cudaGetDeviceProperties(&prop, currentChosenDeviceNumber);
  
  // Create logic to actually choose the device based on one or more device properties
  cudaChooseDevice(&currentChosenDeviceNumber, &prop);

  // Print out the chosen device as below
  printf("The chosen GPU device has an index of: %d\n",currentChosenDeviceNumber); 

  return 0;
}