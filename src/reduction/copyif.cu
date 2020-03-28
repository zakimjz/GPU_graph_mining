
#define NUM_THREADS SCAN_NUM_THREADS
#define VALUES_PER_THREAD SCAN_VALUES_PER_THREAD
#define BLOCKS_PER_SM SCAN_BLOCKS_PER_SM


#define NUM_WARPS (NUM_THREADS / WARP_SIZE)
#define LOG_NUM_WARPS LOG_BASE_2(NUM_WARPS)
#define VALUES_PER_WARP (WARP_SIZE * VALUES_PER_THREAD)
#define NUM_VALUES (NUM_THREADS * VALUES_PER_THREAD)


////////////////////////////////////////////////////////////////////////////////
// Multiscan utility function. Used in the first and third passes of the
// global scan function. Returns the inclusive scan of the arguments in .x and
// the sum of all arguments in .y.

// Each warp is passed a pointer to its own contiguous area of shared memory.
// There must be at least 48 slots of memory. They should also be aligned so
// that the difference between the start of consecutive warps differ by an 
// interval that is relatively prime to 32 (any odd number will do).



////////////////////////////////////////////////////////////////////////////////
// GlobalScanUpsweep adds up all the values in elements_global within the 
// range given by blockCount and writes to blockTotals_global[blockIdx.x].

template<class predicate>
__launch_bounds__(NUM_THREADS, BLOCKS_PER_SM) __global__ 
void CopyIfUpsweep(uint* blockTotals_global, const int2* range_global, predicate pred)
{
	uint block = blockIdx.x;
	uint tid = threadIdx.x;
	int2 range = range_global[block];

	// Loop through all elements in the interval, adding up values.
	// There is no need to synchronize until we perform the multiscan.
	uint sum = 0;
	for(uint index = range.x + tid; index < range.y; index += 2 * NUM_THREADS) {
      uint val = 0;
      if((index + NUM_THREADS) < range.y) val = pred(index + NUM_THREADS);
      sum += pred(index) + val;
    }

	// A full multiscan is unnecessary here - we really only need the total.
	// But this is easy and won't slow us down since this kernel is already
	// bandwidth limited.
	uint total = Multiscan2<NUM_WARPS>(tid, sum).y;

	if(!tid)
		blockTotals_global[block] = total;
}


////////////////////////////////////////////////////////////////////////////////
// GlobalScanDownsweep runs an exclusive scan on the same interval of data as in
// pass 1, and adds blockScan_global[blockIdx.x] to each of them, writing back
// out in-place.


template<class predicate>
__launch_bounds__(NUM_THREADS, BLOCKS_PER_SM) __global__ 
void CopyIfDownsweep(uint* valuesOut_global, 
                     const uint* blockScan_global, const int2* range_global, int count, 
                     int inclusive, predicate pred)
{
	uint block = blockIdx.x;
	uint tid = threadIdx.x;
	uint warp = tid / WARP_SIZE;
	uint lane = (WARP_SIZE - 1) & tid;
	uint index = VALUES_PER_WARP * warp + lane;

	uint blockScan = blockScan_global[block];
	int2 range = range_global[block];

	const int Size = NUM_WARPS * VALUES_PER_THREAD * (WARP_SIZE + 1);
	__shared__ volatile uint shared[Size];

	// Use a stride of 33 slots per warp per value to allow conflict-free
	// transposes from strided to thread order.
	volatile uint* warpShared = shared + 
		warp * VALUES_PER_THREAD * (WARP_SIZE + 1);
	volatile uint* threadShared = warpShared + lane;

	// Transpose values into thread order.
	uint offset = VALUES_PER_THREAD * lane;
	offset += offset / WARP_SIZE;

	while(range.x < range.y) {

		#pragma unroll
		for(int i = 0; i < VALUES_PER_THREAD; ++i) {
			uint source = range.x + index + i * WARP_SIZE;
            if(source < count) {
              uint x = pred(source);
              threadShared[i * (WARP_SIZE + 1)] = x;
            } else {
              threadShared[i * (WARP_SIZE + 1)] = 0;
            }
		}

		// Transpose into thread order by reading from transposeValues.
		// Compute the exclusive or inclusive scan of the thread values and 
		// their sum.
		uint scan[VALUES_PER_THREAD];
		uint sum = 0;

		#pragma unroll
		for(int i = 0; i < VALUES_PER_THREAD; ++i) {
			uint x = warpShared[offset + i];
			scan[i] = sum;
			if(inclusive) scan[i] += x;
			sum += x;
		}


		// Multiscan for each thread's scan offset within the block. Subtract
		// sum to make it an exclusive scan.
		uint2 localScan = Multiscan2<NUM_WARPS>(tid, sum);
		uint scanOffset = localScan.x + blockScan - sum;

		// Add the scan offset to each exclusive scan and put the values back
		// into the shared memory they came out of.
		#pragma unroll
		for(int i = 0; i < VALUES_PER_THREAD; ++i) {
			uint x = scan[i] + scanOffset;
			warpShared[offset + i] = x;
		}

		// Store the scan back to global memory.
		#pragma unroll
		for(int i = 0; i < VALUES_PER_THREAD; ++i) {
			uint x = threadShared[i * (WARP_SIZE + 1)];
            uint target = range.x + index + i * WARP_SIZE;
            if(target < count) {
              valuesOut_global[target] = x;
            }
		}

		// Grab the last element of totals_shared, which was set in Multiscan.
		// This is the total for all the values encountered in this pass.
		blockScan += localScan.y;

		range.x += NUM_VALUES;
        __syncthreads(); // TODO: why !?
	}

    if(tid == 0) {
      valuesOut_global[count] = valuesOut_global[count - 1] + pred(count-1);
    } // if tid
}


#undef NUM_THREADS
#undef NUM_WARPS
#undef LOG_NUM_WARPS
#undef BLOCKS_PER_SM
#undef VALUES_PER_THREAD
#undef VALUES_PER_WARP
#undef NUM_VALUES


