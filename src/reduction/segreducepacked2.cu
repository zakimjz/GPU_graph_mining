#define NUM_THREADS PACKED_NUM_THREADS
#define BLOCKS_PER_SM PACKED_BLOCKS_PER_SM
#define VALUES_PER_THREAD PACKED_VALUES_PER_THREAD

#define NUM_WARPS (NUM_THREADS / WARP_SIZE)
#define LOG_NUM_WARPS LOG_BASE_2(NUM_WARPS)
#define VALUES_PER_WARP (WARP_SIZE * VALUES_PER_THREAD)
#define NUM_VALUES (NUM_THREADS * VALUES_PER_THREAD)

////////////////////////////////////////////////////////////////////////////////
// REDUCTION "DOWNSWEEP PASS".

/**
 * This is the "downsweep" pass of the reduction. There is little difference between reduction and
 * scan on the GPUs: the only difference is how to store the result into global memory.
 *
 * For the GSpan segmented reduce problem, we know that there are blocks of segments of the same
 * size. This can be used as advantage in the scan and in the storage of the results. Currently, we
 * use the advantage only for storage of the result as modifying the whole very optimized algorithm
 * is a pain.
 *
 *
 *
 *
 *
 *
 * @param[in]  packedIn_global    input values
 * @param[out] valuesOut_global   output values
 * @param[in]  start_global       ????
 * @param[in]  rangePairs_global  range of indexes. rangePairs_global[i] contains range in the packedIn_global that should be processed by ith block.
 * @param[in]  count              size of packedIn_global
 * @param[in]  inclusive          
 * @param[in]  segment_sizes      
 * @param[in]  segment_count
 * @param[in]  num_segments
 *
 *
 */

#define FLAG_AFTER_THIS_WARP    1
#define THIS_THREAD_HAS_FLAG    2
#define FLAG_AFTER_THIS_BLOCK   4
#define LAST_THREAD_LAST_VALUE  8

extern "C" __global__ __launch_bounds__(NUM_THREADS, BLOCKS_PER_SM)
  void SegReduceDownsweepPacked_faster(const uint* packedIn_global, uint* valuesOut_global, 
                                       const uint* start_global, const int2* rangePairs_global, int count,
                                       int inclusive, 
                                       uint *segment_sizes, uint *segment_count, uint num_segments)
{
	uint tid = threadIdx.x;
	uint lane = (WARP_SIZE - 1) & tid;
	uint warp = tid / WARP_SIZE;
	uint block = blockIdx.x;
	uint index = VALUES_PER_WARP * warp + lane;

	int2 range = rangePairs_global[block];

    //uint values_to_this_point = block ? rangePairs_global[block - 1].y : 0;
    uint values_to_this_point = tid * VALUES_PER_THREAD + range.x;

	const int Size = NUM_WARPS * VALUES_PER_THREAD * (WARP_SIZE + 1);
	__shared__ volatile uint shared[Size];
	__shared__ volatile uint blockOffset_shared;

	// Use a stride of 33 slots per warp per value to allow conflict-free
	// transposes from strided to thread order.
	volatile uint* warpShared = shared + 
		warp * VALUES_PER_THREAD * (WARP_SIZE + 1);
	volatile uint* threadShared = warpShared + lane;

	// Transpose values into thread order.
	uint offset = VALUES_PER_THREAD * lane;
	offset += offset / WARP_SIZE;


	if(!tid) blockOffset_shared = start_global[block];


	while(range.x < range.y) {
		// Load values into packed.
		uint x[VALUES_PER_THREAD];
		uint flags[VALUES_PER_THREAD];
        //uint flag_after_this_warp = 0;
        //uint this_thread_has_flag = 0;
        //uint flag_after_this_block = 0;
        //uint last_thread_last_value = 0;
        uint store_flags = 0;

		////////////////////////////////////////////////////////////////////////
		// Load and transpose values.

        #pragma unroll
        for(int i = 0; i < VALUES_PER_THREAD; ++i) {
          uint source = range.x + index + i * WARP_SIZE;
          uint packed = packedIn_global[source];
          threadShared[i * (WARP_SIZE + 1)] = packed;
        }


		// Transpose into thread order and separate values from head flags.
		#pragma unroll
		for(int i = 0; i < VALUES_PER_THREAD; ++i) {
			uint packed = warpShared[offset + i];
			x[i] = 0x7fffffff & packed;
			flags[i] = 0x80000000 & packed;
            //this_thread_has_flag |= (((flags[i] != 0) ? 1 : 0) << i);
            store_flags = (flags[i] != 0) ? (store_flags | THIS_THREAD_HAS_FLAG) : store_flags;
            //if(flags[i]) printf("tid: %d; store_flags: %d\n", tid, store_flags);
        }
        //this_thread_has_flag = (this_thread_has_flag & 0xfffffffe);

        {
          uint tmp = __ballot(flags[0]);
          //flag_after_this_warp = __ballot(flags[0]);
          //flag_after_this_warp &= (1 << (lane + 1));
          tmp &= (1 << (lane + 1));
          store_flags = (tmp != 0) ? (store_flags | FLAG_AFTER_THIS_WARP) : store_flags;
        }

        if(warp < (NUM_WARPS - 1) && lane == WARP_SIZE - 1) {
          uint packed = (shared + (warp + 1) * VALUES_PER_THREAD * (WARP_SIZE + 1))[0];
          //flag_after_this_block = ((0x80000000 & packed) != 0);
          store_flags = ((0x80000000 & packed) != 0) ? (store_flags | FLAG_AFTER_THIS_BLOCK) : store_flags;
        } else if(lane == WARP_SIZE - 1 && warp == NUM_WARPS - 1 && (range.x + VALUES_PER_WARP * warp + lane * VALUES_PER_THREAD) < count) {
          uint idx = range.x + VALUES_PER_WARP * NUM_WARPS; // does not work when I put this expression directly as index into packedIn_global
          uint packed = packedIn_global[idx];

          //flag_after_this_block = ((0x80000000 & packed) != 0);
          store_flags = ((0x80000000 & packed) != 0) ? (store_flags | FLAG_AFTER_THIS_BLOCK) : store_flags;
        }

        uint last_thread_start_idx = range.x + VALUES_PER_WARP * warp + lane * VALUES_PER_THREAD;
        if(last_thread_start_idx < count && count <= last_thread_start_idx + VALUES_PER_THREAD) {
          //last_thread_last_value = 1;
          store_flags = store_flags | LAST_THREAD_LAST_VALUE;

          //printf("XXX tid: %d is last in the array, warp: %d; flag_after_this_block: %d; range.x: %d; count: %d; index: %d\n",
          //tid, warp, flag_after_this_block, range.x, count, index);
        }


		////////////////////////////////////////////////////////////////////////
		// Run downsweep function on values and head flags.

		SegScanDownsweep<NUM_WARPS, VALUES_PER_THREAD>(tid, lane, warp, x, 
			flags, warpShared, threadShared, inclusive, &blockOffset_shared);

		////////////////////////////////////////////////////////////////////////
		// Transpose and store scanned values.
        /*
		#pragma unroll
		for(int i = 0; i < VALUES_PER_THREAD; ++i)
			warpShared[offset + i] = x[i];

			#pragma unroll
			for(int i = 0; i < VALUES_PER_THREAD; ++i) {
				uint target = range.x + index + i * WARP_SIZE;
				valuesOut_global[target] = threadShared[i * (WARP_SIZE + 1)];
			}
        */
        #pragma unroll
		for(int i = 0; i < VALUES_PER_THREAD; ++i) {
			warpShared[offset + i] = x[i];
        }

        if(range.x + tid*VALUES_PER_THREAD < count && (store_flags)) {
          const uint segment_index = 0;
          const uint block_values = 0;
          uint dest_idx = 0;

          // we do not have multiple segments with different size.
          //for(int i = 0; i < num_segments; ++i) {
          //if(values_to_this_point < (block_values + segment_sizes[i] * segment_count[i]) ) {
          //segment_index = i;
          //break;
          //} // if
          //block_values += segment_sizes[i] * segment_count[i];
          //dest_idx += segment_count[i];
          //} // for i

          dest_idx += (values_to_this_point - block_values)/segment_sizes[segment_index];


          uint values_to_process = count - values_to_this_point;
          values_to_process = values_to_process > (VALUES_PER_THREAD - 1) ? (VALUES_PER_THREAD - 1) : values_to_process;
          for(int i = 0 ; i < values_to_process; ++i) {
            if(flags[i+1]) {
              //printf("tid: %d; \tvalue: %d; \tdest_idx: %d; \tvalues_to_this_point: %d\n", tid, x[i], dest_idx, values_to_this_point);
              valuesOut_global[dest_idx] = x[i];
              dest_idx++;
            } // if
          } // for i


          if((store_flags & FLAG_AFTER_THIS_BLOCK) != 0) {
            uint idx = segment_sizes[segment_index] - ((values_to_this_point - block_values) % segment_sizes[segment_index]) - 1;
            uint last_idx = (range.x + NUM_VALUES) > range.y ? range.y : (range.x + NUM_VALUES);
            last_idx = (last_idx - values_to_this_point) / segment_sizes[segment_index];
            idx = (idx + last_idx * segment_sizes[segment_index])  % VALUES_PER_THREAD;
            //printf("tid: %d; \tvalue: %d; \tdest_idx: %d; \tvalues_to_this_point: %d; idx: %d; block_values: %d\n", tid, x[idx], dest_idx, values_to_this_point, idx, block_values);
            valuesOut_global[dest_idx] = x[idx];
            dest_idx++;
          }

          if((store_flags & LAST_THREAD_LAST_VALUE) != 0) {
            //uint idx = segment_sizes[segment_index] - ((count - values_to_this_point - block_values) % segment_sizes[segment_index]) - 1;
            //idx = idx % VALUES_PER_THREAD;
            uint idx = (count - values_to_this_point - 1) % VALUES_PER_THREAD;
            //printf("?tid: %d; \tvalue: %d; \tdest_idx: %d; \tvalues_to_this_point: %d; idx: %d; block_values: %d\n", tid, x[idx], dest_idx, values_to_this_point, idx, block_values);
            valuesOut_global[dest_idx] = x[idx];
            dest_idx++;            
          }

          if((store_flags & FLAG_AFTER_THIS_WARP) != 0) {
            //printf("tid: %d; \tvalue: %d; \tdest_idx: %d; \tvalues_to_this_point: %d\n", tid, x[VALUES_PER_THREAD - 1], dest_idx, values_to_this_point);
            valuesOut_global[dest_idx] = x[VALUES_PER_THREAD - 1];
            dest_idx++;
          }
        } // if

        values_to_this_point += NUM_VALUES;
        range.x += NUM_VALUES;
        __syncthreads(); // TODO: why !?
	}
}


#undef FLAG_AFTER_THIS_WARP
#undef THIS_THREAD_HAS_FLAG
#undef FLAG_AFTER_THIS_BLOCK
#undef LAST_THREAD_LAST_VALUE


#undef NUM_THREADS
#undef NUM_WARPS
#undef LOG_NUM_WARPS
#undef BLOCKS_PER_SM
#undef VALUES_PER_THREAD
#undef VALUES_PER_WARP
#undef NUM_VALUES

