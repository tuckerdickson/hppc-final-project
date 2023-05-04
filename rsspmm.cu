#include <iostream>
#include <cuda.h>

using namespace std;

int IN_TILE_ROW_SIZE = 32;
int IN_TILE_SLICE_SIZE = 32;
int WARP_SIZE = 32;
int tid = threadIdx.x;
int tb_idx = blockIdx.x;
int start = 0;
int end = 0;
int row_offset;
int slice_offset;
int warp_id;
int lane_id;
int i;
[int][int] sm_input_value[IN_TILE_ROW_SIZE][WARP_SIZE];
[int][int] input_value[IN_TILE_ROW_SIZE][IN_TILE_SLICE_SIZE];
int seg_start_num[IN_TILE_ROW_SIZE];
int start_seg_position[IN_TILE_ROW_SIZE];
int index_buf;
int value_buf;
int val;
int mod;
int row_idx;
int seg_index[IN_TILE_ROW_SIZE];


//Heavy Row Segments
void heavy(){
	row_offset = tb_idx * IN_TILE_ROW_SIZE ;
	slice_offset = tb_idy * IN_TILE_SLICE_SIZE ;
	warp_id = tid / WARP_SIZE ;
	lane_id = tid % WARP_SIZE ;

	for i = warp_id to IN_TILE_ROW_SIZE step tb . size ()/ WARP_SIZE do
		sm_input_value [ i ][ lane_id ] = input_value [ row_offset +i ][ slice_offset + lane_id ];
	end

	__syncthreads ;

	for i = seg_start_num [ tb_idx ] to seg_start_num [ tb_idx +1] -1 step tb.size ()/ WARP_SIZE do
		val = 0;
		start = start_seg_position [i ];
		end = start_seg_position [i +1];

		for j = start to end -1 do
			mod = ( j - start )% WARP_SIZE

			if mod == 0 then
				index_buf = seg_index [ j + lane_id ];
				value_buf = seg_value [ j + lane_id ];
			end
			val += sm_input_value [ __shfl ( index_buf , mod )][ lane_id ]
			* __shfl ( value_buf , mod );
		end
		
		row_idx = seg_row_position [i ];
		// directly accumulate results in global memory
		atomicAdd (& dest_value [ row_idx ][ slice_offset + lane_id ] , val );
	end
}




//light rows 
void light(){

	row_offset = ( tb_idx * tb . size () + tid ) / WARP_SIZE ;
	slice_offset = tb_idy * IN_TILE_COL_SIZE ;
	lane_id = tid % WARP_SIZE ;
	start = csr_row_pointer [ row_offset ];
	end = csr_row_pointer [ row_offset +1];
	val = 0;
	for i = start to end -1 do
		mod = ( i - start )% WARP_SIZE
		if mod == 0 then
			index_buf = csr_column_idx [ i + lane_id ];
			value_buf = csr_column_val [ i + lane_id ];
		end
		val += input_value [ __shfl ( index_buf , mod )][ lane_id ]
		* __shfl ( value_buf , mod );
	end
	// directly accumulate results in global memory
	atomicAdd (& dest_value [ row_offset ][ slice_offset + lane_id ] , val );
}
