#include <iostream>
#include <cuda.h>

using namespace std;

int IN_TILE_ROW_SIZE = 32;
int IN_TILE_SLICE_SIZE = 32;
int WARP_SIZE = 32;
int tid = threadIdx.x;
int tb_idx = blockIdx.x;
[int][int] sm_input_value[IN_TILE_ROW_SIZE][WARP_SIZE];
[int][int] input_value[IN_TILE_ROW_SIZE][IN_TILE_SLICE_SIZE];
int seg_start_num[IN_TILE_ROW_SIZE];
int start_seg_position[IN_TILE_ROW_SIZE];
int index_buf, value_buf, mod;
int seg_index[IN_TILE_ROW_SIZE];


//Heavy Row Segments
void heavy(){
	int row_offset = tb_idx * IN_TILE_ROW_SIZE ;
	int slice_offset = tb_idy * IN_TILE_SLICE_SIZE ;
	int warp_id = tid / WARP_SIZE ;
	int lane_id = tid % WARP_SIZE ;

	for(int i = warp_id; i <=  IN_TILE_ROW_SIZE; i += (tb . size ()/ WARP_SIZE)){
		sm_input_value [ i ][ lane_id ] = input_value [ row_offset +i ][ slice_offset + lane_id ];
	}

	__syncthreads ;

	for(int i = seg_start_num [ tb_idx ]; i <= (seg_start_num [ tb_idx +1] -1); i+= (tb.size ()/ WARP_SIZE)){
		int val = 0;
		int start = start_seg_position [i ];
		int end = start_seg_position [i +1];

		for(int j = start; j <= (end -1); j++){
		
			mod = ( j - start )% WARP_SIZE
			index_buf = seg_index [ j + lane._id ];
			value_buf = seg_value [ j + lane_id ];
		}
			val += sm_input_value [ __shfl ( index_buf , mod )][ lane_id ]
			* __shfl ( value_buf , mod );
	}
		
	int row_idx = seg_row_position [i];
	// directly accumulate results in global memory
	atomicAdd (& dest_value [ row_idx ][ slice_offset + lane_id ] , val );
	
}




//light rows 
void light(){

	int row_offset = ( tb_idx * tb . size () + tid ) / WARP_SIZE ;
	int slice_offset = tb_idy * IN_TILE_COL_SIZE ;
	int lane_id = tid % WARP_SIZE ;
	int start = csr_row_pointer [ row_offset ];
	int end = csr_row_pointer [ row_offset +1];
	int val = 0;
	for(int i = start; i <= (end -1); i++){
		mod = ( i - start )% WARP_SIZE
		if(mod == 0){
			index_buf = csr_column_idx [ i + lane_id ];
			value_buf = csr_column_val [ i + lane_id ];
		}
		val += input_value [ __shfl ( index_buf , mod )][ lane_id ]
		* __shfl ( value_buf , mod );
	}
	// directly accumulate results in global memory
	atomicAdd (& dest_value [ row_offset ][ slice_offset + lane_id ] , val );
}
