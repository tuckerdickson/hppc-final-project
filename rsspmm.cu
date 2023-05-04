//Heavy Row Segments

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
