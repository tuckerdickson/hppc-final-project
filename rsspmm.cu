#include <cstdlib>
#include <iostream>
#include <cuda.h>
#include <vector>
#include <algorithm>

using namespace std;

vector<vector<int>> initializeMatrix();
vector<vector<vector<int>>> getHeavy(vector<vector<int>> origMatrix, vector<vector<int>> &matrixCopy);
vector<vector<int>> getLight(vector<vector<int>> lightMatix);

const int CPW = 4;
const int THRESH = 2;
int start = 0;
int end = 0;

__global__ void light(int *csr_row_pointer, int *csr_column_idx, int *csr_column_val, int **input_value, int **dest_value){
    int tid = threadIdx.x;
    int tb_idx = blockIdx.x;
    int tb_idy = blockIdx.y;

    int IN_TILE_COL_SIZE = 32;
    int WARP_SIZE = 32;

	int row_offset = (tb_idx * blockDim.x + tid ) / WARP_SIZE;
	int slice_offset = tb_idy * IN_TILE_COL_SIZE;
	int lane_id = tid % WARP_SIZE ;
	int start = csr_row_pointer[row_offset];
	int end = csr_row_pointer[row_offset+1];
	int val = 0;
	for(int i = start; i <= (end-1); i++){
		int mod = (i-start) % WARP_SIZE;
        int index_buf = 0;
        int value_buf = 0;
		if(mod == 0){
			index_buf = csr_column_idx[i+lane_id];
			value_buf = csr_column_val[i+lane_id];
		}
		val += input_value[__shfl_sync(0xffffffff,index_buf,mod)][lane_id] * __shfl_sync(0xffffffff,value_buf,mod);
	}

	// directly accumulate results in global memory
	atomicAdd(&dest_value[row_offset][slice_offset+lane_id], val);
}

int main() {
    // initialize matrix
    vector<vector<int>> matrix = initializeMatrix();
    vector<vector<int>> matrixCopy = matrix;

    // get heavy row matrix (in DCSR format)
    vector<vector<vector<int>>> dcsr = getHeavy(matrix, matrixCopy);

    // get light row matrix (in CSR format)
    vector<vector<int>> csr = getLight(matrixCopy);

    ////////////////////////////////////////////////////////////////

    // host matrices A, B, and C for light multiplication
    int* h_Alv;
    int* h_Alc;
    int* h_Alr;
    int* h_B;
    int* h_C;

    // device matrices A, B, and C for light multiplication
    int* d_Alv;
    int* d_Alc;
    int* d_Alr;
    int* d_B;
    int* d_C;

    // allocate memory on host side
    h_Alv = (int*)malloc(csr[0].size()*sizeof(int));
    h_Alc = (int*)malloc(csr[1].size()*sizeof(int));
    h_Alr = (int*)malloc(csr[2].size()*sizeof(int));
    h_B = (int*)malloc(matrix.size()*matrix.size()*sizeof(int));
    h_C = (int*)malloc(matrix.size()*matrix.size()*sizeof(int));

    // initialize A,B and C

    return 0;
}

vector<vector<int>> initializeMatrix(){
    vector<vector<int>> matrix
    {
        {1,1,0,1,0,0,0,1},
        {0,0,1,0,0,1,0,0},
        {0,0,0,1,0,0,1,0},
        {1,1,1,1,1,1,0,1},
        {1,0,0,0,0,0,1,1},
        {0,1,1,1,1,0,0,0},
        {0,1,1,1,1,1,0,0},
        {0,0,1,1,1,1,1,0}
    };
    return matrix;
}

vector<vector<vector<int>>> getHeavy(vector<vector<int>> origMatrix, vector<vector<int>> &matrixCopy){
    // number of blocks in the matrix
    int nblocks = origMatrix[0].size()/CPW; 

    // create heavy row (DCSR) matrix
    vector<vector<vector<int>>> dcsr;

    for(int b = 0; b < nblocks; b++){
        
        vector<vector<int>> dcsrBlock;
        vector<int> dcsrVal, dcsrColidx, dcsrRowptr, dcsrRowidx;

        dcsrRowptr.push_back(0);
        for(int r = 0; r < origMatrix.size(); r++){

            vector<int> currRow;
            for(int c = b*CPW; c < (b*CPW)+CPW; c++){
                currRow.push_back(origMatrix[r][c]);
            }

            int nnz = count(currRow.begin(), currRow.end(), 1);
            if(nnz > THRESH){
                dcsrRowidx.push_back(r);
                dcsrRowptr.push_back(dcsrRowptr[dcsrRowptr.size()-1]+nnz);
                for(int c = b*CPW; c < (b*CPW)+CPW; c++){
                    if(origMatrix[r][c] != 0) {
                        dcsrVal.push_back(origMatrix[r][c]);
                        matrixCopy[r][c] = 0;
                        dcsrColidx.push_back(c-(b*CPW));
                    }
                }
            }
        }

        dcsrBlock.push_back(dcsrVal);
        dcsrBlock.push_back(dcsrColidx);
        dcsrBlock.push_back(dcsrRowptr);
        dcsrBlock.push_back(dcsrRowidx);
        dcsr.push_back(dcsrBlock);

    }
    return dcsr;
}

vector<vector<int>> getLight(vector<vector<int>> lightMatix){
    // create light row (CSR) matrix
    vector<vector<int>> csr;
    vector<int> csrVal, csrColidx, csrRowptr;
    csrRowptr.push_back(0);

    for(int r = 0; r < lightMatix.size(); r++){
        int nnz = count(lightMatix[r].begin(), lightMatix[r].end(), 1);
        csrRowptr.push_back(csrRowptr[csrRowptr.size()-1]+nnz);

        for(int c = 0; c < lightMatix[r].size(); c++){
            if(lightMatix[r][c] != 0){
                csrVal.push_back(lightMatix[r][c]);
                csrColidx.push_back(c);  
            }
        }
    }

    csr.push_back(csrVal);
    csr.push_back(csrColidx);
    csr.push_back(csrRowptr);

    return csr;
}

// __global__ void heavy(){
// 	int row_offset = tb_idx * IN_TILE_ROW_SIZE;
// 	int slice_offset = tb_idy * IN_TILE_SLICE_SIZE;
// 	int warp_id = tid / WARP_SIZE;
// 	int lane_id = tid % WARP_SIZE;

// 	for(int i = warp_id; i <=  IN_TILE_ROW_SIZE; i += (tb.size() / WARP_SIZE)){
// 		sm_input_value[i][lane_id] = input_value[row_offset+i][slice_offset+lane_id];
// 	}

// 	__syncthreads ;

// 	for(int i = seg_start_num [ tb_idx ]; i <= (seg_start_num [ tb_idx +1] -1); i+= (tb.size ()/ WARP_SIZE)){
// 		int val = 0;
// 		int start = start_seg_position [i ];
// 		int end = start_seg_position [i +1];

// 		for(int j = start; j <= (end -1); j++){
		
// 			mod = ( j - start )% WARP_SIZE
// 			index_buf = seg_index [ j + lane._id ];
// 			value_buf = seg_value [ j + lane_id ];
// 		}
// 			val += sm_input_value [ __shfl ( index_buf , mod )][ lane_id ]
// 			* __shfl ( value_buf , mod );
// 	}
		
// 	int row_idx = seg_row_position [i];
// 	// directly accumulate results in global memory
// 	atomicAdd (& dest_value [ row_idx ][ slice_offset + lane_id ] , val );
// }
