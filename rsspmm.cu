#include <cstdlib>
#include <iostream>
#include <fstream>
#include <cuda.h>
#include <vector>
#include <algorithm>
#include <math.h>
#include <sys/time.h>

using namespace std;

vector<vector<int>> initializeMatrix();
vector<vector<vector<int>>> getHeavy(vector<vector<int>> origMatrix, vector<vector<int>> &matrixCopy);
vector<vector<int>> getLight(vector<vector<int>> lightMatix);

const int CPW = 4;
int thresh;
int start = 0;
int end = 0;

__global__ void light(int *csr_row_pointer, int *csr_column_idx, int *csr_column_val, int *input_value, int *dest_value, int n){
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
		val += input_value[(__shfl_sync(0xffffffff,index_buf,mod)*n)+lane_id] * __shfl_sync(0xffffffff,value_buf,mod);
	}

	// directly accumulate results in global memory
	atomicAdd(&dest_value[(row_offset*n)+slice_offset+lane_id], val);
}

__global__ void heavy(int *seg_value, int *seg_index, int *start_seg_position, int *seg_row_position, int *sm_input_value, int *input_value, int *dest_value, int n){
    int tid = threadIdx.x;
    int tb_idx = blockIdx.x;
    int tb_idy = blockIdx.y;

    int IN_TILE_COL_SIZE = 32;
    int IN_TILE_ROW_SIZE = 32;
    int IN_TILE_SLICE_SIZE = 32;
    int WARP_SIZE = 32;

	int row_offset = tb_idx * IN_TILE_ROW_SIZE;
	int slice_offset = tb_idy * IN_TILE_SLICE_SIZE;
	int warp_id = tid / WARP_SIZE;
	int lane_id = tid % WARP_SIZE;

	__syncthreads;

	int ii = 0;
    int val = 0;

	//for(int i = seg_start_num[tb_idx]; i <= (seg_start_num[tb_idx+1]-1); i += (blockDim.x / WARP_SIZE)){
    for(int i = 0; i < sizeof(start_seg_position)/sizeof(start_seg_position[0]); i++){
		ii = i+1;
		int start = start_seg_position[i];
		int end = start_seg_position[i+1];
        val = 0;
        int mod = 0;
        int index_buf = 0;
        int value_buf = 0;

		for(int j = start; j <= (end-1); j++){
			mod = (j-start ) % WARP_SIZE;
			index_buf = seg_index[j+lane_id];
			value_buf = seg_value[j+lane_id];
		}

		val += sm_input_value[__shfl_sync(0xffffffff,index_buf,mod )*n+lane_id] * __shfl_sync(0xffffffff,value_buf,mod);
	}
		
	int row_idx = seg_row_position[ii];
	// directly accumulate results in global memory
	atomicAdd (&dest_value[row_idx*n+slice_offset+lane_id], val);
}

int main() {
    const int SIZE = 7;
    for(int i = 0; i < SIZE; i++){
        thresh = pow(2,i);
        cout << "*============================= BEGINNING TEST FOR THRESHOLD = " << thresh << " =============================*" << endl;

        struct timeval tv1, tv2, tv3, tv4;
        struct timezone tz1, tz2, tz3, tz4;

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

        // allocate memory on host for light matrices
        h_Alv = (int*)malloc(csr[0].size()*sizeof(int));
        h_Alc = (int*)malloc(csr[1].size()*sizeof(int));
        h_Alr = (int*)malloc(csr[2].size()*sizeof(int));

        h_B = (int*)malloc(matrix.size()*matrix.size()*sizeof(int));
        h_C = (int*)malloc(matrix.size()*matrix.size()*sizeof(int));

        // initialize light A
        copy(csr[0].begin(), csr[0].end(), h_Alv);
        copy(csr[1].begin(), csr[1].end(), h_Alc);
        copy(csr[2].begin(), csr[2].end(), h_Alr);

        // initialize B and C
        for(int i = 0; i < matrix.size(); i++){
            for(int j = 0; j < matrix[0].size(); j++){
                h_B[(i*matrix.size())+j] = rand() % 2;
                h_C[(i*matrix.size())+j] = 0;
            }
        }

        // allocate light memory on GPU
        cudaMalloc(&d_Alv, csr[0].size()*sizeof(int));
        cudaMalloc(&d_Alc, csr[1].size()*sizeof(int));
        cudaMalloc(&d_Alr, csr[2].size()*sizeof(int));

        cudaMalloc(&d_B, matrix.size()*matrix.size()*sizeof(int));
        cudaMalloc(&d_C, matrix.size()*matrix.size()*sizeof(int));

        // copy host light data to device
        cudaMemcpy(d_Alv, h_Alv, csr[0].size()*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Alc, h_Alc, csr[1].size()*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Alr, h_Alr, csr[2].size()*sizeof(int), cudaMemcpyHostToDevice);

        cudaMemcpy(d_B, h_B, matrix.size()*matrix.size()*sizeof(int), cudaMemcpyHostToDevice);

        int threadsPerBlk = 16;
        dim3 blockSize = (threadsPerBlk, threadsPerBlk);
        dim3 gridSize = (matrix.size()/blockSize.x, matrix.size()/blockSize.y);

        // perform matrix multiplication on device
        gettimeofday(&tv1, &tz1);
        light <<< gridSize, blockSize >>> (d_Alr, d_Alc, d_Alv, d_B, d_C, matrix.size());
        gettimeofday(&tv2, &tz2);

        double lightTime = 1e6*(tv2.tv_sec - tv1.tv_sec) + (tv2.tv_usec - tv1.tv_usec);
        cout << "Light multiplication completed in: " << lightTime << " usec" << endl;

        // perform multiplication for each heavy block
        gettimeofday(&tv3, &tz3);
        for(int i = 0; i < dcsr.size(); i++){
            // host matrices A, B, and C for heavy multiplication
            int* h_Ahv;
            int* h_Ahc;
            int* h_Ahrp;
            int* h_Ahri;

            // device matrices A, B, and C for heavy multiplication
            int* d_Ahv;
            int* d_Ahc;
            int* d_Ahrp;
            int* d_Ahri;

            // allocate memory on host for heavy matrices
            h_Ahv = (int*)malloc(dcsr[i][0].size()*sizeof(int));
            h_Ahc = (int*)malloc(dcsr[i][1].size()*sizeof(int));
            h_Ahrp = (int*)malloc(dcsr[i][2].size()*sizeof(int));
            h_Ahri = (int*)malloc(dcsr[i][3].size()*sizeof(int));

            // initialize heavy A
            copy(dcsr[i][0].begin(), dcsr[i][0].end(), h_Ahv);
            copy(dcsr[i][1].begin(), dcsr[i][1].end(), h_Ahc);
            copy(dcsr[i][2].begin(), dcsr[i][2].end(), h_Ahrp);
            copy(dcsr[i][3].begin(), dcsr[i][3].end(), h_Ahri);

            // allocate heavy memory on GPU
            cudaMalloc(&d_Ahv, dcsr[i][0].size()*sizeof(int));
            cudaMalloc(&d_Ahc, dcsr[i][1].size()*sizeof(int));
            cudaMalloc(&d_Ahrp, dcsr[i][2].size()*sizeof(int));
            cudaMalloc(&d_Ahri, dcsr[i][3].size()*sizeof(int));

            // copy host heavy data to device
            cudaMemcpy(d_Ahv, h_Ahv, dcsr[i][0].size()*sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_Ahc, h_Ahc, dcsr[i][1].size()*sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_Ahrp, h_Ahrp, dcsr[i][2].size()*sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_Ahri, h_Ahri, dcsr[i][3].size()*sizeof(int), cudaMemcpyHostToDevice);

            heavy <<< gridSize, blockSize >>> (d_Ahv, d_Ahc, d_Ahrp, d_Ahri, d_B, d_B, d_C, matrix.size());
        }
        gettimeofday(&tv4, &tz4);

        double heavyTime = 1e6*(tv4.tv_sec - tv3.tv_sec) + (tv4.tv_usec - tv3.tv_usec);
        cout << "Heavy multiplication completed in: " << heavyTime << " usec" << endl;
        cout << "Total time: " << lightTime + heavyTime << " usec" << endl;

        // copy device data to host
        cudaMemcpy(h_C, d_C, matrix.size()*matrix.size()*sizeof(int), cudaMemcpyDeviceToHost);
    }

    return 0;
}

vector<vector<int>> initializeMatrix(){
    ifstream fin("bcsstk16.mtx");

    int ncols, nrows, nnz;
    int col, row, nz;

    fin >> ncols >> nrows >> nnz;
    vector<vector<int>> matrix(nrows, vector<int>(ncols,1));

    while(fin >> row >> col){
        matrix.at(row).at(col) = 1;
    }

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
            if(nnz > thresh){
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

