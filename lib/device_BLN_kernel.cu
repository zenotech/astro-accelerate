// Added by Karel Adamek 

#ifndef BLN_KERNEL_H_
#define BLN_KERNEL_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include "AstroAccelerate/params.h"

__global__ void BLN_MSD_GPU_grid(float const* __restrict__ d_input, float *d_output, int x_steps, int y_steps, int nColumns, int msd) {
	extern __shared__ float Ms_Ss[];
	
	int warp_id, local_id, dim_y, pos;
	float x; // current element
	float M; // streaming mean
	float S; // streaming sum of squares (x_i-\bar{x})
	float j;
	float ftemp;
	
	local_id = threadIdx.x & (WARP - 1);
	warp_id = threadIdx.x>>5;
	dim_y = blockDim.x>>5;
	
	//----------------------------------------------
	//---- Calculating of streaming mean and sum of squares
	pos = (blockIdx.y*dim_y + warp_id)*y_steps*nColumns + blockIdx.x*WARP*x_steps + local_id;
	M=__ldg(&d_input[pos]);
	S=0;
	j=1.0f;
	for(int xf=1; xf<x_steps; xf++){
		pos = pos + WARP;
		x = __ldg(&d_input[pos]);
		j = j+1.0f;
		M = M + x;
		ftemp = (j*x - M);
		S = S + 1.0f/(j*(j-1.0f))*ftemp*ftemp;			
	}
	
	pos = pos + nColumns - (x_steps-1)*WARP;
	for(int yf=1; yf<y_steps; yf++){
		for(int xf=0; xf<x_steps; xf++){
			x = __ldg(&d_input[pos]);
			j = j+1.0f;
			M = M + x;
			ftemp = (j*x - M);
			S = S + 1.0f/(j*(j-1.0f))*ftemp*ftemp;
			pos = pos + WARP;
		}
		pos = pos + nColumns - x_steps*WARP;
	}
	
	Ms_Ss[threadIdx.x]=M;
	Ms_Ss[blockDim.x + threadIdx.x]=S;
	
	__syncthreads();
	
	// now all threads had saved their work, reduction follows
	
	// first we must load initial values
	//j=Neco;
	for(int i=(blockDim.x>>1); i>HALF_WARP; i=i>>1){
		if(threadIdx.x<i){
			j=j*2;
			ftemp = (M - Ms_Ss[i + threadIdx.x]);
			S = S + Ms_Ss[blockDim.x + i + threadIdx.x] + (1.0f/j)*ftemp*ftemp;
			M = M + Ms_Ss[i + threadIdx.x];
			
			Ms_Ss[threadIdx.x]=M;
			Ms_Ss[blockDim.x + threadIdx.x]=S;
		}
		// in the last iteration we do not need to save the results... or synchronize threads...
		__syncthreads();
	}
	
	// by now we should have only 32 partial results. shuffle reduction follows
	for(int q=HALF_WARP; q>0; q=q>>1){
		j=j*2;
		ftemp = (M - __shfl_down(M, q));
		S = S + __shfl_down(S, q) + (1.0f/j)*ftemp*ftemp;
		M = M + __shfl_down(M, q);
	}
	
	//----------------------------------------------
	//---- Writing data
	if(threadIdx.x==0){
		pos = blockIdx.y*gridDim.x + blockIdx.x;
		if(msd) {
			// produce mean and sd instead of T and S
			d_output[3*pos] = M/j;
			d_output[3*pos + 1] = sqrt(S/j);
		}
		else {
			d_output[3*pos] = M;
			d_output[3*pos + 1] = S;
		}
	}
}



__global__ void BLN_outlier_rejection(float const* __restrict__ d_input, float *d_output, float *d_stats, int size, float nElements, int nIterations, float multiplier) {
	__shared__ float Ms[WARP*WARP];
	__shared__ float Ss[WARP*WARP];
	__shared__ float js[WARP*WARP];
	__shared__ float s_signal_mean;
	__shared__ float s_signal_sd;
	
	
	int warp_id, pos;
	float M, Mt, S, j, jv;
	float ftemp;
	float signal_mean, signal_sd;
	
	__shared__ float s_ss_xx[WARP*WARP];
	__shared__ float s_ss_x[WARP*WARP];
	float ss_temp;
	float ss_xx;
	float ss_x;
	
	warp_id = threadIdx.x>>5;
	
	//-----------------------------------------------------------------
	//---- Calculation of the initial MSD
	pos=threadIdx.x;
	if(size>blockDim.x){
		M=__ldg(&d_input[3*pos]);
		S=__ldg(&d_input[3*pos+1]);
		j=nElements;
		//-------------------------- STATS --------------------------
		ss_temp = sqrt(__ldg(&d_input[3*pos+1])/((float) nElements));
		ss_x    = ss_temp;
		ss_xx   = (ss_temp*ss_temp);
		//-------------------------- STATS --------------------------
		pos = pos + blockDim.x;
		while (pos<size){
			jv=nElements;
			ftemp = ( jv/j*M - __ldg(&d_input[3*pos]) );
			S = S + __ldg(&d_input[3*pos+1]) + (j/(jv*(j+jv)))*ftemp*ftemp;
			M = M + __ldg(&d_input[3*pos]);
			j=j+jv;
			pos = pos + blockDim.x;
			//-------------------------- STATS --------------------------
			ss_temp = sqrt(__ldg(&d_input[3*pos+1])/((float) nElements));
			ss_x  += ss_temp;
			ss_xx += (ss_temp*ss_temp);
			//-------------------------- STATS --------------------------
		}
		
		__syncthreads();
		
		Ms[threadIdx.x]=M;
		Ss[threadIdx.x]=S;
		js[threadIdx.x]=j;
		
		//-------------------------- STATS --------------------------
		s_ss_x[threadIdx.x]  = ss_x;
		s_ss_xx[threadIdx.x] = ss_xx;
		//-------------------------- STATS --------------------------
		
		// now all threads had saved their work, reduction follows		
		// first we must load initial values
		for(int i=(blockDim.x>>1); i>HALF_WARP; i=i>>1){
			if(threadIdx.x<i){
				jv=js[i + threadIdx.x];
				ftemp = (jv/j*M - Ms[i + threadIdx.x]);
				S = S + Ss[i + threadIdx.x] + (j/(jv*(j+jv)))*ftemp*ftemp;
				M = M + Ms[i + threadIdx.x];
				j=j+jv;
				
				//-------------------------- STATS --------------------------
				ss_x  = ss_x + s_ss_x[threadIdx.x + i];
				ss_xx = ss_xx + s_ss_xx[threadIdx.x + i];
				//-------------------------- STATS --------------------------
				
				Ms[threadIdx.x]=M;
				Ss[threadIdx.x]=S;
				js[threadIdx.x]=j;
				//-------------------------- STATS --------------------------
				s_ss_x[threadIdx.x]  = ss_x;
				s_ss_xx[threadIdx.x] = ss_xx;
				//-------------------------- STATS --------------------------
			}
			__syncthreads();
		}
		
		// by now we should have only 32 partial results. shuffle reduction follows
		for(int q=HALF_WARP; q>0; q=q>>1){
			jv=__shfl_down(j, q);
			ftemp = (jv/j*M - __shfl_down(M, q));
			S = S + __shfl_down(S, q) + (j/(jv*(j+jv)))*ftemp*ftemp;
			M = M + __shfl_down(M, q);
			j=j+jv;
			
			//-------------------------- STATS --------------------------
			ss_x  = ss_x + __shfl_down(ss_x, q);
			ss_xx = ss_xx + __shfl_down(ss_xx, q);
			//-------------------------- STATS --------------------------
		}
		
	}
	else {
		if(threadIdx.x==0){
			//printf("Doing one thread thing\n");
			pos=0;
			M=__ldg(&d_input[3*pos]);
			S=__ldg(&d_input[3*pos+1]);
			j=nElements;
			//-------------------------- STATS --------------------------
			ss_temp = sqrt(__ldg(&d_input[3*pos+1])/((float) nElements));
			ss_x    = ss_temp;
			ss_xx   = (ss_temp*ss_temp);
			//-------------------------- STATS --------------------------
			for(pos=1; pos<size; pos++){
				jv=__ldg(&d_input[3*pos+2]);
				ftemp = ( jv/j*M - __ldg(&d_input[3*pos]) );
				S = S + __ldg(&d_input[3*pos+1]) + (j/(jv*(j+jv)))*ftemp*ftemp;
				M = M + __ldg(&d_input[3*pos]);
				j=j+jv;
				
				//-------------------------- STATS --------------------------
				ss_temp = sqrt(__ldg(&d_input[3*pos+1])/((float) nElements));
				ss_x  += ss_temp;
				ss_xx += (ss_temp*ss_temp);
				//-------------------------- STATS --------------------------
			}
		}
	}
	
	if(threadIdx.x==0){
		s_signal_mean = M/j;
		s_signal_sd   = sqrt(S/j);
	}
	
	__syncthreads();
	
	signal_mean = s_signal_mean;
	signal_sd   = s_signal_sd;
	//---- Calculation of the initial MSD
	//-----------------------------------------------------------
	
	//if(threadIdx.x==0) printf("Initial mean:%f; and standard deviation:%f;\n", signal_mean, signal_sd);
	
	
	//-------------------------- STATS --------------------------
	if(threadIdx.x==0){
		//printf("size:%f; alt_size:%f; ss_x:%f; ss_xx:%f; \n", (float) size, j/nElements, ss_x, ss_xx);
		d_stats[0]=signal_mean;
		d_stats[1]=signal_sd;
		d_stats[2]=ss_x/((float) (j/nElements)); //mean of sigma
		d_stats[3]=sqrt((ss_xx - (ss_x*ss_x)/((float) (j/nElements)))/((float) (j/nElements)));  //sigma of sigma
	}
	//-------------------------- STATS --------------------------
	
	
	
	//----------------------------------------------
	//---- Iterations with outlier rejection
	for(int f=0; f<nIterations; f++){
		pos=threadIdx.x;
		if(size>blockDim.x){
			M=0;
			S=0;
			j=0;
			
			//-------------------------- STATS --------------------------
			ss_x    = 0;
			ss_xx   = 0;
			//-------------------------- STATS --------------------------
			while (pos<size){
				Mt=__ldg(&d_input[3*pos]);
				if( (Mt/nElements > (signal_mean - multiplier*signal_sd)) && (Mt/nElements < (signal_mean + multiplier*signal_sd)) ){
					if(j==0){
						M = Mt;
						S = __ldg(&d_input[3*pos+1]);
						j = nElements;
						//-------------------------- STATS --------------------------
						ss_temp = sqrt(__ldg(&d_input[3*pos+1])/((float) nElements));
						ss_x    = ss_temp;
						ss_xx   = (ss_temp*ss_temp);
						//-------------------------- STATS --------------------------
					}
					else{
						jv=nElements;
						ftemp = ( jv/j*M - Mt);
						S = S + __ldg(&d_input[3*pos+1]) + (j/(jv*(j+jv)))*ftemp*ftemp;
						M = M + Mt;
						j=j+jv;
						//-------------------------- STATS --------------------------
						ss_temp = sqrt(__ldg(&d_input[3*pos+1])/((float) nElements));
						ss_x  += ss_temp;
						ss_xx += (ss_temp*ss_temp);
						//-------------------------- STATS --------------------------
					}
				}
				pos = pos + blockDim.x;
			}
			
			__syncthreads();
			
			Ms[threadIdx.x]=M;
			Ss[threadIdx.x]=S;
			js[threadIdx.x]=j;
			//-------------------------- STATS --------------------------
			s_ss_x[threadIdx.x]  = ss_x;
			s_ss_xx[threadIdx.x] = ss_xx;
			//-------------------------- STATS --------------------------
			// now all threads had saved their work, reduction follows		
			// first we must load initial values
			for(int i=(blockDim.x>>1); i>HALF_WARP; i=i>>1){
				if(threadIdx.x<i){
					jv=js[i + threadIdx.x];
					if(jv!=0){
						if(j==0){
							S = Ss[i + threadIdx.x];
							M = Ms[i + threadIdx.x];
							j = jv;
							//-------------------------- STATS --------------------------
							ss_x  = s_ss_x[threadIdx.x + i];
							ss_xx = s_ss_xx[threadIdx.x + i];
							//-------------------------- STATS --------------------------
						}
						else {
							ftemp = (jv/j*M - Ms[i + threadIdx.x]);
							S = S + Ss[i + threadIdx.x] + (j/(jv*(j+jv)))*ftemp*ftemp;
							M = M + Ms[i + threadIdx.x];
							j=j+jv;
							//-------------------------- STATS --------------------------
							ss_x  = ss_x + s_ss_x[threadIdx.x + i];
							ss_xx = ss_xx + s_ss_xx[threadIdx.x + i];
							//-------------------------- STATS --------------------------
						}
					}
					
					Ms[threadIdx.x]=M;
					Ss[threadIdx.x]=S;
					js[threadIdx.x]=j;
					//-------------------------- STATS --------------------------
					s_ss_x[threadIdx.x]  = ss_x;
					s_ss_xx[threadIdx.x] = ss_xx;
					//-------------------------- STATS --------------------------
				}
				__syncthreads();
			}
			
			// by now we should have only 32 partial results. shuffle reduction follows
			for(int q=HALF_WARP; q>0; q=q>>1){
				jv=__shfl_down(j, q);
				if(jv!=0){
					if(j==0) {
						S = __shfl_down(S, q);
						M = __shfl_down(M, q);
						j = jv;
						//-------------------------- STATS --------------------------
						ss_x  = __shfl_down(ss_x, q);
						ss_xx = __shfl_down(ss_xx, q);
						//-------------------------- STATS --------------------------
					}
					else {
						ftemp = (jv/j*M - __shfl_down(M, q));
						S = S + __shfl_down(S, q) + (j/(jv*(j+jv)))*ftemp*ftemp;
						M = M + __shfl_down(M, q);
						j=j+jv;	
						//-------------------------- STATS --------------------------
						ss_x  = ss_x + __shfl_down(ss_x, q);
						ss_xx = ss_xx + __shfl_down(ss_xx, q);
						//-------------------------- STATS --------------------------
					}

				}
			}
			
		}
		else {
			if(threadIdx.x==0){
				//printf("Doing one thread thing\n");
				M=0;
				S=0;
				j=0;
				for(pos=0; pos<size; pos++){
					Mt=__ldg(&d_input[3*pos]);
					if( (Mt/nElements > (signal_mean - multiplier*signal_sd)) && (Mt/nElements < (signal_mean + multiplier*signal_sd)) ){
						if(j==0){
							M=Mt;
							S=__ldg(&d_input[3*pos+1]);
							j=nElements;
							
							//-------------------------- STATS --------------------------
							ss_temp = sqrt(__ldg(&d_input[3*pos+1])/((float) nElements));
							ss_x    = ss_temp;
							ss_xx   = (ss_temp*ss_temp);
							//-------------------------- STATS --------------------------
						}
						else{
							jv=nElements;
							ftemp = ( jv/j*M - __ldg(&d_input[3*pos]) );
							S = S + __ldg(&d_input[3*pos+1]) + (j/(jv*(j+jv)))*ftemp*ftemp;
							M = M + __ldg(&d_input[3*pos]);
							j=j+jv;
							
							//-------------------------- STATS --------------------------
							ss_temp = sqrt(__ldg(&d_input[3*pos+1])/((float) nElements));
							ss_x  += ss_temp;
							ss_xx += (ss_temp*ss_temp);
							//-------------------------- STATS --------------------------
						}
					}
				}
			}
		}
		
		if(threadIdx.x==0){
			s_signal_mean = M/j;
			s_signal_sd   = sqrt(S/j);
		}
		
		__syncthreads();
		
		signal_mean = s_signal_mean;
		signal_sd   = s_signal_sd;
		
		
		
		//if(threadIdx.x==0) printf("Corrected mean:%f; and standard deviation:%f;\n", signal_mean, signal_sd);
		if(threadIdx.x==0){
			//printf("size:%f; alt_size:%f; ss_x:%f; ss_xx:%f; \n", (float) size, j/nElements, ss_x, ss_xx);
			d_stats[4*(f+1) + 0]=signal_mean;
			d_stats[4*(f+1) + 1]=signal_sd;
			d_stats[4*(f+1) + 2]=ss_x/((float) (j/nElements));
			d_stats[4*(f+1) + 3]=sqrt((ss_xx - (ss_x*ss_x)/((float) (j/nElements)))/((float) (j/nElements)));
		}
	}
	//---- Iterations with outlier rejection
	//----------------------------------------------
	
	
	
	//----------------------------------------------
	//---- Writing data
	if(threadIdx.x==0){
		d_output[0] = signal_mean;
		d_output[1] = signal_sd;
		d_output[2] = j;
	}
	//---- Writing data
	//----------------------------------------------
}

#endif
