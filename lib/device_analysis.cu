#include <stdio.h>
#include <stdlib.h>
#include "AstroAccelerate/params.h"

#include "AstroAccelerate/device_MSD_plane.h"
#include "AstroAccelerate/device_MSD_limited.h"
#include "AstroAccelerate/device_SNR_limited.h"
#include "AstroAccelerate/device_SPS_inplace.h"
#include "AstroAccelerate/device_threshold.h"
#include "AstroAccelerate/device_single_FIR.h"
#include "timer.h"



//---------------------------------------------------------------------------------
//-------> Kahan MSD
void d_kahan_summation(float *signal, int nDMs, int nTimesamples, int offset, float *result, float *error)
{
	double sum;
	double sum_error;
	double a, b;

	sum = 0;
	sum_error = 0;
	for (int d = 0; d < nDMs; d++)
	{
		for (int s = 0; s < ( nTimesamples - offset ); s++)
		{
			a = signal[d * nTimesamples + s] - sum_error;
			b = sum + a;
			sum_error = ( b - sum );
			sum_error = sum_error - a;
			sum = b;
		}
	}
	*result = sum;
	*error = sum_error;
}

void d_kahan_sd(float *signal, int nDMs, int nTimesamples, int offset, double mean, float *result, float *error)
{
	double sum;
	double sum_error;
	double a, b, dtemp;

	sum = 0;
	sum_error = 0;
	for (int d = 0; d < nDMs; d++)
	{
		for (int s = 0; s < ( nTimesamples - offset ); s++)
		{
			dtemp = ( signal[d * nTimesamples + s] - sum_error - mean );
			a = dtemp * dtemp;
			b = sum + a;
			sum_error = ( b - sum );
			sum_error = sum_error - a;
			sum = b;
		}
	}
	*result = sum;
	*error = sum_error;
}

void MSD_Kahan(float *h_input, int nDMs, int nTimesamples, int offset, double *mean, double *sd)
{
	float error, signal_mean, signal_sd;
	int nElements = nDMs * ( nTimesamples - offset );

	d_kahan_summation(h_input, nDMs, nTimesamples, offset, &signal_mean, &error);
	signal_mean = signal_mean / nElements;

	d_kahan_sd(h_input, nDMs, nTimesamples, offset, signal_mean, &signal_sd, &error);
	signal_sd = sqrt(signal_sd / nElements);

	*mean = signal_mean;
	*sd = signal_sd;
}
//-------> Kahan MSD
//---------------------------------------------------------------------------------

void find_min_i_max(float *h_temp, int nDMs, int nTimesamples, int offset, float *max, float *min)
{
	float signal_max, signal_min;
	signal_max = h_temp[0];
	signal_min = h_temp[0];
	for (int d = 0; d < nDMs; d++)
	{
		for (int s = 0; s < ( nTimesamples - offset ); s++)
		{
			if (h_temp[d * nTimesamples + s] > signal_max)
				signal_max = h_temp[d * nTimesamples + s];
			if (h_temp[d * nTimesamples + s] < signal_min)
				signal_min = h_temp[d * nTimesamples + s];
		}
	}
	*max = signal_max;
	*min = signal_min;
}

void print_to_file(float *list, int size, float tsamp, float start_time, float dm_low, float dm_step, char *filename)
{
	FILE *file_out;

	if (( file_out = fopen(filename, "w") ) == NULL)
	{
		fprintf(stderr, "Error opening output file!\n");
		exit(0);
	}

	for (int f = 0; f < size; f++)
	{
		fprintf(file_out, "%f, %f, %f, %f\n", list[4 * f + 1] * tsamp + start_time, dm_low + list[4 * f] * dm_step, list[4 * f + 2], list[4 * f + 3]);
	}

	fclose(file_out);
}

void export_file_nDM_nTimesamples(float *data, int nDMs, int nTimesamples, char *filename)
{
	FILE *file_out;
	char str[200];

	sprintf(str, "%s_DM.dat", filename);
	if (( file_out = fopen(str, "w") ) == NULL)
	{
		fprintf(stderr, "Error opening output file!\n");
		exit(0);
	}

	printf("export nDMs\n");
	for (int s = 0; s < nTimesamples; s++)
	{
		for (int d = 0; d < nDMs; d++)
		{
			fprintf(file_out, "%f ", data[d * nTimesamples + s]);
		}
		fprintf(file_out, "\n");
	}

	fclose(file_out);

	sprintf(str, "%s_Time.dat", filename);
	if (( file_out = fopen(str, "w") ) == NULL)
	{
		fprintf(stderr, "Error opening output file!\n");
		exit(0);
	}

	printf("export nTimesamples\n");
	for (int d = 0; d < nDMs; d++)
	{
		for (int s = 0; s < nTimesamples; s++)
		{
			fprintf(file_out, "%f ", data[d * nTimesamples + s]);
		}
		fprintf(file_out, "\n");
	}

	fclose(file_out);

}



void analysis_GPU(int i, float tstart, int t_processed, int nsamp, int nchans, int maxshift, int max_ndms, int *ndms, int *outBin, float cutoff, float *output_buffer, float *dm_low, float *dm_high, float *dm_step, float tsamp){

	FILE *fp_out;
	char filename[200];

	int k, dm_count, remaining_time, bin_factor, counter;

	float start_time;

	unsigned long int j;
	unsigned long int vals;
	int nTimesamples = t_processed;
	int nDMs = ndms[i];

	float mean, stddev, stddev_orig;

	double total;

	// Calculate the total number of values
	vals = (unsigned long int) ( nDMs * nTimesamples );

	//chunk=(int)(vals/24);

	//start_time = ((input_increment/nchans)*tsamp);
	start_time = tstart;
	remaining_time = ( t_processed );

	sprintf(filename, "analysed-t_%.2f-dm_%.2f-%.2f.dat", start_time, dm_low[i], dm_high[i]);
	//if ((fp_out=fopen(filename, "w")) == NULL) {
	if (( fp_out = fopen(filename, "wb") ) == NULL)
	{
		fprintf(stderr, "Error opening output file!\n");
		exit(0);
	}

	double signal_mean, signal_sd, total_time, partial_time;
	float signal_mean_1, signal_sd_1, signal_mean_16, signal_sd_16, modifier;
	float max, min, threshold;
	int offset;
	float *h_temp = (float*) malloc(vals * sizeof(float));
	float *h_output_list;

	//---------------------------------------------------------------------------
	//----------> GPU part
	printf("\n GPU analysis part\n\n");
	printf("Dimensions nDMs:%d; nTimesamples:%d;\n", ndms[i], t_processed);
	GpuTimer timer;

	float *d_MSD;
	cudaMalloc((void**) &d_MSD, 3 * sizeof(float));
	float *d_SNR_MSD;
	cudaMalloc((void**) &d_SNR_MSD, 3 * sizeof(float));
	float *d_list;
	cudaMalloc((void**) &d_list, vals * sizeof(float));
	unsigned char *d_SNR_taps;
	cudaMalloc((void**) &d_SNR_taps, vals * sizeof(unsigned char));
	cudaMemset((void*) d_SNR_taps, 0, vals * sizeof(unsigned char));
	int *gmem_pos;
	cudaMalloc((void**) &gmem_pos, 1 * sizeof(int));
	cudaMemset((void*) gmem_pos, 0, sizeof(int));
	float h_MSD[3];
	int h_list_size;

	//sprintf(filename, "analysed-t_%.2f-dm_%.2f-%.2f", start_time, dm_low[i], dm_high[i]);
	//cudaMemcpy(h_temp, output_buffer, vals*sizeof(float), cudaMemcpyDeviceToHost);
	//export_file_nDM_nTimesamples(h_temp, nDMs, nTimesamples, filename);

	total_time = 0;
	// ----------------------------------------------------------------------------------------------------	
	// ---------> Mean and standard deviation is calculated once, higher taps are ignored
	timer.Start();
	MSD_limited(output_buffer, d_MSD, nDMs, nTimesamples, 128);
	timer.Stop();
	partial_time = timer.Elapsed();
	total_time += partial_time;
	printf("MSD limited took:%f ms\n", partial_time);

	timer.Start();
	cudaMemcpy(h_MSD, d_MSD, 3 * sizeof(float), cudaMemcpyDeviceToHost);
	signal_mean_1 = h_MSD[0];
	signal_sd_1 = h_MSD[1];
	//printf("Bin: %d, Mean: %f, Stddev: %f\n", 1, signal_mean_1, signal_sd_1);

	offset = PD_FIR(output_buffer, d_list, PD_MAXTAPS, nDMs, nTimesamples);
	MSD_limited(d_list, d_MSD, nDMs, nTimesamples, offset);
	cudaMemcpy(h_MSD, d_MSD, 3 * sizeof(float), cudaMemcpyDeviceToHost);
	signal_mean_16 = h_MSD[0];
	signal_sd_16 = h_MSD[1];
	//printf("Bin: %d, Mean: %f, Stddev: %f\n", PD_MAXTAPS, signal_mean_16, signal_sd_16);

	h_MSD[0] = signal_mean_1;
	h_MSD[1] = ( signal_sd_16 - signal_sd_1 ) / ( (float) ( PD_MAXTAPS - 1 ) );
	h_MSD[2] = signal_sd_1;
	modifier = h_MSD[1];
	cudaMemcpy(d_MSD, h_MSD, 3 * sizeof(float), cudaMemcpyHostToDevice);
	timer.Stop();
	partial_time = timer.Elapsed();
	total_time += partial_time;
	printf("Linear sd took:%f ms\n", partial_time);

	/*
	for(int f=2; f<=PD_MAXTAPS; f++){
	offset=PD_FIR(output_buffer, d_list, f, nDMs, nTimesamples);
	MSD_limited(d_list, d_SNR_MSD, nDMs, nTimesamples, offset);
	cudaMemcpy(h_MSD, d_SNR_MSD, 3*sizeof(float), cudaMemcpyDeviceToHost);
	printf("Bin: %d, Mean: %f; Stddev: %f; estimated sd: %f\n", f, h_MSD[0], h_MSD[1], signal_sd_1 + (f-1)*modifier);
	}
	*/

	timer.Start();
	PD_SEARCH_INPLACE(output_buffer, d_SNR_taps, d_MSD, PD_MAXTAPS, nDMs, nTimesamples);
	timer.Stop();
	partial_time = timer.Elapsed();
	total_time += partial_time;
	printf("PD_SEARCH took:%f ms\n", partial_time);

	timer.Start();
	THRESHOLD(output_buffer, d_SNR_taps, d_list, gmem_pos, 7.0, nDMs, nTimesamples, vals / 4);
	timer.Stop();
	partial_time = timer.Elapsed();
	total_time += partial_time;
	printf("THR_WARP took:%f ms\n", partial_time);
	// ---------> Mean and standard deviation is calculated once, higher taps are ignored
	// ----------------------------------------------------------------------------------------------------	
	printf("\n====> TOTAL TIME:%f\n\n", total_time);

	cudaMemcpy(&h_list_size, gmem_pos, sizeof(int), cudaMemcpyDeviceToHost);
	h_output_list = (float*) malloc(h_list_size * 4 * sizeof(float));
	cudaMemcpy(h_output_list, d_list, h_list_size * 4 * sizeof(float), cudaMemcpyDeviceToHost);

	float *b_list_out;
	b_list_out = (float*) malloc(h_list_size * 4 * sizeof(float));
	
	#pragma omp parallel for
	for (int count = 0; count < h_list_size; count++)
	{
		b_list_out[4 * count] = h_output_list[4 * count] * dm_step[i] + dm_low[i];
		b_list_out[4 * count + 1] = h_output_list[4 * count + 1] * tsamp + start_time;
		b_list_out[4 * count + 2] = h_output_list[4 * count + 2];
		b_list_out[4 * count + 3] = h_output_list[4 * count + 3];
	}

	fwrite(&b_list_out[0], h_list_size * sizeof(float), 4, fp_out);

	cudaFree(d_MSD);
	cudaFree(d_SNR_MSD);
	cudaFree(d_list);
	cudaFree(d_SNR_taps);
	cudaFree(gmem_pos);
	//----------> GPU part
	//---------------------------------------------------------------------------

	free(h_temp);
	free(b_list_out);
	free(h_output_list);

	fclose(fp_out);
}
