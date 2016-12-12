//Added by Karel Adamek

#ifndef __BLN__
#define __BLN__

extern void BLN_init(void);
extern int BLN(float *d_input, float *d_MSD, float *d_stats, int CellDim_x, int CellDim_y, int nDMs, int nTimesamples, int nIterations, int offset, float multiplier);

#endif
