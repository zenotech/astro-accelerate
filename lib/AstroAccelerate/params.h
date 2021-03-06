#define ACCMAX 350
#define ACCSTEP 11
#define UNROLLS 16
#define SNUMREG 10
#define SDIVINT 8
#define SDIVINDM 40
#define SFDIVINDM 40.0f
#define CARD 0
#define NOPSSHIFT 5
#define NOPSLOOP 3
#define NDATAPERLOOP 1
#define BINDIVINT 6
#define BINDIVINF 32
#define CT 256
#define CF 2
#define NOPS 4.0
#define STATST 128
#define STATSLOOP 8




//Added by Karel Adamek
#define WARP 32
#define HALF_WARP 16
#define MSD_ELEM_PER_THREAD 8
#define MSD_WARPS_PER_BLOCK 16
#define THR_ELEM_PER_THREAD 4
#define THR_WARPS_PER_BLOCK 4
#define PD_NTHREADS 512
#define PD_NWINDOWS 2
#define PD_MAXTAPS 16
#define PD_SMEM_SIZE 1280
#define PD_FIR_ACTIVE_WARPS 2
#define PD_FIR_NWINDOWS 2