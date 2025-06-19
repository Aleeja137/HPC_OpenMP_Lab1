// Compile with gcc -O2 -fopenmp -fopt-info-vec
// Compile with gcc -O2 -fopenmp -fopt-info-vec -march=native
// gcc -O2 -fno-tree-vectorize -fopenmp -fopt-info-vec  disables vectorization 

#include <stdio.h>
#include <omp.h>
/* Program for multiplication of D=A*B */

#define NRA 2048		/* number of rows in matrix A */
#define NCA 2048        	/* number of columns in matrix A */
#define NCB 2048		/* number of columns in matrix B */
static float a[NRA][NCA] __attribute__ ((aligned(256)));    /* matrix A to be multiplied */
static float b[NCA][NCB] __attribute__ ((aligned(256)));     	/* matrix B to be multiplied */
static float d[NRA][NCB] __attribute__ ((aligned(256)));      	/* result matrix A*B */

void main()
{
int    i, j, k;			/* misc */

double start, end;
float temp;


   /* Initialize A, B, D*/
   for (i=0; i< NRA; i++)
      for (j=0; j< NCA; j++)
         a[i][j]= 1.;
   for (i=0; i< NCA; i++)
      for (j=0; j< NCB; j++)
         b[i][j]= 1.;

   for(i=0;i< NRA;i++)
      for(j=0;j< NCB;j++)
         d[i][j] = 0.0;



   start = omp_get_wtime();

   /* ==== >>> Original <<< ====
   for(i=0;i< NRA;i++)
   {
      for(j=0;j< NCB;j++)
      {
	      temp = 0.0;
         for(k=0;k< NCA;k++)
         {
            temp += a[i][k] * b[j][k];
         }
	      d[i][j] = temp;
      }
   }
   // ==== >>> Original <<< ====  */

   /* ==== >>> Vectorization without parallelization: m_vec_no_par (option 1) <<< ====
   for(i=0;i< NRA;i++)
   {
      #pragma omp simd collapse(2)aligned(a,b,d:256)
      for(j=0;j< NCB;j++)
      {
         for(k=0;k< NCA;k++) {
   	      d[i][j] += a[i][k] * b[j][k];
         }
      }
   }
   // ==== >>> Vectorization without parallelization: m_vec_no_par (option 1) <<< ==== */

   /* Perform matrix multiply A.BT */

   /* ==== >>> Vectorization without parallelization: m_vec_no_par (option 2) <<< ====
   for(i=0;i< NRA;i++)
   {
      for(j=0;j< NCB;j++)
      {
	      temp = 0.0;
         #pragma omp simd aligned(a,b,d:256) safelen(16)
         for(k=0;k< NCA;k++)
         {
            temp += a[i][k] * b[j][k];
         }
	      d[i][j] = temp;
      }
   }
   // ==== >>> Vectorization without parallelization: m_vec_no_par (option 2) <<< ==== */
   
   /* ==== >>> Parallelization without vectorization: m_no_vec_par <<< ====
   #pragma omp parallel for collapse(2) private(i,j,k) schedule(static)
   for(i=0;i< NRA;i++)
   {
      for(j=0;j< NCB;j++)
      {
	      temp = 0.0;
         for(k=0;k< NCA;k++)
         {
            temp += a[i][k] * b[j][k];
         }
	      d[i][j] = temp;
      }
   }
   // ==== >>> Parallelization without vectorization: m_no_vec_par <<< ==== */

   /* ==== >>> Parallelization with vectorization: m_vec_par  <<< ==== */
   #pragma omp parallel for collapse(2) private(i,j,k) schedule(static)
   for(i=0;i< NRA;i++)
   {
      for(j=0;j< NCB;j++)
      {
	      temp = 0.0;
         #pragma omp simd reduction(+:temp) aligned(a,b,d:256) safelen(16)
         for(k=0;k< NCA;k++)
         {
            temp += a[i][k] * b[j][k];
         }
	      d[i][j] = temp;
      }
   }
   // ==== >>> Parallelization witwithhout vectorization: m_vec_par  <<< ==== */

   end = omp_get_wtime();

   printf("Done\n");

   printf("d[0][0]= %f\n ", d[0][0]);
   printf("d[NRA-1][NCB-1]= %f\n ", d[NRA-1][NCB-1]);
   printf("Execution time = %g seconds\n ", end-start);
   printf ("\n");
}

