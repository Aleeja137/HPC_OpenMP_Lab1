// Solo para 2 procesos MPI.
#include <stdio.h>
#include "mpi.h"
#include <omp.h>

int funcion(int myid, int npes, int iam, int nt)
{
    return(npes*1000000 + myid*10000 + nt*100 + iam);
}

int main(int argc, char *argv[]) {
  int npes, myid, namelen, proporcionado;
  char node_name[MPI_MAX_PROCESSOR_NAME];
  int iam = 0, nt = 1, r1, r2;
  MPI_Status estado;

  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &proporcionado);
  printf("Provided level %d de %d, %d, %d, %d\n", proporcionado, MPI_THREAD_SINGLE, 
		 MPI_THREAD_FUNNELED, MPI_THREAD_SERIALIZED, MPI_THREAD_MULTIPLE);
  MPI_Comm_size(MPI_COMM_WORLD, &npes);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Get_processor_name(node_name, &namelen);

  #pragma omp parallel default(shared) private(iam, nt, r1, r2)
  {
    nt = omp_get_num_threads();
    iam = omp_get_thread_num();
    r1 = funcion(myid, npes, iam, nt);
    if(myid == 0)
	MPI_Send(&r1, 1, MPI_INT, 1, iam, MPI_COMM_WORLD);
	// MPI_Send(&r1, 1, MPI_INT, 1, (iam+1)%nt, MPI_COMM_WORLD);
	// Para que el hilo 0 reciba del hilo nt-1, para no reciban de hilos homónimos
    else
	MPI_Recv(&r2, 1, MPI_INT, 0, iam, MPI_COMM_WORLD, &estado);
    if(myid == 1)
	MPI_Send(&r1, 1, MPI_INT, 0, iam, MPI_COMM_WORLD);
	// MPI_Send(&r1, 1, MPI_INT, 0, (iam+1)%nt, MPI_COMM_WORLD);
    else
	MPI_Recv(&r2, 1, MPI_INT, 1, iam, MPI_COMM_WORLD, &estado);

    printf("Hello from thread %d out of %d from process %d out of %d on %s the data is %d\n",
           iam, nt, myid, npes, node_name, r2);
  }

  MPI_Finalize();
}


