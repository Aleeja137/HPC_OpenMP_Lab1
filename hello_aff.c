#define _GNU_SOURCE
#include <stdio.h>
#include <sched.h>
#include <omp.h>

#define N 100

int main(int argc, char *argv[]) {
  int i, iam, nt, aff, nplaces, myplace, procs_place, ids[N], cpu_id, nplaces_mypartition, mypartition_ids[N];
  
   aff = omp_get_proc_bind();
   nplaces = omp_get_num_places();
   printf("Affinity %d with %d places\n",aff, nplaces);
  
  #pragma omp parallel default(shared) private(iam, nt, myplace, procs_place, ids, i, cpu_id, nplaces_mypartition, mypartition_ids)
  {
    nt = omp_get_num_threads();
    iam = omp_get_thread_num();
    cpu_id = sched_getcpu();
    myplace = omp_get_place_num();
    procs_place = omp_get_place_num_procs(myplace);
    omp_get_place_proc_ids(myplace, ids);
    nplaces_mypartition = omp_get_partition_num_places();
    omp_get_partition_place_nums(mypartition_ids);
    
    printf("Hello from thread %d out of %d. My place is %d out of %d with %d processors. My partition has %d places. I'm on core %d\n",
                             iam,       nt,         myplace,   nplaces, procs_place,              nplaces_mypartition, cpu_id);

    #pragma omp single
    {
    	for(i=0; i<procs_place; i++)
           printf("%d ",ids[i]);
        printf("\nPrinted by thread %d\n",iam);
        for(i=0; i<nplaces_mypartition; i++)
           printf("%d ",mypartition_ids[i]);
        printf("\nPrinted by thread %d\n",iam);
    }
  }
  return 0; 
}


