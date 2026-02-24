#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char* argv[]) {

    int my_rank, comm_sz;
    int m = 4, n = 4, p = 4;

    double *A = NULL, *B = NULL, *C = NULL;
    double *local_A, *local_C;

    int local_m;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    local_m = m / comm_sz;

    local_A = malloc(local_m * n * sizeof(double));
    local_C = malloc(local_m * p * sizeof(double));
    B = malloc(n * p * sizeof(double));

    if (my_rank == 0) {
        A = malloc(m * n * sizeof(double));
        C = malloc(m * p * sizeof(double));

        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                A[i*n + j] = i + j + 1;

        for (int i = 0; i < n; i++)
            for (int j = 0; j < p; j++)
                B[i*p + j] = 1.0;
    }

    /* Scatter A */
    MPI_Scatter(A, local_m * n, MPI_DOUBLE,
                local_A, local_m * n, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    /* Broadcast B */
    MPI_Bcast(B, n * p, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Local multiplication */
    for (int i = 0; i < local_m; i++) {
        for (int j = 0; j < p; j++) {
            local_C[i*p + j] = 0.0;
            for (int k = 0; k < n; k++) {
                local_C[i*p + j] += local_A[i*n + k] * B[k*p + j];
            }
        }
    }

    /* Gather C */
    MPI_Gather(local_C, local_m * p, MPI_DOUBLE,
               C, local_m * p, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        printf("Result matrix C:\n");
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < p; j++)
                printf("%.2f ", C[i*p + j]);
            printf("\n");
        }
    }

    free(local_A);
    free(local_C);
    free(B);

    if (my_rank == 0) {
        free(A);
        free(C);
    }

    MPI_Finalize();
    return 0;
}