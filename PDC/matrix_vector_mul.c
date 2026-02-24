#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char* argv[]) {
    int my_rank, comm_sz;
    int m = 4;   // number of rows
    int n = 4;   // number of columns

    double *A = NULL;      // full matrix (rank 0)
    double *x = NULL;      // vector x (all ranks)
    double *y = NULL;      // result vector (rank 0)

    double *local_A;       // local rows of A
    double *local_y;       // local result

    int local_m;           // rows per process

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    local_m = m / comm_sz;

    /* Allocate local memory */
    local_A = (double*)malloc(local_m * n * sizeof(double));
    local_y = (double*)malloc(local_m * sizeof(double));
    x = (double*)malloc(n * sizeof(double));

    /* Rank 0 initializes matrix and vector */
    if (my_rank == 0) {
        A = (double*)malloc(m * n * sizeof(double));
        y = (double*)malloc(m * sizeof(double));

        // Initialize matrix A
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                A[i*n + j] = i + j + 1;

        // Initialize vector x
        for (int j = 0; j < n; j++)
            x[j] = 1.0;
    }

    /* Distribute rows of A */
    MPI_Scatter(A, local_m * n, MPI_DOUBLE,
                local_A, local_m * n, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    /* Broadcast vector x */
    MPI_Bcast(x, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Local matrix-vector multiplication */
    for (int i = 0; i < local_m; i++) {
        local_y[i] = 0.0;
        for (int j = 0; j < n; j++) {
            local_y[i] += local_A[i*n + j] * x[j];
        }
    }

    /* Gather result vector y */
    MPI_Gather(local_y, local_m, MPI_DOUBLE,
               y, local_m, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    /* Rank 0 prints result */
    if (my_rank == 0) {
        printf("Result vector y:\n");
        for (int i = 0; i < m; i++) {
            printf("%.2f\n", y[i]);
        }
    }

    /* Free memory */
    free(local_A);
    free(local_y);
    free(x);

    if (my_rank == 0) {
        free(A);
        free(y);
    }

    MPI_Finalize();
    return 0;
}