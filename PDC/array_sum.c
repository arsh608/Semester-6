#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, np;
    int size = 20;              // total number of elements
    double *arr = NULL;         // full array (master only)
    int local_size;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    /* ---------------- MASTER CODE ---------------- */
    if (rank == 0) {
        int i, index;

        arr = (double *)malloc(size * sizeof(double));

        // initialize array
        for (i = 0; i < size; i++) {
            arr[i] = i + 1;
        }

        local_size = size / np;

        // send equal chunks to processes 1 .. np-2
        for (i = 1; i < np - 1; i++) {
            index = i * local_size;

            MPI_Send(&local_size, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&arr[index], local_size, MPI_DOUBLE,
                     i, 1, MPI_COMM_WORLD);
        }

        // last process gets remaining elements
        index = i * local_size;
        int elements_left = size - index;

        MPI_Send(&elements_left, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        MPI_Send(&arr[index], elements_left, MPI_DOUBLE,
                 i, 1, MPI_COMM_WORLD);

        // master can also compute its own part
        double sum = 0.0;
        for (i = 0; i < local_size; i++) {
            sum += arr[i];
        }

        printf("Master computed local sum = %.2f\n", sum);

        free(arr);
    }

    /* ---------------- SLAVE CODE ---------------- */
    else {
        double local_sum = 0.0;

        // receive number of elements
        MPI_Recv(&local_size, 1, MPI_INT,
                 0, 0, MPI_COMM_WORLD, &status);

        printf("Process %d received size %d\n", rank, local_size);

        // allocate space for received data
        double *arr2 = (double *)malloc(local_size * sizeof(double));

        // receive array chunk
        MPI_Recv(arr2, local_size, MPI_DOUBLE,
                 0, 1, MPI_COMM_WORLD, &status);

        // compute local sum
        for (int i = 0; i < local_size; i++) {
            local_sum += arr2[i];
        }

        printf("Process %d local sum = %.2f\n", rank, local_sum);

        free(arr2);
    }

    MPI_Finalize();
    return 0;
}
