#include <stdio.h>
#include <mpi.h>

/* Function prototype for trapezoid calculation */
double Trap(double local_a, double local_b, int local_n, double h);

/* Function to get input from the master and broadcast to all processes */
void Get_input(int my_rank, int comm_sz, double* a_p, double* b_p, int* n_p);

int main(int argc, char* argv[]) {
    int my_rank, comm_sz;
    double a, b;         // Interval endpoints
    int n;               // Total number of trapezoids
    double h;            // Step size
    int local_n;         // Number of trapezoids per process
    double local_a, local_b;
    double local_int, total_int;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    /* Get input a, b, n from master and distribute to all processes */
    Get_input(my_rank, comm_sz, &a, &b, &n);

    /* Compute step size and local interval */
    h = (b - a) / n;
    local_n = n / comm_sz;
    local_a = a + my_rank * local_n * h;
    local_b = local_a + local_n * h;

    /* Each process computes its local integral */
    local_int = Trap(local_a, local_b, local_n, h);

    // /* Master gathers all local integrals */
    // if (my_rank != 0) {
    //     MPI_Send(&local_int, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    // } else {
    //     total_int = local_int;
    //     double temp;
    //     for (int source = 1; source < comm_sz; source++) {
    //         MPI_Recv(&temp, 1, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    //         total_int += temp;
    //     }
    //     printf("With n = %d trapezoids, integral = %.6f\n", n, total_int);
    // }

    MPI_Reduce(&local_int, &total_int, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        printf("With n = %d trapezoids, integral = %.6f\n", n, total_int);
    }

    MPI_Finalize();
    return 0;
}

/* Function to get input from master and broadcast to all processes */
void Get_input(int my_rank, int comm_sz, double* a_p, double* b_p, int* n_p) {
    if (my_rank == 0) {
        printf("Enter a, b, and n: ");
        scanf("%lf %lf %d", a_p, b_p, n_p);

        /* Send input to all other processes */
        for (int dest = 1; dest < comm_sz; dest++) {
            MPI_Send(a_p, 1, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
            MPI_Send(b_p, 1, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
            MPI_Send(n_p, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
        }
    } else {
        /* Receive input from master */
        MPI_Recv(a_p, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(b_p, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(n_p, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

/* Example trapezoid function: integral of f(x) = x^2 */
double Trap(double local_a, double local_b, int local_n, double h) {
    double integral;
    double x;
    int i;

    integral = (local_a * local_a + local_b * local_b) / 2.0;
    for (i = 1; i < local_n; i++) {
        x = local_a + i * h;
        integral += x * x;
    }
    integral *= h;
    return integral;
}
