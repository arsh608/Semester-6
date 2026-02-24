#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

double trap(double a, double b, int n);

double f(double x) {
    return x*x;
}

int main(int argc, char* argv[]){
    if (argc < 2) {
        printf("Usage: %s <number_of_threads>\n", argv[0]);
        return 1;
    }

    double global_result = 0.0;
    double a, b;
    int n;

    int count = strtol(argv[1], NULL, 10);
    
    printf("Enter the value of a, b and n: ");
    scanf("%lf %lf %d", &a, &b, &n);

    if (n % count != 0) {
        printf("n must be divisible by number of threads\n");
        return 1;
    }


    // #pragma omp parallel num_threads(count)
    // {
    //     double local_result = trap(a, b, n);

    //     #pragma omp critical
    //     global_result += local_result;
    // }
    
    #pragma omp parallel num_threads(count) reduction(+: global_result)
    {
        global_result += trap(a, b, n);
    }


    printf("With n=%d trapezoids, our estimate\n", n);
    printf("of the integral from %f to %f = %.14e\n",a,b,global_result);
    return 0;
}

double trap(double a, double b, int n){
    int rank = omp_get_thread_num();
    int count = omp_get_num_threads();
    double x;

    double h = (b-a)/n;
    int local_n = n/count;
    double local_a = a + rank * local_n * h;
    double local_b = local_a + local_n * h;

    double local_result = (f(local_a)+f(local_b))/2.0;
    for(int i=1; i <= local_n - 1; i++){
        x = local_a + i*h;
        local_result += f(x);
    }
    local_result = local_result * h;

    return local_result;
}