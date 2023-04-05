#include<stdio.h>
#include<mpi.h>
#define A_rows 1000 //number of rows of A matrix
#define A_columns 1000 //number of columns of A matrix
#define B_rows 1000 //number of rows of B matrix
#define B_columns 1000 //number of columns of B matrix
#define MASTER_TO_SLAVE_TAG 1 //tag for messages sent from master to slaves
#define SLAVE_TO_MASTER_TAG 4 //tag for messages sent from slaves to master
void create_matrix(); //creates the A and B matrices
void print_matrix(); //print the all the matrices;

int rank; //process rank
int size; //number of processes
int i, j, k; //helper variables
double mat_a[A_rows][A_columns]; //declare input matrix A
double mat_b[B_rows][B_columns]; //declare input matrix B
double mat_result[A_rows][B_columns]; //declare output matrix C
double start_time; //hold start time
double end_time; //hold end time
int low_bound; //low bound of the number of rows of matrix A allocated to a slave
int upper_bound; //upper bound of the number of rows of matrix A allocated to a slave
int portion; //portion of the number of rows of matrix A allocated to a slave
MPI_Status status; //store status of a MPI_Recv
MPI_Request request; //capture request of a MPI_Isend

void create_matrix()
{
	for (i = 0; i < A_rows; i++)
	{
		for (j = 0; j < A_columns; j++) 
			mat_a[i][j] = i + j;
    }
	
	for (i = 0; i < B_rows; i++)
	{
        for (j = 0; j < B_columns; j++) 
			mat_b[i][j] = i * j;
    }
}

void print_matrix()
{
    printf("Matrix A =");
	for (i = 0; i < A_rows; i++)
	{
        printf("\n");
        for (j = 0; j < A_columns; j++)
            printf("%8.2f  ", mat_a[i][j]);
    }
    
	printf("\n\n\n");
	printf("Matrix B =");
    for (i = 0; i < B_rows; i++) 
	{
        printf("\n");
        for (j = 0; j < B_columns; j++)
            printf("%8.2f  ", mat_b[i][j]);
    }

	printf("\n\n\n");
	printf("Matrix A * Matrix B =");
	for (i = 0; i < A_rows; i++)
	{
        printf("\n");
        for (j = 0; j < B_columns; j++)
            printf("%8.2f  ", mat_result[i][j]);
	}
    printf("\n\n");
}

int main(int argc, char *argv[])
{

    MPI_Init(&argc, &argv); //initialize MPI operations
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); //get the rank
    MPI_Comm_size(MPI_COMM_WORLD, &size); //get number of processes

    /* master initializes work*/
    if (rank == 0)
    {
        create_matrix();
        start_time = MPI_Wtime();
        for (i = 1; i < size; i++)
        {   
            //for each slave other than the master
            portion = (A_rows / (size - 1)); // calculate portion without master
            low_bound = (i - 1) * portion;
            if (((i + 1) == size) && ((A_rows % (size - 1)) != 0))
            {   //if rows of matrix A cannot be equally divided among slaves
                upper_bound = A_rows; //last slave gets all the remaining rows
            }
            else
            {
                upper_bound = low_bound + portion; //rows of matrix A are equally divisable among slaves
            }
            //send the low bound first without blocking, to the intended slave
            MPI_Isend(&low_bound, 1, MPI_INT, i, MASTER_TO_SLAVE_TAG, MPI_COMM_WORLD, &request);
            //next send the upper bound without blocking, to the intended slave
            MPI_Isend(&upper_bound, 1, MPI_INT, i, MASTER_TO_SLAVE_TAG + 1, MPI_COMM_WORLD, &request);
            //finally send the allocated row portion of [A] without blocking, to the intended slave
            MPI_Isend(&mat_a[low_bound][0], (upper_bound - low_bound) * A_columns, MPI_DOUBLE, i, MASTER_TO_SLAVE_TAG + 2, MPI_COMM_WORLD, &request);
        }
    }
    //broadcast [B] to all the slaves
    MPI_Bcast(&mat_b, B_rows*B_columns, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* work done by slaves*/
    if (rank > 0)
    {
        //receive low bound from the master
        MPI_Recv(&low_bound, 1, MPI_INT, 0, MASTER_TO_SLAVE_TAG, MPI_COMM_WORLD, &status);
        //next receive upper bound from the master
        MPI_Recv(&upper_bound, 1, MPI_INT, 0, MASTER_TO_SLAVE_TAG + 1, MPI_COMM_WORLD, &status);
        //finally receive row portion of matrix A to be processed from the master
        MPI_Recv(&mat_a[low_bound][0], (upper_bound - low_bound) * A_columns, MPI_DOUBLE, 0, MASTER_TO_SLAVE_TAG + 2, MPI_COMM_WORLD, &status);
        for (i = low_bound; i < upper_bound; i++)
        {   
            //iterate through a given set of rows of matrix A
            for (j = 0; j < B_columns; j++)
            {   
                //iterate through columns of matrix B
		for (k = 0; k < B_rows; k++) 
		{	
			//iterate through rows of [B]
                	mat_result[i][j] += (mat_a[i][k] * mat_b[k][j]);
                }
            }
        }
        //send back the low bound first without blocking, to the master
        MPI_Isend(&low_bound, 1, MPI_INT, 0, SLAVE_TO_MASTER_TAG, MPI_COMM_WORLD, &request);
        //send the upper bound next without blocking, to the master
        MPI_Isend(&upper_bound, 1, MPI_INT, 0, SLAVE_TO_MASTER_TAG + 1, MPI_COMM_WORLD, &request);
        //finally send the processed portion of data without blocking, to the master
        MPI_Isend(&mat_result[low_bound][0], (upper_bound - low_bound) * B_columns, MPI_DOUBLE, 0, SLAVE_TO_MASTER_TAG + 2, MPI_COMM_WORLD, &request);
    }

    /* master gathers processed work*/
    if (rank == 0) 
    {
        for (i = 1; i < size; i++) 
        {   
            // untill all slaves have handed back the processed data
            //receive low bound from a slave
            MPI_Recv(&low_bound, 1, MPI_INT, i, SLAVE_TO_MASTER_TAG, MPI_COMM_WORLD, &status);
            //receive upper bound from a slave
            MPI_Recv(&upper_bound, 1, MPI_INT, i, SLAVE_TO_MASTER_TAG + 1, MPI_COMM_WORLD, &status);
            //receive processed data from a slave
            MPI_Recv(&mat_result[low_bound][0], (upper_bound - low_bound) * B_columns, MPI_DOUBLE, i, SLAVE_TO_MASTER_TAG + 2, MPI_COMM_WORLD, &status);
        }
        end_time = MPI_Wtime();
        printf("\nRunning Time = %f\n\n", end_time - start_time);
        //print_matrix();
    }
    MPI_Finalize(); //finalize MPI operations
    return 0;
}