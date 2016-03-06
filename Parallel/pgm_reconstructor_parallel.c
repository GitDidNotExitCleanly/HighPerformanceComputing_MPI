#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#include "pgmio.h"

#define FILE_NAME "../Edges/edgenew192x128.pgm"
#define OUTPUT_FILE_NAME "../Reconstructed_Images/imagenew192x128_parallel.pgm"

#define M 192
#define N 128

#define PRINTFREQ  100
#define DELTA_THRESHOLD 0.1f

float boundaryval(int i, int m) {
	float val;

	val = 2.0*((float)(i-1))/((float)(m-1));
	if (i >= m/2+1) val = 2.0-val;

	return val;
}

int main (int argc, char **argv)
{
	// 1d information
	int size;	
	int myrank;
	
	// 2d information
	int rows;
	int columns;
	int myrow;
	int mycolumn;

	// new commnicator with 2d catesian
	MPI_Comm comm_2d_cart;


	/******************************* initialize MPI *******************************/

	MPI_Init(NULL,NULL);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	MPI_Comm_rank(MPI_COMM_WORLD,&myrank);


	/***************************** create 2D cartesian *****************************/

	// creates a new communicator to which 2D topology information has been attached
	rows = (size % 2 == 0) ? 2 : 1;
	columns = size / rows;

	int dims[2] = {columns,rows};
	int periods[2] = {0,1};	// periodic in the horizonal direction	
	MPI_Cart_create(MPI_COMM_WORLD,2,dims,periods,1,&comm_2d_cart);

	// determine my coordinate in the cartesian grid
	int mycoords[2];
	MPI_Cart_coords(comm_2d_cart,myrank,2,mycoords);

	myrow = mycoords[1];
	mycolumn = mycoords[0];


	/***************** image decomposition *****************/

	if (rows > (M/2) || columns > (N/2)) {
		printf("Error : Too many processes ! Pixels cannot be evenly divided. \n");
	}

	int evenly_divided_M = (M % rows == 0) ? 1 : 0;
	int standard_Mp = (int)ceil((double)M / (double)rows);
	int special_Mp = (evenly_divided_M == 0) ? M - (rows-1)*standard_Mp : standard_Mp;

	int Mp = (myrow == rows-1) ? special_Mp : standard_Mp;

	int evenly_divided_N = (N % columns == 0) ? 1 : 0;
	int standard_Np = (int)ceil((double)N / (double)columns);
	int special_Np = (evenly_divided_N == 0) ? N - (columns-1)*standard_Np : standard_Np;

	int Np = (mycolumn == columns-1) ? special_Np : standard_Np;


	/********* create necessary data structure (2D array and MPI data type) *********/

	float old[Mp+2][Np+2], new[Mp+2][Np+2], edge[Mp+2][Np+2];

	float buf[Mp][Np];

	float masterbuf[M][N];

	int i, j, iter;
	char *filename;
	float val;

	float delta;

	// new data type for data scatter and gather
	MPI_Datatype IMAGE_PARTITION_STANDARD;
	MPI_Datatype IMAGE_PARTITION_SPECIAL_M;
	MPI_Datatype IMAGE_PARTITION_SPECIAL_N;
	MPI_Datatype IMAGE_PARTITION_SPECIAL;

	if (myrank == 0) {
		// define new data type for data scatter and gather (for standard-size vector message)
		MPI_Type_vector(standard_Mp,standard_Np,N,MPI_FLOAT,&IMAGE_PARTITION_STANDARD);
		MPI_Type_commit(&IMAGE_PARTITION_STANDARD);

		// define new data type for data scatter and gather (for small count vector message)
		MPI_Type_vector(special_Mp,standard_Np,N,MPI_FLOAT,&IMAGE_PARTITION_SPECIAL_M);
		MPI_Type_commit(&IMAGE_PARTITION_SPECIAL_M);

		// define new data type for data scatter and gather (for small blocklength vector message)
		MPI_Type_vector(standard_Mp,special_Np,N,MPI_FLOAT,&IMAGE_PARTITION_SPECIAL_N);
		MPI_Type_commit(&IMAGE_PARTITION_SPECIAL_N);

		// define new data type for data scatter and gather (for small count and small small blocklength vector message)
		MPI_Type_vector(special_Mp,special_Np,N,MPI_FLOAT,&IMAGE_PARTITION_SPECIAL);
		MPI_Type_commit(&IMAGE_PARTITION_SPECIAL);
	}

	// define new data type for halo swapping with left or right process
	MPI_Datatype LEFT_OR_RIGHT_EDGE;
	MPI_Type_vector(Mp,1,Np+2,MPI_FLOAT,&LEFT_OR_RIGHT_EDGE);
	MPI_Type_commit(&LEFT_OR_RIGHT_EDGE);


	/************************************* read file *************************************/	

	// master process only
	if (myrank == 0) {
		printf("Processing %d x %d image\n", M, N);

		filename = FILE_NAME;
		printf("\nReading <%s>\n", filename);

		pgmread(filename, masterbuf, M, N);
		printf("\n");
	}


	/************** scatter data to other processes **************/

	// correct but inefficient 
	if (myrank == 0) {
		// current position in masterbuf : (m,n)
		int m = 0;
		int n = 0;
		// current size : temp_Mp*temp_Np
		int temp_Mp;
		int temp_Np;
		// current position in 2D cartesian
		int currentcoords[2];
		int targetrank;
		// local copy index
		int row_index;
		int column_index;

		for (i=0;i<rows;i++) {
			n = 0;			
			temp_Mp = (i == rows-1) ? special_Mp : standard_Mp;

			for (j=0;j<columns;j++) {			
				// find target rank
				currentcoords[0] = j;
				currentcoords[1] = i;
				MPI_Cart_rank(comm_2d_cart,currentcoords,&targetrank);

				temp_Np = (j == columns-1) ? special_Np : standard_Np;

				if (targetrank == 0) {
					// local copy
					for (row_index=0;row_index<temp_Mp;row_index++) {
						for (column_index=0;column_index<temp_Np;column_index++) {
							buf[row_index][column_index] = masterbuf[m+row_index][n+column_index];
						}
					}														
				}
				else {
					// use corresponding data type to send message

					if (i == rows-1 && j == columns-1) {
						MPI_Send(&masterbuf[m][n],1,IMAGE_PARTITION_SPECIAL,targetrank,0,comm_2d_cart);
					}
					else if (i != rows-1 && j == columns-1) {
						MPI_Send(&masterbuf[m][n],1,IMAGE_PARTITION_SPECIAL_N,targetrank,0,comm_2d_cart);
					}
					else if (i == rows-1 && j != columns-1) {
						MPI_Send(&masterbuf[m][n],1,IMAGE_PARTITION_SPECIAL_M,targetrank,0,comm_2d_cart);
					}
					else {
						MPI_Send(&masterbuf[m][n],1,IMAGE_PARTITION_STANDARD,targetrank,0,comm_2d_cart);
					}
				}
				n += temp_Np;
			}
			m += temp_Mp;
		}
	}
	else {
		MPI_Recv(buf,Mp*Np,MPI_FLOAT,0,0,comm_2d_cart,MPI_STATUS_IGNORE);
	}


	/******************** initialize 2D arrays (edge,old) on each process ********************/

	for (i=1;i<Mp+1;i++) {
		for (j=1;j<Np+1;j++) {
			edge[i][j]=buf[i-1][j-1];
		}
	}

	for (i=0; i<Mp+2;i++) {
		for (j=0;j<Np+2;j++) {
			old[i][j]=255.0;
		}
	}

	// Set fixed boundary conditions on the top and bottom edges
	for (i=1; i < Mp+1; i++) {
		// compute sawtooth vaclue
		val = boundaryval(myrow*Mp+i, M);

		old[i][0]   = 255.0*val;
		old[i][Np+1] = 255.0*(1.0-val);
	}


	/************** begin main iteration **************/

	// find neighbour id
	int top;
	int down;
	int left;
	int right;
	MPI_Cart_shift(comm_2d_cart,1,1,&top,&down);
	MPI_Cart_shift(comm_2d_cart,0,1,&left,&right);

	// non-periodic top and bottom edges
	if (left < 0) left = MPI_PROC_NULL;
	if (right < 0) right = MPI_PROC_NULL;

	// non-blocking commnication handles
	MPI_Request top_receive_request;
	MPI_Request down_receive_request;
	MPI_Request top_send_request;
	MPI_Request down_send_request;

	MPI_Request left_receive_request;
	MPI_Request right_receive_request;
	MPI_Request left_send_request;
	MPI_Request right_send_request;

	// initialize delta
	delta = DELTA_THRESHOLD;

	// start recording time
	MPI_Barrier(comm_2d_cart);
	double startTime = MPI_Wtime(); 

	for (iter=1; delta >= DELTA_THRESHOLD; iter++) {

		/****************** halo swapping --- issue communication ******************/

		if (rows > 1) {
			/* if rows > 1, implement periodic boundary conditions on left and right sides by commnication */

			// receive from top
			MPI_Irecv(old[0],Np+2,MPI_FLOAT,top,1,comm_2d_cart,&top_receive_request);	
			// receive from down
			MPI_Irecv(old[Mp+1],Np+2,MPI_FLOAT,down,2,comm_2d_cart,&down_receive_request);

			// send to top
			MPI_Isend(old[1],Np+2,MPI_FLOAT,top,2,comm_2d_cart,&top_send_request);
			// send to down
			MPI_Isend(old[Mp],Np+2,MPI_FLOAT,down,1,comm_2d_cart,&down_send_request);	
		}
		else {
			/* otherwsie, do it locally */

			// Implement periodic boundary conditions on left and right sides (local copy)
			for (j=1; j < Np+1; j++) {
				old[0][j]   = old[Mp][j];
				old[Mp+1][j] = old[1][j];
			}
		}

		if (columns > 1) {
			// receive from left
			MPI_Irecv(&old[1][0],1,LEFT_OR_RIGHT_EDGE,left,0,comm_2d_cart,&left_receive_request);
			// receive from right
			MPI_Irecv(&old[1][Np+1],1,LEFT_OR_RIGHT_EDGE,right,0,comm_2d_cart,&right_receive_request);

			// send to left
			MPI_Isend(&old[1][1],1,LEFT_OR_RIGHT_EDGE,left,0,comm_2d_cart,&left_send_request);
			// send to right
			MPI_Isend(&old[1][Np],1,LEFT_OR_RIGHT_EDGE,right,0,comm_2d_cart,&right_send_request);
		}
		

		/********** overlap some computation with non-blocking communication **********/

		// get new image (communication free part)
		for (i=2;i<Mp;i++) {
			for (j=2;j<Np;j++) {
				new[i][j]=0.25*(old[i-1][j]+old[i+1][j]+old[i][j-1]+old[i][j+1] - edge[i][j]);
			}
		}


		/******************* halo swapping --- wait for swapped part *******************/

		if (rows > 1) {
			/* if rows > 1, implement periodic boundary conditions on left and right sides by commnication */

			// wait for top edge to be finished
			MPI_Wait(&top_send_request,MPI_STATUS_IGNORE);
			MPI_Wait(&top_receive_request,MPI_STATUS_IGNORE);			

			// wait for bottom edge to be finished
			MPI_Wait(&down_send_request,MPI_STATUS_IGNORE);
			MPI_Wait(&down_receive_request,MPI_STATUS_IGNORE);
		}

		for (j=2;j<Np;j++) {
			// get new image (top excluding top-left and top-right corners)
			new[1][j]=0.25*(old[1-1][j]+old[1+1][j]+old[1][j-1]+old[1][j+1] - edge[1][j]);
			// get new image (down excluding bottom-left and bottom-right corners)
			new[Mp][j]=0.25*(old[Mp-1][j]+old[Mp+1][j]+old[Mp][j-1]+old[Mp][j+1] - edge[Mp][j]);
		}

		if (columns > 1) {
			// wait for left edge to be finished
			MPI_Wait(&left_send_request,MPI_STATUS_IGNORE);
			MPI_Wait(&left_receive_request,MPI_STATUS_IGNORE);

			// wait for right edge to be finished
			MPI_Wait(&right_send_request,MPI_STATUS_IGNORE);
			MPI_Wait(&right_receive_request,MPI_STATUS_IGNORE);
		}

		for (i=1;i<Mp+1;i++) {
			// get new image (left including top-left and top-right corners)
			new[i][1]=0.25*(old[i-1][1]+old[i+1][1]+old[i][0]+old[i][2] - edge[i][1]);
			// get new image (right including bottom-left and bottom-right corners)
			new[i][Np]=0.25*(old[i-1][Np]+old[i+1][Np]+old[i][Np-1]+old[i][Np+1] - edge[i][Np]);
		}


		/**** print average value of pixels and calculate delta for terminating the loop ****/

		if (iter % PRINTFREQ == 0) {

			float local_max = -1;
			float temp = -1;

			for (i=1;i<Mp+1;i++) {
				for (j=1;j<Np+1;j++) {
					// calculate local delta
					if (local_max == -1) {
						local_max = abs(new[i][j]-old[i][j]);
					}
					else {
						temp = abs(new[i][j]-old[i][j]);
						if (temp > local_max) {
							local_max = temp;				
						}
					}
				}
			}

			// get global delta
			MPI_Allreduce(&local_max,&delta,1,MPI_FLOAT,MPI_MAX,comm_2d_cart);
		}


		/**************************** end of one iteration ****************************/

		// store new back to old for next iteration
		for (i=1;i<Mp+1;i++) {
			for (j=1;j<Np+1;j++) {
				old[i][j]=new[i][j];
			}
		}
	}

	// stop recording time
	MPI_Barrier(comm_2d_cart);
	double endTime = MPI_Wtime(); 

	/******************* finished all iterations *******************/

	if (myrank == 0) {
		printf("\nFinished %d iterations\n", iter-1);
	}

	// copy result back to private buf  
	for (i=1;i<Mp+1;i++) {
		for (j=1;j<Np+1;j++) {
			buf[i-1][j-1]=old[i][j];
		}
	}


	/************** gather data from other processes **************/

	// correct but inefficient
	if (myrank == 0) {
		// current position in masterbuf : (m,n)
		int m = 0;
		int n = 0;
		// current size : temp_Mp*temp_Np
		int temp_Mp;
		int temp_Np;
		// current position in 2D cartesian
		int currentcoords[2];
		int targetrank;
		// local copy index
		int row_index;
		int column_index;
		
		for (i=0;i<rows;i++) {
			n = 0;			
			temp_Mp = (i == rows-1) ? special_Mp : standard_Mp;

			for (j=0;j<columns;j++) {			
				// find target rank
				currentcoords[0] = j;
				currentcoords[1] = i;
				MPI_Cart_rank(comm_2d_cart,currentcoords,&targetrank);

				temp_Np = (j == columns-1) ? special_Np : standard_Np;

				if (targetrank == 0) {
					// local copy
					for (row_index=0;row_index<temp_Mp;row_index++) {
						for (column_index=0;column_index<temp_Np;column_index++) {
							masterbuf[m+row_index][n+column_index] = buf[row_index][column_index];
						}
					}														
				}
				else {
					// use corresponding data type to receive message

					if (i == rows-1 && j == columns-1) {
						MPI_Recv(&masterbuf[m][n],1,IMAGE_PARTITION_SPECIAL,targetrank,0,comm_2d_cart,MPI_STATUS_IGNORE);
					}
					else if (i != rows-1 && j == columns-1) {
						MPI_Recv(&masterbuf[m][n],1,IMAGE_PARTITION_SPECIAL_N,targetrank,0,comm_2d_cart,MPI_STATUS_IGNORE);
					}
					else if (i == rows-1 && j != columns-1) {
						MPI_Recv(&masterbuf[m][n],1,IMAGE_PARTITION_SPECIAL_M,targetrank,0,comm_2d_cart,MPI_STATUS_IGNORE);
					}
					else {
						MPI_Recv(&masterbuf[m][n],1,IMAGE_PARTITION_STANDARD,targetrank,0,comm_2d_cart,MPI_STATUS_IGNORE);
					}		
				}
				n += temp_Np;
			}
			m += temp_Mp;
		}
	}
	else {
		MPI_Send(buf,Mp*Np,MPI_FLOAT,0,0,comm_2d_cart);
	}


	/********************** write result into ppm file **********************/

	// master process only
	if (myrank == 0) {
		filename = OUTPUT_FILE_NAME;
		printf("\nWriting <%s>\n", filename); 
		pgmwrite(filename, masterbuf, M, N);

		// print out total time
		printf("\nRuntime = %lf seconds\n",(endTime-startTime));	
	}


	/********** finalize MPI **********/

	if (myrank == 0) {
		MPI_Type_free(&IMAGE_PARTITION_STANDARD);
		MPI_Type_free(&IMAGE_PARTITION_SPECIAL_M);
		MPI_Type_free(&IMAGE_PARTITION_SPECIAL_N);
		MPI_Type_free(&IMAGE_PARTITION_SPECIAL);
	}

	MPI_Type_free(&LEFT_OR_RIGHT_EDGE);

	MPI_Finalize();
	return 0;
}
