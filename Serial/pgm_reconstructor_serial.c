/*
 * A simple serial solution to the Case Study exercise from the MP
 * course.  Note that this uses the alternative boundary conditions
 * that are appropriate for the assessed coursework.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "pgmio.h"

#define FILE_NAME "../Edges/edgenew192x128.pgm"
#define OUTPUT_FILE_NAME "../Reconstructed_Images/imagenew192x128_serial.pgm"

#define M 192
#define N 128

#define PRINTFREQ 100
#define DELTA_THRESHOLD 0.1f

float boundaryval(int i, int m);

int main (int argc, char **argv)
{
	float old[M+2][N+2], new[M+2][N+2], edge[M+2][N+2];

	float buf[M][N];

	int i, j, iter;
	char *filename;
	float val;

	float delta;

	printf("Processing %d x %d image\n", M, N);

	filename = FILE_NAME;

	printf("\nReading <%s>\n", filename);
	pgmread(filename, buf, M, N);
	printf("\n");


	for (i=1;i<M+1;i++)
	{
		for (j=1;j<N+1;j++)
		{
			edge[i][j]=buf[i-1][j-1];
		}
	}

	for (i=0; i<M+2;i++)
	{
		for (j=0;j<N+2;j++)
		{
			old[i][j]=255.0;
		}
	}

	/* Set fixed boundary conditions on the top and bottom edges */

	for (i=1; i < M+1; i++)
	{
		/* compute sawtooth value */

		val = boundaryval(i, M);

		old[i][0]   = 255.0*val;
		old[i][N+1] = 255.0*(1.0-val);
	}

	// initialize delta
	delta = DELTA_THRESHOLD;

	for (iter=1; delta >= DELTA_THRESHOLD; iter++)
	{
		/* Implement periodic boundary conditions on left and right sides */

		for (j=1; j < N+1; j++)
		{
			old[0][j]   = old[M][j];
			old[M+1][j] = old[1][j];
		}

		for (i=1;i<M+1;i++)
		{
			for (j=1;j<N+1;j++)
			{
				new[i][j]=0.25*(old[i-1][j]+old[i+1][j]+old[i][j-1]+old[i][j+1]
																		   - edge[i][j]);
			}
		}

		if(iter%PRINTFREQ==0) {

			float local_max = -1;
			float temp = -1;
			for (i=1;i<M+1;i++) {
				for (j=1;j<N+1;j++) {
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
			delta = local_max;
		}	

		for (i=1;i<M+1;i++)
		{
			for (j=1;j<N+1;j++)
			{
				old[i][j]=new[i][j];
			}
		}
	}

	printf("\nFinished %d iterations\n", iter-1);

	for (i=1;i<M+1;i++)
	{
		for (j=1;j<N+1;j++)
		{
			buf[i-1][j-1]=old[i][j];
		}
	}

	filename = OUTPUT_FILE_NAME;
	printf("\nWriting <%s>\n", filename); 
	pgmwrite(filename, buf, M, N);
} 

float boundaryval(int i, int m)
{
	float val;

	val = 2.0*((float)(i-1))/((float)(m-1));
	if (i >= m/2+1) val = 2.0-val;

	return val;
}
