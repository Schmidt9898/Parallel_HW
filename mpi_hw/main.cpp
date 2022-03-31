#include <random>
#include <iostream>
#include <chrono>
#include <ctime>
#include <vector>
#include <cstdlib>

#include <mpi.h>


  //std::vector<real> a(N*N);
  //std::vector<real> b(N*N);
  //std::vector<real> c(N*N);
float* get_part_mat(float * M,int rank_n,int tasksize,int n,int N)
{
	int y=rank_n%n;
	int x=rank_n/n;
	//std::cout<<x<<" x,"<<y<<" y\n";


	float * m=new float[n*n]();
	int size_x=n*x,size_y=n*y;
	for (int i=0;i<n;i++)
	{
		for (int j=0;j<n;j++)
		{
	//std::cout<<i<<" x,"<<j<<" y\n";
			m[i*n+j]=M[(size_x+i)*N+j];
		}
	}
	return m;
}
void put_part_mat(float * m,float * M,int rank_n,int tasksize,int n,int N)
{
	int y=rank_n%n;
	int x=rank_n/n;

	//std::cout<<x<<" x,"<<y<<" y\n";

	int size_x=n*x,size_y=n*y;
	for (int i=0;i<n;i++)
	{
		for (int j=0;j<n;j++)
		{
			M[(size_x+i)*N+j]=m[i*n+j];
		}
	}
}







void swap(float* &a,float *&b)
{
	float * c=a;
	a=b;
	b=c;
}

void MM(float *a,float *b,float *c,int N)
{
	//#pragma omp parallel for
	for (int i = 0; i < N; i++)// sor a
		for(int k=0;k<N;k++)
			for (int j = 0; j < N; j++)//oszlop b
				c[i*N+j] += a[i*N+k] * b[k*N+j];
}
void MA(float *a,float *b,int N)
{
	//#pragma omp parallel for
	for (int i = 0; i < N; i++)// sor a
			b[i] += a[i];
}






int main(int argc, char *argv[])  {

int n=3;
float* ai;
float* bi;
float* ao=new float[n*n]();
float* bo=new float[n*n]();
float* c=new float[n*n]();
//float * temp;
float* temp=new float[n*n]();
//std::vector<float> tempB(n*n,0);



int N=9;
std::vector<float> A(N*N);
std::vector<float> B(N*N);
std::vector<float> C(N*N);


int task_need=(N/n)*(N/n);



int numtasks, rank;
  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
//MPI_Comm_rank(MPI_COMM_WORLD, &rank);
/*
// Obtain the group of processes in the world communicator
MPI_Group world_group;
MPI_Comm_group(MPI_COMM_WORLD, &world_group);

// Remove all unnecessary ranks
MPI_Group new_group;
int ranges[3] = { task_need, numtasks-1, 1 };
MPI_Group_range_excl(world_group, 1, ranges, &new_group);

// Create a new communicator
MPI_Comm newworld;
MPI_Comm_create(MPI_COMM_WORLD, new_group, &newworld);

if (newworld == MPI_COMM_NULL)
{
   // Bye bye cruel world
   MPI_Finalize();
   exit(0);
}

*/


if(task_need>numtasks)
	{
		std::cout<<task_need<<">"<<numtasks<<"MPI.\n";
  		MPI_Finalize();
		return 0;
	}




	MPI_Request reqs[4];
	MPI_Status stats[4];
	MPI_Comm cartcomm;
	int jobb,bal,fel,le;
	int  dims[2]={n,n} ;
	int periods[2]={1,1}, reorder=0, coords[2];
    //Create cartesian communicator for 2D, dims[0]*dims[1] processes,
    //without periodicity and reordering
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cartcomm);
    //Get my rank in the new communicator
    MPI_Comm_rank(cartcomm, &rank);
    //Get my coordinates coords[0] and coords[1]
    MPI_Cart_coords(cartcomm, rank, 2, coords);
    //Get my neighbours in dimension 0
    MPI_Cart_shift(cartcomm, 0, 1, &fel, &le);
    //Get my neighbours in dirmension 1
    MPI_Cart_shift(cartcomm, 1, 1, &bal, &jobb);

    //printf("rank= %d coords= %d %d  neighbors(u,d,l,r)= %d %d %d %d\n",
    //    rank,coords[0],coords[1],fel,le,bal,jobb);

if(rank==0)
{

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
		A[i*N+j] = 1;
		if(i==j)
			B[i*N+j] = 1;
			
		}
	}




	ai=get_part_mat(A.data(),0,numtasks,n,N);
	bi=get_part_mat(B.data(),0,numtasks,n,N);
	for(int i=1;i<numtasks;i++)
		{
		float* t=get_part_mat(A.data(),i,numtasks,n,N);
		MPI_Send(t, n*n, MPI_FLOAT, i, 1, MPI_COMM_WORLD);
		delete[] t;
		t=get_part_mat(B.data(),i,numtasks,n,N);
		MPI_Send(t, n*n, MPI_FLOAT, i, 2, MPI_COMM_WORLD);
		delete[] t;
		}
}else
	{
		ai=new float[n*n]();
		bi=new float[n*n]();
		MPI_Status s;
		MPI_Recv(ai, n*n, MPI_FLOAT, 0, 1,MPI_COMM_WORLD, &s);
		MPI_Recv(bi, n*n, MPI_FLOAT, 0, 2,MPI_COMM_WORLD, &s);
	}




	//calculate minimatrix


	//init
	int x=coords[0],y=coords[1];
	//C=A(x,0)*B(0,y) + A(x,1)*B(1,y) + A(x,2)*B(2,y)
	//c=A(x,0)*B(0,y) + A(x,1)*B(1,y) + A(x,2)*B(2,y)
	//skewing
	//xa by
	for(int i=0;i<x;i++)
	{

	swap(ai,ao);//put into send buff
	MPI_Isend(ao, n*n, MPI_FLOAT, bal, 1, MPI_COMM_WORLD, &reqs[0]);
    MPI_Irecv(ai, n*n, MPI_FLOAT, jobb, 1, MPI_COMM_WORLD, &reqs[1]);
	MPI_Waitall(2, reqs, stats);
	}
	for(int i=0;i<y;i++)
	{
	swap(bi,bo);//put into send buff
	MPI_Isend(bo, n*n, MPI_FLOAT, fel, 2, MPI_COMM_WORLD, &reqs[0]);
    MPI_Irecv(bi, n*n, MPI_FLOAT, le, 2, MPI_COMM_WORLD, &reqs[1]);
	MPI_Waitall(2, reqs, stats);
	}
	
	MM(ai,bi,temp,n);

	MA(temp,c,n);
	//k loop
	int k=N/n -1;
	for (int i=0;i<k;i++)
	{


	swap(ai,ao);//put into send buff
	swap(bi,bo);//put into send buff
	MPI_Isend(ao, n*n, MPI_FLOAT, bal, 1, MPI_COMM_WORLD, &reqs[0]);
	MPI_Isend(bo, n*n, MPI_FLOAT, fel, 2, MPI_COMM_WORLD, &reqs[1]);

    MPI_Irecv(ai, n*n, MPI_FLOAT, jobb, 1, MPI_COMM_WORLD, &reqs[2]);
    MPI_Irecv(bi, n*n, MPI_FLOAT, le, 2, MPI_COMM_WORLD, &reqs[3]);

	MPI_Waitall(4, reqs, stats);

	MM(ai,bi,temp,n);
	MA(temp,c,n);
	//std::cout<<c[0]<<" temp.\n";
	}




if(rank==0)
{
	put_part_mat(c,C.data(),0,numtasks,n,N);
std::cout<<"gather.\n";

	float* t=new float[n*n]();
for(int i=1;i<numtasks;i++)
	{
	MPI_Status s;
	MPI_Recv(t, n*n, MPI_FLOAT, i, 3,MPI_COMM_WORLD, &s);
	put_part_mat(t,C.data(),i,numtasks,n,N);
//std::cout<<t[0]<<"t\n";
	}
	delete[] t;
std::cout<<"done gather.\n";
	
}
else
	{
		MPI_Send(c, n*n, MPI_FLOAT, 0, 3, MPI_COMM_WORLD);
		//std::cout<<"send.\n";
	}



if(rank==0)
{


std::vector<float> C_test(N*N);
MM(A.data(),B.data(),C_test.data(),N);
int i=0;
for(i=0;i<C.size();i++)
{
	//std::cout<<C[i]<<"\n";
	//if(C[i]!=C_test[i]){
		std::cout<<i<<" i "<<C[i]<<"!="<<C_test[i]<<"\n";//<<"failed\n";
	//break;
	//}
}
//if(i>=C.size())
//	std::cout<<"passed\n";



}





  MPI_Finalize();
return 0;

}

