//Parallel Bubble Sort using Openmp
#include<bits/stdc++.h>
#include<omp.h>
using namespace std;


void bubble(int a[], int n)
{
	for( int i = 0; i < n; i++ )
	 {
		int first = i % 2;
		#pragma omp parallel for shared(a,first)
		for( int j = first; j < n-1; j += 2 )
		{
			if( a[j] > a[j+1] )
			 {
			 	swap( a[j], a[j+1] );
			 }
		}
	}
	//return a;
}

void bubble_seq(int a[], int n)
{
	for( int i = 0; i < n-1; i++ )
	 {
		for( int j = 0; j < n-i-1; j += 1 )
		{
			if( a[j] > a[j+1] )
			 {
			 	swap( a[j], a[j+1] );
			 }
		}
	}
	//return a;
}
int main()
{
	int *a,*a_copy,n;
	cout<<"\n enter total no of elements=>";
	cin>>n;
	a=new int[n];
	a_copy=new int[n];
	//cout<<"\n enter elements=>";
	#pragma omp parallel for shared(a,n)
	for(int i=0;i<n;i++)
	{
		a[i]=rand()%10000;
		a_copy[i]=a[i];
	}
	cout<<"\nElements are==";
	for(int i=0;i<n;i++)
	{
	cout<<a[i]<<"\t";
	}
	int *result_seq=new int[n];
	int *result_parallel=new int[n];
	
	
	
	cout<<"\n******Parallel*****";
	double start=omp_get_wtime();
	
	bubble(a,n);
	
	double end=omp_get_wtime();
	
	cout<<"\n sorted array is=>\n";
	
	for(int i=0;i<n;i++)
	{
		cout<<a[i]<<"\n";
	}
	
	cout<<"\ntime for parallel=="<<end-start;
	
	cout<<"\n******Sequential*****";
	double start_seq=omp_get_wtime();
	
	bubble(a_copy,n);
	
	double end_seq=omp_get_wtime();
	/*
	cout<<"\n sorted array is=>\n";
	for(int i=0;i<n;i++)
	{
		cout<<result_seq[i]<<"\n";
	}
	*/
	cout<<"\ntime for sequential=="<<end_seq-start_seq<<"\n";
	return 0;
}
/*
 g++ bubble.cpp -fopenmp
*/
