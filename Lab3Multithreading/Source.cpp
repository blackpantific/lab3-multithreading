#include <cstdlib>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <omp.h>
//#include <CL/opencl.h>
#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#include "conio.h"
#include <vector>
#include <string>

#define _CRT_SECURE_NO_WARNINGS

using namespace std;

// ��������� �������

cl_device_id InformationAboutDevice(
	cl_platform_id* platformID,
	int numberOfDevice);

void lection4_multipl_matrix(
	cl_context context,
	cl_int status, 
	cl_command_queue queue,
	cl_kernel kernel,
	float*& matrix1,
	float*& matrix2,
	float*& resultMatrix, 
	int* NKM, 
	cl_device_id deviceID);

void multipl_matrix_local_memory(
	cl_context context,
	cl_int status, 
	cl_command_queue queue,
	cl_kernel kernel,
	float*& matrix1, 
	float*& matrix2, 
	float*& resultMatrix,
	int* NKM);

void multipl_matrix_local_memory_v4(
	cl_context context, 
	cl_int status,
	cl_command_queue queue,
	cl_kernel kernel,
	float*& matrix1, 
	float*& matrix2, 
	float*& resultMatrix,
	int* NKM);

void multipl_matrix_vector(
	cl_context context,
	cl_int status,
	cl_command_queue queue,
	cl_kernel kernel,
	float*& matrix1,
	float*& matrix2, 
	float*& resultMatrix,
	int* NKM);

void get_matrixs_from_file(
	string input_file_path,
	int NKM[],
	float*& matrix1,
	float*& matrix2,
	float*& resultMatrix);

void get_matrixs_from_file_v2(
	string input_file_path,
	int NKM[], 
	float*& matrix1,
	float*& matrix2,
	float*& resultMatrix, 
	int multiplicity);

void get_matrixs_transpose_from_file_for_vector(
	string input_file_path,
	int NKM[],
	float*& matrix1, 
	float*& matrix2,
	float*& resultMatrix);

void DeviceInfo(cl_device_id deviceID);

void KernelInfo(cl_kernel kernel, cl_device_id deviceID);

void write_matrix_to_file();

void write_matrix_to_file_v2();

void get_kernel_code_from_file();

void free_openCL();

void get_matrixs_transpose_from_file(
	string input_file_path,
	int NKM[], 
	float*& matrix1, 
	float*& matrix2, 
	float*& resultMatrix);




// ���������� ����������

int numberOfDevice = 0;//by default
string pathInputFile = "C:\\Users\\black\\Desktop\\matrix.txt";
string pathOutputFile = "C:\\Users\\black\\Desktop\\matrixResult.txt";
int numberOfRealization = 0;


int NKM[3] = { 0,0,0 };
int NKMBase[3] = { 0,0,0 };//�� ���������� �������������� �����
float* matrix1 = 0;
float* matrix2 = 0;
float* resultMatrix = 0;

cl_platform_id platformID;
cl_device_id deviceID;
cl_int status;
cl_context context;
cl_command_queue queue;
cl_program program;
size_t param_value = 0;
cl_kernel kernel = NULL;

char* buf = NULL;
const char* buf_p;
size_t sizeBuf;

cl_mem arg_buffer_a;
cl_mem arg_buffer_b;
cl_mem arg_buffer_c;

int localWorkSize = 32;
int WPT = 0;
int RTS = 0;


int main(int argc, char** argv)
{
	numberOfDevice = atoi(argv[1]);//by default
	pathInputFile = argv[2];
	pathOutputFile = argv[3];
	numberOfRealization = atoi(argv[4]);



	try {

		if (numberOfRealization == 1) {
			get_matrixs_from_file(pathInputFile, NKM, matrix1, matrix2, resultMatrix);//�������� ������ �� �������� �� �����
		}
		else if(numberOfRealization == 2){
			
			get_matrixs_from_file_v2(pathInputFile, NKM, matrix1, matrix2, resultMatrix, localWorkSize);
		}
		else if (numberOfRealization == 3) {
			get_matrixs_transpose_from_file_for_vector(pathInputFile, NKM, matrix1, matrix2, resultMatrix);
		}
		else if (numberOfRealization == 4) {
			get_matrixs_transpose_from_file(pathInputFile, NKM, matrix1, matrix2, resultMatrix);
		}





		deviceID = InformationAboutDevice(&platformID, numberOfDevice);

		cl_context_properties properties[3] = { CL_CONTEXT_PLATFORM , (cl_context_properties)platformID, 0 };

		DeviceInfo(deviceID);

		context = clCreateContext(properties, 1, &deviceID, NULL, NULL, &status);
		if (!context)
		{
			throw "Error: Failed to create a compute context!\n";
		}



		// �������� ��� �������
		if (numberOfRealization == 1) {
			get_kernel_code_from_file();
		}
		else if (numberOfRealization == 2) {
			
			get_kernel_code_from_file();

		}
		else if (numberOfRealization == 3) {
			get_kernel_code_from_file();
		}
		else if (numberOfRealization == 4) {
			get_kernel_code_from_file();
		}
		

		queue = clCreateCommandQueue(context, deviceID, CL_QUEUE_PROFILING_ENABLE, &status);
		if (!queue)
		{
			throw "Error: Failed to create a queue!\n";
		}

		program = clCreateProgramWithSource(context, 1, &buf_p, &sizeBuf, &status);
		if (!program)
		{
			throw "Error: Failed to create a program!\n";
		}


		if (NKM[0] < 8) {
			WPT = NKM[0];
		}
		else {
			WPT = 8;
		}
		RTS = localWorkSize / WPT;

		const string param_s = "-D COLSROWS=" + to_string(NKM[1]) + " -D TS=" + to_string(localWorkSize) +
			" -D WPT=" + to_string(WPT) + " -D RTS=" + to_string(RTS);//"-D COLSROWS=2 -D PSG=2";
		int size = param_s.size();
		char* parameters = new char[size + 1];
		//strcpy_s(parameters, size +1, param_s.c_str());
		strcpy_s(parameters, size + 1, param_s.c_str());

		status = clBuildProgram(program, 1, &deviceID, parameters, NULL, NULL);

		free(parameters);

		status = clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_LOG, NULL, NULL, &param_value);

		char* log = NULL;
		if (param_value != 0)
		{
			log = (char*)malloc(sizeof(char));
			status = clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_LOG, param_value, log, NULL);
			printf("\n%s", log);
		}
		if (status!= CL_SUCCESS)
		{
			throw "Error: Failed to build a program!\n";
		}



		if (numberOfRealization == 1) {
			kernel = clCreateKernel(program, "matrix_multiplication", &status);//������ �������������� ���
			if (!kernel || status != CL_SUCCESS)
			{
				throw "Error: Failed to create compute kernel!\n";
			}
		}
		else if (numberOfRealization == 2) {
			//kernel = clCreateKernel(program, "matricesMul", &status);//������ �������������� ���
			kernel = clCreateKernel(program, "myGEMM3", &status);//������ �������������� ���
			if (!kernel || status != CL_SUCCESS)
			{
				throw "Error: Failed to create compute kernel!\n";
			}
		}
		else if (numberOfRealization == 3) {
			kernel = clCreateKernel(program, "matricesMulVector", &status);//������ �������������� ���
			if (!kernel || status != CL_SUCCESS)
			{
				throw "Error: Failed to create compute kernel!\n";
			}
		}
		else if (numberOfRealization == 4) {
			kernel = clCreateKernel(program, "matricesMul", &status);//������ �������������� ���
			if (!kernel || status != CL_SUCCESS)
			{
				throw "Error: Failed to create compute kernel!\n";
			}
		}





		KernelInfo(kernel, deviceID);

		if (numberOfRealization == 1) {
			lection4_multipl_matrix(context, status, queue, kernel, matrix1, matrix2, resultMatrix, NKM, deviceID);
		}
		else if (numberOfRealization == 2) {
			//multipl_matrix_local_memory(context, status, queue, kernel, matrix1, matrix2, resultMatrix, NKM);
			multipl_matrix_local_memory_v4(context, status, queue, kernel, matrix1, matrix2, resultMatrix, NKM);
		}
		else if (numberOfRealization == 3) {
			multipl_matrix_vector(context, status, queue, kernel, matrix1, matrix2, resultMatrix, NKM);
		}
		else if (numberOfRealization == 4) {
			multipl_matrix_local_memory(context, status, queue, kernel, matrix1, matrix2, resultMatrix, NKM);
		}
		else {
			throw "Incorrect number of realization";
		}


		if (numberOfRealization == 2) {
			write_matrix_to_file_v2();
		}
		else {
			write_matrix_to_file();
		}
		

		free_openCL();


		free(matrix1);
		free(matrix2);
		free(resultMatrix);
		free(buf);


	}
	catch (const char* msg)
	{
		std::cout << msg << std::endl;
		if (buf != NULL)
			delete [] buf;
		if(matrix1!=NULL)
		free(matrix1);
		if(matrix2!=NULL)
		free(matrix2);
		if(resultMatrix!=NULL)
		free(resultMatrix);

		free_openCL();

		return 1;
	}
	catch (...)
	{
		std::cout << "Unknown error" << std::endl;
		if (buf != NULL)
			delete [] buf;
		if (matrix1 != NULL)
			free(matrix1);
		if (matrix2 != NULL)
			free(matrix2);
		if (resultMatrix != NULL)
			free(resultMatrix);

		free_openCL();
		
		return 1;
	}

	return 0;
}







//������ ���������� � �������������� ���������� ������ ��� ���������
void lection4_multipl_matrix(cl_context context, cl_int status, cl_command_queue queue, cl_kernel kernel,
	float*& matrix1, float*& matrix2, float*& resultMatrix, int* NKM, cl_device_id deviceID)
{

	//float a[6] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }; //(3;2)
	//float b[6] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };//(2;3)
	//float* c = (float*)malloc(sizeof(float) * 9);

	double start_time, end_time;

	auto matrix1Rows = NKM[2];
	auto matrix1Columns = NKM[1];
	auto matrix2Rows = NKM[1];
	auto matrix2Columns = NKM[0];

	auto matrix1ElementsCount = matrix1Rows * matrix1Columns;
	auto matrix2ElementsCount = matrix2Rows * matrix2Columns;
	auto resultMatrixCapacity = matrix1Rows * matrix2Columns;


	start_time = omp_get_wtime();//������ ������� �� ������ �������� ������ � ����� �� ������

	/// <summary>
	/// ������ ��������� �� �������, ������� �������� ������ � ����� �� ������ ����������� ���
	/// � ������� clEnqueueWriteBuffer
	/// </summary>
	arg_buffer_a = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * matrix1ElementsCount, NULL, &status);
	//clReleaseMemObject

	status = clEnqueueWriteBuffer(queue, arg_buffer_a, CL_FALSE, 0, sizeof(float) * matrix1ElementsCount,
		matrix1, 0, NULL, NULL);

	arg_buffer_b = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * matrix2ElementsCount, NULL, &status);
	//clReleaseMemObject

	status = clEnqueueWriteBuffer(queue, arg_buffer_b, CL_FALSE, 0, sizeof(float) * matrix2ElementsCount,
		matrix2, 0, NULL, NULL);

	arg_buffer_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * resultMatrixCapacity, NULL, &status);
	//clReleaseMemObject

	status = clEnqueueWriteBuffer(queue, arg_buffer_c, CL_FALSE, 0, sizeof(float) * resultMatrixCapacity,
		resultMatrix, 0, NULL, NULL);
	if (status != CL_SUCCESS)
	{
		throw "Error: Failed in clEnqueueWriteBuffer!\n";
	}

	//int wA = 2;
	//int wB = 3;


	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &arg_buffer_a);
	status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &arg_buffer_b);
	status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &arg_buffer_c);
	status |= clSetKernelArg(kernel, 3, sizeof(int), &matrix1Columns);
	status |= clSetKernelArg(kernel, 4, sizeof(int), &matrix2Columns);
	if (status != CL_SUCCESS)
	{
		throw "Error: Failed in clSetKernelArg!\n";
	}


	size_t dimentions = 2;
	size_t global_work_size[2];
	global_work_size[0] = matrix2Columns;
	global_work_size[1] = matrix1Rows;

	//size_t local_work_size[2];//������������ ��������
	//global_work_size[0] = (size_t)4;
	//global_work_size[1] = (size_t)4;

	cl_event ourEvent = 0;


	status = clEnqueueNDRangeKernel(queue, kernel, dimentions, NULL, global_work_size, NULL, 0,
		NULL, &ourEvent);
	if (status != CL_SUCCESS)
	{
		throw "Error: Failed in clEnqueueNDRangeKernel!\n";
	}

	/// <summary>
	/// � ������ ������� �� �������� ������ � ������� �� ����, ������� ����� ���� ������� �� �����������
	/// ������� ������ ������� ����������
	/// </summary>
	status = clEnqueueReadBuffer(queue, arg_buffer_c, CL_TRUE, 0,
		sizeof(int) * resultMatrixCapacity, resultMatrix, 0, NULL, NULL);//����� ��������� ReadBuffer ������ ���� ����������(CL_TRUE)
	if (status != CL_SUCCESS)
	{
		throw "Error: Failed in clEnqueueReadBuffer!\n";
	}

	end_time = omp_get_wtime();
	auto timeSingle = end_time - start_time;

	cl_ulong gstart, gend;

	status = clGetEventProfilingInfo(ourEvent, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &gstart, NULL);
	status = clGetEventProfilingInfo(ourEvent, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &gend, NULL);

	double nanoSeconds = gend - gstart;
	printf("\nTime: %f\t%f \n", nanoSeconds / 1000000.0, timeSingle * 1000);

}

// ������ ���������� � �������������� ��������� ������
void multipl_matrix_local_memory(cl_context context, cl_int status, cl_command_queue queue, cl_kernel kernel,
	float*& matrix1, float*& matrix2, float*& resultMatrix, int* NKM)
{

	double start_time, end_time;

	auto matrix1Rows = NKM[2];
	auto matrix1Columns = NKM[1];
	auto matrix2Rows = NKM[1];
	auto matrix2Columns = NKM[0];

	auto matrix1ElementsCount = matrix1Rows * matrix1Columns;
	auto matrix2ElementsCount = matrix2Rows * matrix2Columns;
	auto resultMatrixCapacity = matrix1Rows * matrix2Columns;

	start_time = omp_get_wtime();//������ ������� �� ������ �������� ������ � ����� �� ������

	/// <summary>
	/// ������ ��������� �� �������, ������� �������� ������ � ����� �� ������ ����������� ���
	/// � ������� clEnqueueWriteBuffer
	/// </summary>
	arg_buffer_a = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * matrix1ElementsCount, NULL, &status);

	status = clEnqueueWriteBuffer(queue, arg_buffer_a, CL_FALSE, 0, sizeof(float) * matrix1ElementsCount,
		matrix1, 0, NULL, NULL);

	arg_buffer_b = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * matrix2ElementsCount, NULL, &status);

	status = clEnqueueWriteBuffer(queue, arg_buffer_b, CL_FALSE, 0, sizeof(float) * matrix2ElementsCount,
		matrix2, 0, NULL, NULL);

	arg_buffer_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * resultMatrixCapacity, NULL, &status);

	status = clEnqueueWriteBuffer(queue, arg_buffer_c, CL_FALSE, 0, sizeof(float) * resultMatrixCapacity,
		resultMatrix, 0, NULL, NULL);
	if (status != CL_SUCCESS)
	{
		throw "Error: Failed in clEnqueueWriteBuffer!\n";
	}

	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &arg_buffer_a);
	status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &arg_buffer_b);
	status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &arg_buffer_c);
	//status |= clSetKernelArg(kernel, 3, sizeof(float) * matrix1Columns, NULL);//matrix1Rows ����� �� ���������
	status |= clSetKernelArg(kernel, 3, sizeof(int), &matrix1Columns);
	status |= clSetKernelArg(kernel, 4, sizeof(int), &matrix2Columns);//� ������� �� ������������������� ������� ��� �������� ��������
	if (status != CL_SUCCESS)
	{
		throw "Error: Failed in clSetKernelArg!\n";
	}


	size_t dimentions = 1;
	size_t global_work_size[1];
	global_work_size[0] = matrix1Rows;

	size_t local_work_size[1];
	local_work_size[0] = 250;//250 �������� ��-��������� ��� ������ 500 �� 500, ������� ��� � ������ 250

	cl_event ourEvent = 0;

	status = clEnqueueNDRangeKernel(queue, kernel, dimentions, NULL, global_work_size, NULL, 0,
		NULL, &ourEvent);
	if (status != CL_SUCCESS)
	{
		throw "Error: Failed in clEnqueueNDRangeKernel!\n";
	}

	/// <summary>
	/// � ������ ������� �� �������� ������ � ������� �� ����, ������� ����� ���� ������� �� �����������
	/// ������� ������ ������� ����������
	/// </summary>
	status = clEnqueueReadBuffer(queue, arg_buffer_c, CL_TRUE, 0,
		sizeof(int) * resultMatrixCapacity, resultMatrix, 0, NULL, NULL);//����� ��������� ReadBuffer ������ ���� ����������(CL_TRUE)
	if (status != CL_SUCCESS)
	{
		throw "Error: Failed in clEnqueueReadBuffer!\n";
	}

	end_time = omp_get_wtime();
	auto timeSingle = end_time - start_time;

	cl_ulong gstart, gend;

	status = clGetEventProfilingInfo(ourEvent, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &gstart, NULL);
	status = clGetEventProfilingInfo(ourEvent, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &gend, NULL);

	double nanoSeconds = gend - gstart;
	printf("\nTime: %f\t%f \n", nanoSeconds / 1000000.0, timeSingle * 1000);

}

//��������� ���������� � �������������� ��������� ������
void multipl_matrix_local_memory_v4(cl_context context, cl_int status, cl_command_queue queue, cl_kernel kernel,
	float*& matrix1, float*& matrix2, float*& resultMatrix, int* NKM)
{

	double start_time, end_time;

	auto matrix1Rows = NKM[2];
	auto matrix1Columns = NKM[1];
	auto matrix2Rows = NKM[1];
	auto matrix2Columns = NKM[0];

	auto matrix1ElementsCount = matrix1Rows * matrix1Columns;
	auto matrix2ElementsCount = matrix2Rows * matrix2Columns;
	auto resultMatrixCapacity = matrix1Rows * matrix2Columns;

	start_time = omp_get_wtime();//������ ������� �� ������ �������� ������ � ����� �� ������

	/// <summary>
	/// ������ ��������� �� �������, ������� �������� ������ � ����� �� ������ ����������� ���
	/// � ������� clEnqueueWriteBuffer
	/// </summary>
	arg_buffer_a = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * matrix1ElementsCount, NULL, &status);

	status = clEnqueueWriteBuffer(queue, arg_buffer_a, CL_FALSE, 0, sizeof(float) * matrix1ElementsCount,
		matrix1, 0, NULL, NULL);

	arg_buffer_b = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * matrix2ElementsCount, NULL, &status);

	status = clEnqueueWriteBuffer(queue, arg_buffer_b, CL_FALSE, 0, sizeof(float) * matrix2ElementsCount,
		matrix2, 0, NULL, NULL);

	arg_buffer_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * resultMatrixCapacity, NULL, &status);

	status = clEnqueueWriteBuffer(queue, arg_buffer_c, CL_FALSE, 0, sizeof(float) * resultMatrixCapacity,
		resultMatrix, 0, NULL, NULL);
	if (status != CL_SUCCESS)
	{
		throw "Error: Failed in clEnqueueWriteBuffer!\n";
	}

	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &arg_buffer_a);
	status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &arg_buffer_b);
	status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &arg_buffer_c);
	status |= clSetKernelArg(kernel, 3, sizeof(int), &matrix1Rows);//M
	status |= clSetKernelArg(kernel, 4, sizeof(int), &matrix2Columns);//N
	status |= clSetKernelArg(kernel, 5, sizeof(int), &matrix1Columns);//K
	if (status != CL_SUCCESS)
	{
		throw "Error: Failed in clSetKernelArg!\n";
	}


	size_t dimentions = 2;
	size_t global_work_size[2];
	//kernel2
	//global_work_size[0] = matrix1Rows;
	//global_work_size[1] = matrix2Columns;

	global_work_size[0] = matrix1Rows;
	global_work_size[1] = matrix2Columns / WPT;

	//kernel3

	size_t local_work_size[2];
	//kernel2
	//local_work_size[0] = localWorkSize;
	//local_work_size[1] = localWorkSize;

	//kernel3
	local_work_size[0] = localWorkSize;
	local_work_size[1] = localWorkSize / WPT;

	cl_event ourEvent = 0;

	status = clEnqueueNDRangeKernel(queue, kernel, dimentions, NULL, global_work_size, local_work_size, 0,
		NULL, &ourEvent);
	if (status != CL_SUCCESS)
	{
		throw "Error: Failed in clEnqueueNDRangeKernel!\n";
	}

	/// <summary>
	/// � ������ ������� �� �������� ������ � ������� �� ����, ������� ����� ���� ������� �� �����������
	/// ������� ������ ������� ����������
	/// </summary>
	status = clEnqueueReadBuffer(queue, arg_buffer_c, CL_TRUE, 0,
		sizeof(int) * resultMatrixCapacity, resultMatrix, 0, NULL, NULL);//����� ��������� ReadBuffer ������ ���� ����������(CL_TRUE)
	if (status != CL_SUCCESS)
	{
		throw "Error: Failed in clEnqueueReadBuffer!\n";
	}

	end_time = omp_get_wtime();
	auto timeSingle = end_time - start_time;

	cl_ulong gstart, gend;

	status = clGetEventProfilingInfo(ourEvent, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &gstart, NULL);
	status = clGetEventProfilingInfo(ourEvent, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &gend, NULL);

	double nanoSeconds = gend - gstart;
	printf("\nTime: %f\t%f \n", nanoSeconds / 1000000.0, timeSingle * 1000);


	/*for (size_t i = 0; i < resultMatrixCapacity; i++)
	{
		printf("\nmatrix[%d] = %f", i, resultMatrix[i]);
	}*/

}



//������ ��������� ����������(���������� ������ �� ������� �� ���������, �������� �������� 4 
void multipl_matrix_vector(cl_context context, cl_int status, cl_command_queue queue, cl_kernel kernel,
	float*& matrix1, float*& matrix2, float*& resultMatrix, int* NKM)
{

	double start_time, end_time;

	auto matrix1Rows = NKM[2];
	auto matrix1Columns = NKM[1];
	auto matrix2Rows = NKM[1];
	auto matrix2Columns = NKM[0];

	auto matrix1ElementsCount = matrix1Rows * matrix1Columns;
	auto matrix2ElementsCount = matrix2Rows * matrix2Columns;
	auto resultMatrixCapacity = matrix1Rows * matrix2Columns;

	start_time = omp_get_wtime();//������ ������� �� ������ �������� ������ � ����� �� ������

	/// <summary>
	/// ������ ��������� �� �������, ������� �������� ������ � ����� �� ������ ����������� ���
	/// � ������� clEnqueueWriteBuffer
	/// </summary>
	arg_buffer_a = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * matrix1ElementsCount, NULL, &status);

	status = clEnqueueWriteBuffer(queue, arg_buffer_a, CL_FALSE, 0, sizeof(float) * matrix1ElementsCount,
		matrix1, 0, NULL, NULL);

	arg_buffer_b = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * matrix2ElementsCount, NULL, &status);

	status = clEnqueueWriteBuffer(queue, arg_buffer_b, CL_FALSE, 0, sizeof(float) * matrix2ElementsCount,
		matrix2, 0, NULL, NULL);

	arg_buffer_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * resultMatrixCapacity, NULL, &status);

	status = clEnqueueWriteBuffer(queue, arg_buffer_c, CL_FALSE, 0, sizeof(float) * resultMatrixCapacity,
		resultMatrix, 0, NULL, NULL);
	if (status != CL_SUCCESS)
	{
		throw "Error: Failed in clEnqueueWriteBuffer!\n";
	}

	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &arg_buffer_a);
	status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &arg_buffer_b);
	status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &arg_buffer_c);
	//status |= clSetKernelArg(kernel, 3, sizeof(float) * matrix1Columns, NULL);//matrix1Rows ����� �� ���������
	status |= clSetKernelArg(kernel, 3, sizeof(int), &matrix1Columns);
	status |= clSetKernelArg(kernel, 4, sizeof(int), &matrix2Columns);//� ������� �� ������������������� ������� ��� �������� ��������
	if (status != CL_SUCCESS)
	{
		throw "Error: Failed in clSetKernelArg!\n";
	}


	size_t dimentions = 1;
	size_t global_work_size[1];
	global_work_size[0] = matrix1Rows;

	size_t local_work_size[1];
	local_work_size[0] = 250;//250 �������� ��-��������� ��� ������ 500 �� 500, ������� ��� � ������ 250

	cl_event ourEvent = 0;

	status = clEnqueueNDRangeKernel(queue, kernel, dimentions, NULL, global_work_size, NULL, 0,
		NULL, &ourEvent);
	if (status != CL_SUCCESS)
	{
		throw "Error: Failed in clEnqueueNDRangeKernel!\n";
	}

	/// <summary>
	/// � ������ ������� �� �������� ������ � ������� �� ����, ������� ����� ���� ������� �� �����������
	/// ������� ������ ������� ����������
	/// </summary>
	status = clEnqueueReadBuffer(queue, arg_buffer_c, CL_TRUE, 0,
		sizeof(int) * resultMatrixCapacity, resultMatrix, 0, NULL, NULL);//����� ��������� ReadBuffer ������ ���� ����������(CL_TRUE)
	if (status != CL_SUCCESS)
	{
		throw "Error: Failed in clEnqueueReadBuffer!\n";
	}

	end_time = omp_get_wtime();
	auto timeSingle = end_time - start_time;

	cl_ulong gstart, gend;

	status = clGetEventProfilingInfo(ourEvent, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &gstart, NULL);
	status = clGetEventProfilingInfo(ourEvent, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &gend, NULL);

	double nanoSeconds = gend - gstart;
	printf("\nTime: %f\t%f \n", nanoSeconds / 1000000.0, timeSingle * 1000);

}












// ����� ������������

void get_matrixs_from_file_v2(string input_file_path, int NKM[], float*& matrix1, float*& matrix2, float*& resultMatrix, int multiplicity)
{
	int multiplicity1 = 4;
	float** table1FromFile = NULL;
	float** table2FromFile = NULL;

	float** table1WithAddedElements = NULL;
	float** table2WithAddedElements = NULL;

	char* bufIterator = NULL;
	char* buf = NULL;

	ifstream in(input_file_path, ios::binary);
	int size = in.seekg(0, ios::end).tellg();
	if (size == -1)
		throw "File is empty";
	in.seekg(0);
	buf = new char[size + 1];
	in.read(buf, size);
	buf[size] = 0;
	bufIterator = buf;
	in.close();
	string tempString = "";

	while (true) {


		if (*bufIterator == ' ' || *bufIterator == 13) {


			if (NKM[0] == 0) {
				NKM[0] = stoi(tempString);
				tempString = "";
			}
			else
				if (NKM[1] == 0) {
					NKM[1] = stoi(tempString);
					tempString = "";
				}
				else
					if (NKM[2] == 0) {
						NKM[2] = stoi(tempString);
						tempString = "";
					}

			if (*bufIterator == ' ') {
				bufIterator++;
			}
			else {
				bufIterator++;
				bufIterator++;
				break;
			}

		}
		else {
			tempString += *bufIterator;
			bufIterator++;
		}

	}

	NKMBase[2] = NKM[2];
	NKMBase[1] = NKM[1];
	NKMBase[0] = NKM[0];


	auto matrix1RowsFromFile = NKM[2];//M
	auto matrix1ColumnsFromFile = NKM[1];//K
	auto matrix2RowsFromFile = NKM[1];//K
	auto matrix2ColumnsFromFile = NKM[0];//N

	auto matrix1RowsWithNulls = NKM[2];//M
	auto matrix1ColumnsWithNulls = NKM[1];//K
	auto matrix2RowsWithNulls = NKM[1];//K
	auto matrix2ColumnsWithNulls = NKM[0];//N


	int matrix1RowsAddedNumber = 0;
	int matrix1ColumnsAddedNumber = 0;
	int matrix2RowsAddedNumber = 0;
	int matrix2ColumnsAddedNumber = 0;


	if ((matrix1RowsFromFile % multiplicity) != 0) {
		matrix1RowsAddedNumber = multiplicity - (matrix1RowsFromFile % multiplicity);
		matrix1RowsWithNulls += matrix1RowsAddedNumber;
	}

	if ((matrix1ColumnsFromFile % multiplicity) != 0) {
		matrix1ColumnsAddedNumber = multiplicity - (matrix1ColumnsWithNulls % multiplicity);
		matrix1ColumnsWithNulls += matrix1ColumnsAddedNumber;
	}

	if ((matrix2RowsFromFile % multiplicity) != 0) {
		matrix2RowsAddedNumber = multiplicity - (matrix2RowsWithNulls % multiplicity);
		matrix2RowsWithNulls += matrix2RowsAddedNumber;
	}

	if ((matrix2ColumnsFromFile % multiplicity) != 0) {
		matrix2ColumnsAddedNumber = multiplicity - (matrix2ColumnsWithNulls % multiplicity);
		matrix2ColumnsWithNulls += matrix2ColumnsAddedNumber;
	}


	if (matrix1ColumnsWithNulls != matrix2RowsWithNulls) {
		throw "Impossible! The collsrows don't match!";
	}










	table1FromFile = (float**)calloc(matrix1RowsFromFile, sizeof(float*));

	for (int i = 0; i < matrix1RowsFromFile; i++)
	{
		table1FromFile[i] = (float*)calloc(matrix1ColumnsFromFile, sizeof(float));//table[i] - ��� ��� ��������� �� ������� ������ ��� ��������

		int j = 0;
		while (j != matrix1ColumnsFromFile)
		{
			if ((int)*bufIterator != 32 && (int)*bufIterator != 13 && (int)*bufIterator != 10 && *bufIterator != '\0')
			{
				tempString += *bufIterator;
				bufIterator++;

			}
			else
			{
				if (tempString == "")
				{
					throw "Wrong number exception";
				}
				table1FromFile[i][j] = stod(tempString);
				j += 1;
				bufIterator++;
				tempString = "";
			}
			if (j == matrix1ColumnsFromFile)
			{
				bufIterator++;
			}

		}
	}


	//for (size_t i = 0; i < matrix1RowsFromFile; i++)
	//{
	//	for (size_t j = 0; j < matrix1ColumnsFromFile; j++)
	//	{
	//		printf("\nmatrix[%d][%d] = %f", i, j, table1FromFile[i][j]);
	//	}
	//}



	table2FromFile = (float**)calloc(matrix2RowsFromFile, sizeof(float*));

	for (int i = 0; i < matrix2RowsFromFile; i++)
	{
		table2FromFile[i] = (float*)calloc(matrix2ColumnsFromFile, sizeof(float));//table[i] - ��� ��� ��������� �� ������� ������ ��� ��������

		int j = 0;
		while (j != matrix2ColumnsFromFile)
		{
			if ((int)*bufIterator != 32 && (int)*bufIterator != 13 && (int)*bufIterator != 10 && *bufIterator != '\0')
			{
				tempString += *bufIterator;
				bufIterator++;

			}
			else
			{
				if (tempString == "")
				{
					throw "Wrong number exception";
				}
				table2FromFile[i][j] = stod(tempString);
				j += 1;
				bufIterator++;
				tempString = "";
			}
			if (j == matrix2ColumnsFromFile)
			{
				bufIterator++;
			}

		}
	}


	//for (size_t i = 0; i < matrix2RowsFromFile; i++)
	//{
	//	for (size_t j = 0; j < matrix2ColumnsFromFile; j++)
	//	{
	//		printf("\nmatrix[%d][%d] = %f", i, j, table2FromFile[i][j]);
	//	}
	//}



	table1WithAddedElements = (float**)calloc(matrix1RowsWithNulls, sizeof(float*));

	for (size_t i = 0; i < matrix1RowsWithNulls; i++)
	{
		table1WithAddedElements[i] = (float*)calloc(matrix1ColumnsWithNulls, sizeof(float));
	}



	table2WithAddedElements = (float**)calloc(matrix2RowsWithNulls, sizeof(float*));

	for (size_t i = 0; i < matrix2RowsWithNulls; i++)
	{
		table2WithAddedElements[i] = (float*)calloc(matrix2ColumnsWithNulls, sizeof(float));
	}




	for (size_t i = 0; i < matrix1RowsFromFile; i++)
	{
		for (size_t j = 0; j < matrix1ColumnsFromFile; j++)
		{
			table1WithAddedElements[i][j] = table1FromFile[i][j];
		}

	}

	/*for (size_t i = 0; i < matrix1RowsWithNulls; i++)
	{
		for (size_t j = 0; j < matrix1ColumnsWithNulls; j++)
		{
			printf("\nmatrix[%d][%d] = %f", i, j, table1WithAddedElements[i][j]);
		}
	}*/

	for (size_t i = 0; i < matrix2RowsFromFile; i++)
	{
		for (size_t j = 0; j < matrix2ColumnsFromFile; j++)
		{
			table2WithAddedElements[i][j] = table2FromFile[i][j];
		}

	}


	//for (size_t i = 0; i < matrix2RowsWithNulls; i++)
	//{
	//	for (size_t j = 0; j < matrix2ColumnsWithNulls; j++)
	//	{
	//		printf("\nmatrix[%d][%d] = %f", i, j, table2WithAddedElements[i][j]);
	//	}
	//}


	auto matrix1ElementsCountWithAdded = matrix1RowsWithNulls * matrix1ColumnsWithNulls;
	auto matrix2ElementsCountWithAdded = matrix2RowsWithNulls * matrix2ColumnsWithNulls;

	matrix1 = (float*)calloc(matrix1ElementsCountWithAdded, sizeof(float));
	matrix2 = (float*)calloc(matrix2ElementsCountWithAdded, sizeof(float));

	int increment = 0;

	for (size_t j = 0; j < matrix1ColumnsWithNulls; j++)
	{
		for (size_t i = 0; i < matrix1RowsWithNulls; i++)
		{
			matrix1[increment] = table1WithAddedElements[i][j];
			increment++;
		}
	}

	/*for (size_t i = 0; i < matrix1ElementsCountWithAdded; i++)
	{
		printf("\nmatrix1[%d] = %f", i, matrix1[i]);
	}*/

	increment = 0;

	for (size_t j = 0; j < matrix2ColumnsWithNulls; j++)
	{
		for (size_t i = 0; i < matrix2RowsWithNulls; i++)
		{
			matrix2[increment] = table2WithAddedElements[i][j];
			increment++;
		}
	}

	/*for (size_t i = 0; i < matrix2ElementsCountWithAdded; i++)
	{
		printf("\nmatrix1[%d] = %f", i, matrix2[i]);
	}*/

	NKM[2] = matrix1RowsWithNulls;
	NKM[1] = matrix1ColumnsWithNulls;
	NKM[0] = matrix2ColumnsWithNulls;


	int resultMatrixCapacity = matrix1RowsWithNulls * matrix2ColumnsWithNulls;

	resultMatrix = (float*)calloc(resultMatrixCapacity, sizeof(float));

	free(buf);

	for (size_t i = 0; i < matrix1RowsFromFile; i++)
	{
		free(table1FromFile[i]);
	}
	free(table1FromFile);

	for (size_t i = 0; i < matrix2RowsFromFile; i++)
	{
		free(table2FromFile[i]);
	}
	free(table2FromFile);

	for (size_t i = 0; i < matrix1RowsWithNulls; i++)
	{
		free(table1WithAddedElements[i]);
	}
	free(table1WithAddedElements);

	for (size_t i = 0; i < matrix2RowsWithNulls; i++)
	{
		free(table2WithAddedElements[i]);
	}
	free(table2WithAddedElements);


}

void get_matrixs_transpose_from_file(string input_file_path, int NKM[], float*& matrix1, float*& matrix2, float*& resultMatrix) {

	char* bufIterator = NULL;
	char* buf = NULL;
	float** matrix2Temp;
	float** matrix2TempTranspose;

	ifstream in(input_file_path, ios::binary);
	int size = in.seekg(0, ios::end).tellg();
	if (size == -1)
		throw "File is empty";
	in.seekg(0);
	buf = new char[size + 1];
	in.read(buf, size);
	buf[size] = 0;
	bufIterator = buf;
	in.close();
	string tempString = "";

	while (true) {


		if (*bufIterator == ' ' || *bufIterator == 13) {


			if (NKM[0] == 0) {
				NKM[0] = stoi(tempString);
				tempString = "";
			}
			else
				if (NKM[1] == 0) {
					NKM[1] = stoi(tempString);
					tempString = "";
				}
				else
					if (NKM[2] == 0) {
						NKM[2] = stoi(tempString);
						tempString = "";
					}

			if (*bufIterator == ' ') {
				bufIterator++;
			}
			else {
				bufIterator++;
				bufIterator++;
				break;
			}

		}
		else {
			tempString += *bufIterator;
			bufIterator++;
		}

	}

	auto matrix1Rows = NKM[2];
	auto matrix1Columns = NKM[1];
	auto matrix2Rows = NKM[1];
	auto matrix2Columns = NKM[0];

	auto matrix1ElementsCount = matrix1Rows * matrix1Columns;
	auto matrix2ElementsCount = matrix2Rows * matrix2Columns;

	matrix1 = (float*)calloc(matrix1ElementsCount, sizeof(float));
	matrix2 = (float*)calloc(matrix2ElementsCount, sizeof(float));
	matrix2Temp = (float**)calloc(matrix2Rows, sizeof(float*));

	int i = 0;
	while (i != matrix1ElementsCount) {

		if ((int)*bufIterator != 32 && (int)*bufIterator != 13 && (int)*bufIterator != 10 && *bufIterator != '\0')
		{
			tempString += *bufIterator;
			bufIterator++;

		}
		else
		{
			if (tempString == "")
			{
				throw "Wrong number exception";
			}
			matrix1[i] = stod(tempString);
			i++;
			bufIterator++;
			tempString = "";
		}
		if ((int)*bufIterator == 10)
		{
			bufIterator++;
		}

	}

	for (int i = 0; i < matrix2Rows; i++)
	{
		matrix2Temp[i] = (float*)calloc(matrix2Columns, sizeof(float));

		int j = 0;
		while (j != matrix2Columns)
		{
			if ((int)*bufIterator != 32 && (int)*bufIterator != 13 && (int)*bufIterator != 10 && *bufIterator != '\0')
			{
				tempString += *bufIterator;
				bufIterator++;

			}
			else
			{
				if (tempString == "")
				{
					throw "Wrong number exception";
				}
				matrix2Temp[i][j] = stod(tempString);
				j += 1;
				bufIterator++;
				tempString = "";
			}
			if (j == matrix2Columns)
			{
				bufIterator++;
			}

		}
	}

	matrix2TempTranspose = (float**)calloc(matrix2Columns, sizeof(float*));

	for (size_t i = 0; i < matrix2Columns; i++)
	{
		matrix2TempTranspose[i] = (float*)calloc(matrix2Rows, sizeof(float));
	}


	float t = 0.0;
	for (int i = 0; i < matrix2Rows; i++)
	{
		for (int j = 0; j < matrix2Columns; j++)
		{
			matrix2TempTranspose[j][i] = matrix2Temp[i][j];
		}
	}

	int iterator = 0;
	for (size_t i = 0; i < matrix2Columns; i++)
	{
		for (size_t j = 0; j < matrix2Rows; j++)
		{
			matrix2[iterator] = matrix2TempTranspose[i][j];
			iterator++;
		}
	}


	int resultMatrixCapacity = matrix1Rows * matrix2Columns;

	resultMatrix = (float*)calloc(resultMatrixCapacity, sizeof(float));
}

void get_matrixs_transpose_from_file_for_vector(string input_file_path, int NKM[], float*& matrix1, float*& matrix2, float*& resultMatrix) {

	char* bufIterator = NULL;
	char* buf = NULL;
	float** matrix2Temp;
	float** matrix2TempTranspose;

	ifstream in(input_file_path, ios::binary);
	int size = in.seekg(0, ios::end).tellg();
	if (size == -1)
		throw "File is empty";
	in.seekg(0);
	buf = new char[size + 1];
	in.read(buf, size);
	buf[size] = 0;
	bufIterator = buf;
	in.close();
	string tempString = "";

	while (true) {


		if (*bufIterator == ' ' || *bufIterator == 13) {


			if (NKM[0] == 0) {
				NKM[0] = stoi(tempString);
				tempString = "";
			}
			else
				if (NKM[1] == 0) {
					NKM[1] = stoi(tempString);
					tempString = "";
				}
				else
					if (NKM[2] == 0) {
						NKM[2] = stoi(tempString);
						tempString = "";
					}

			if (*bufIterator == ' ') {
				bufIterator++;
			}
			else {
				bufIterator++;
				bufIterator++;
				break;
			}

		}
		else {
			tempString += *bufIterator;
			bufIterator++;
		}

	}

	auto matrix1Rows = NKM[2];
	auto matrix1Columns = NKM[1];
	auto matrix2Rows = NKM[1];
	auto matrix2Columns = NKM[0];

	int res = matrix1Columns % 4;
	if (0 == matrix1Columns % 4) {

		auto matrix1ElementsCount = matrix1Rows * matrix1Columns;
		auto matrix2ElementsCount = matrix2Rows * matrix2Columns;

		matrix1 = (float*)calloc(matrix1ElementsCount, sizeof(float));
		matrix2 = (float*)calloc(matrix2ElementsCount, sizeof(float));
		matrix2Temp = (float**)calloc(matrix2Rows, sizeof(float*));

		int i = 0;
		while (i != matrix1ElementsCount) {

			if ((int)*bufIterator != 32 && (int)*bufIterator != 13 && (int)*bufIterator != 10 && *bufIterator != '\0')
			{
				tempString += *bufIterator;
				bufIterator++;

			}
			else
			{
				if (tempString == "")
				{
					throw "Wrong number exception";
				}
				matrix1[i] = stod(tempString);
				i++;
				bufIterator++;
				tempString = "";
			}
			if ((int)*bufIterator == 10)
			{
				bufIterator++;
			}

		}

		for (int i = 0; i < matrix2Rows; i++)
		{
			matrix2Temp[i] = (float*)calloc(matrix2Columns, sizeof(float));

			int j = 0;
			while (j != matrix2Columns)
			{
				if ((int)*bufIterator != 32 && (int)*bufIterator != 13 && (int)*bufIterator != 10 && *bufIterator != '\0')
				{
					tempString += *bufIterator;
					bufIterator++;

				}
				else
				{
					if (tempString == "")
					{
						throw "Wrong number exception";
					}
					matrix2Temp[i][j] = stod(tempString);
					j += 1;
					bufIterator++;
					tempString = "";
				}
				if (j == matrix2Columns)
				{
					bufIterator++;
				}

			}
		}

		matrix2TempTranspose = (float**)calloc(matrix2Columns, sizeof(float*));

		for (size_t i = 0; i < matrix2Columns; i++)
		{
			matrix2TempTranspose[i] = (float*)calloc(matrix2Rows, sizeof(float));
		}


		float t = 0.0;
		for (int i = 0; i < matrix2Rows; i++)
		{
			for (int j = 0; j < matrix2Columns; j++)
			{
				matrix2TempTranspose[j][i] = matrix2Temp[i][j];
			}
		}

		int iterator = 0;
		for (size_t i = 0; i < matrix2Columns; i++)
		{
			for (size_t j = 0; j < matrix2Rows; j++)
			{
				matrix2[iterator] = matrix2TempTranspose[i][j];
				iterator++;
			}
		}


		int resultMatrixCapacity = matrix1Rows * matrix2Columns;

		resultMatrix = (float*)calloc(resultMatrixCapacity, sizeof(float));
	}
	else {
		matrix1Columns += (4 - res);
		matrix2Rows += (4 - res);

		auto matrix1ElementsCount = matrix1Rows * matrix1Columns;
		auto matrix2ElementsCount = matrix2Rows * matrix2Columns;

		matrix1 = (float*)calloc(matrix1ElementsCount, sizeof(float));
		matrix2 = (float*)calloc(matrix2ElementsCount, sizeof(float));
		matrix2Temp = (float**)calloc(matrix2Rows, sizeof(float*));

		int i = 0;
		while (i != matrix1ElementsCount) {

			if ((int)*bufIterator != 32 && (int)*bufIterator != 13 && (int)*bufIterator != 10 && *bufIterator != '\0')
			{
				tempString += *bufIterator;
				bufIterator++;

			}
			else
			{
				if (tempString == "")
				{
					throw "Wrong number exception";
				}
				matrix1[i] = stod(tempString);
				i++;
				bufIterator++;
				tempString = "";
			}
			if ((int)*bufIterator == 10)
			{
				bufIterator++;
			}

		}

		for (int i = 0; i < matrix2Rows; i++)
		{
			matrix2Temp[i] = (float*)calloc(matrix2Columns, sizeof(float));

			int j = 0;
			while (j != matrix2Columns)
			{
				if ((int)*bufIterator != 32 && (int)*bufIterator != 13 && (int)*bufIterator != 10 && *bufIterator != '\0')
				{
					tempString += *bufIterator;
					bufIterator++;

				}
				else
				{
					if (tempString == "")
					{
						throw "Wrong number exception";
					}
					matrix2Temp[i][j] = stod(tempString);
					j += 1;
					bufIterator++;
					tempString = "";
				}
				if (j == matrix2Columns)
				{
					bufIterator++;
				}

			}
		}

		matrix2TempTranspose = (float**)calloc(matrix2Columns, sizeof(float*));

		for (size_t i = 0; i < matrix2Columns; i++)
		{
			matrix2TempTranspose[i] = (float*)calloc(matrix2Rows, sizeof(float));
		}


		float t = 0.0;
		for (int i = 0; i < matrix2Rows; i++)
		{
			for (int j = 0; j < matrix2Columns; j++)
			{
				matrix2TempTranspose[j][i] = matrix2Temp[i][j];
			}
		}

		int iterator = 0;
		for (size_t i = 0; i < matrix2Columns; i++)
		{
			for (size_t j = 0; j < matrix2Rows; j++)
			{
				matrix2[iterator] = matrix2TempTranspose[i][j];
				iterator++;
			}
		}


		int resultMatrixCapacity = matrix1Rows * matrix2Columns;

		resultMatrix = (float*)calloc(resultMatrixCapacity, sizeof(float));
	}

}

void free_openCL() {

	if(arg_buffer_a!=NULL)
	clReleaseMemObject(arg_buffer_a);
	if (arg_buffer_b != NULL)
	clReleaseMemObject(arg_buffer_b);
	if (arg_buffer_c != NULL)
	clReleaseMemObject(arg_buffer_c);
	if (kernel != NULL)
	clReleaseKernel(kernel);
	if (program != NULL)
	clReleaseProgram(program);
	if (queue!= NULL)
	clReleaseCommandQueue(queue);
	if (context != NULL)
	clReleaseContext(context);

}

void get_kernel_code_from_file() {

	ifstream in("Program.txt", ios::binary);
	sizeBuf = in.seekg(0, ios::end).tellg();
	if (sizeBuf == -1)
		throw "File is empty";
	in.seekg(0);
	buf = new char[sizeBuf + 1];
	in.read(buf, sizeBuf);
	buf[sizeBuf] = 0;
	in.close();
	buf_p = buf;
}

void write_matrix_to_file() {

	// WRITE RESULT MATRIX TO FILE


	auto matrix1Rows = NKM[2];
	auto matrix1Columns = NKM[1];
	auto matrix2Rows = NKM[1];
	auto matrix2Columns = NKM[0];

	vector<char> outputData;

	string tmp = to_string(matrix2Columns);
	char const* N = tmp.c_str();

	string tmp1 = to_string(matrix1Rows);
	char const* M = tmp1.c_str();

	outputData.insert(outputData.end(), N, N + strlen(N));
	outputData.push_back(' ');
	outputData.insert(outputData.end(), M, M + strlen(M));
	outputData.push_back('\r');
	outputData.push_back('\n');

	int increment = 0;
	for (size_t i = 0; i < matrix1Rows; i++)//������ �����
	{
		for (size_t j = 0; j < matrix2Columns; j++)
		{
			char* char_arr;
			string str_obj(to_string(resultMatrix[increment]));
			char_arr = &str_obj[0];


			outputData.insert(outputData.end(), char_arr, char_arr + strlen(char_arr));
			outputData.push_back(' ');

			increment++;
		}
		outputData.pop_back();
		outputData.push_back('\r');
		outputData.push_back('\n');

	}
	outputData.pop_back();
	outputData.pop_back();

	char* outputArray = &outputData[0];

	if (numberOfRealization == 1) {
		fstream bin(pathOutputFile/*"C:\\Users\\black\\Desktop\\matrixResult.txt"*/, ios::out | ios::binary);
		bin.write(outputArray, sizeof(char) * outputData.size());
		bin.close();
	}
	else if (numberOfRealization == 2) {
		fstream bin(pathOutputFile/*"C:\\Users\\black\\Desktop\\matrixResult1.txt"*/, ios::out | ios::binary);
		bin.write(outputArray, sizeof(char) * outputData.size());
		bin.close();
	}
	else if (numberOfRealization == 3) {
		fstream bin(pathOutputFile/*"C:\\Users\\black\\Desktop\\matrixResult2.txt"*/, ios::out | ios::binary);
		bin.write(outputArray, sizeof(char) * outputData.size());
		bin.close();
	}
	else if (numberOfRealization == 4) {
		fstream bin(pathOutputFile/*"C:\\Users\\black\\Desktop\\matrixResult2.txt"*/, ios::out | ios::binary);
		bin.write(outputArray, sizeof(char) * outputData.size());
		bin.close();
	}

	//free(outputArray);

}

void write_matrix_to_file_v2() {

	// WRITE RESULT MATRIX TO FILE


	auto matrix1Rows = NKMBase[2];
	auto matrix1Columns = NKMBase[1];
	auto matrix2Rows = NKMBase[1];
	auto matrix2Columns = NKMBase[0];

	vector<char> outputData;

	string tmp = to_string(matrix2Columns);
	char const* N = tmp.c_str();

	string tmp1 = to_string(matrix1Rows);
	char const* M = tmp1.c_str();

	outputData.insert(outputData.end(), N, N + strlen(N));
	outputData.push_back(' ');
	outputData.insert(outputData.end(), M, M + strlen(M));
	outputData.push_back('\r');
	outputData.push_back('\n');


	for (size_t i = 0; i < matrix1Rows; i++)//������ �����
	{
		for (size_t j = 0; j < matrix2Columns; j++)
		{//NKM[2] - ����������, ����� ������� �� ������������ ����� ������� �� ������� ���������� ��������

			char* char_arr;
			string str_obj(to_string(resultMatrix[j * NKM[2] + i]));
			char_arr = &str_obj[0];

			outputData.insert(outputData.end(), char_arr, char_arr + strlen(char_arr));
			outputData.push_back(' ');



		}//C[globalCol * M + globalRow]
		outputData.pop_back();
		outputData.push_back('\r');
		outputData.push_back('\n');

	}
	outputData.pop_back();
	outputData.pop_back();

	char* outputArray = &outputData[0];

	if (numberOfRealization == 1) {
		fstream bin(pathOutputFile/*"C:\\Users\\black\\Desktop\\matrixResult.txt"*/, ios::out | ios::binary);
		bin.write(outputArray, sizeof(char) * outputData.size());
		bin.close();
	}
	else if (numberOfRealization == 2) {
		fstream bin(pathOutputFile/*"C:\\Users\\black\\Desktop\\matrixResult1.txt"*/, ios::out | ios::binary);
		bin.write(outputArray, sizeof(char) * outputData.size());
		bin.close();
	}
	else if (numberOfRealization == 3) {
		fstream bin(pathOutputFile/*"C:\\Users\\black\\Desktop\\matrixResult2.txt"*/, ios::out | ios::binary);
		bin.write(outputArray, sizeof(char) * outputData.size());
		bin.close();
	}

	//free(outputArray);

}

void get_matrixs_from_file(string input_file_path, int NKM[], float*& matrix1, float*& matrix2, float*& resultMatrix)  {

	char* bufIterator = NULL;
	char* buf = NULL;

	ifstream in(input_file_path, ios::binary);
	int size = in.seekg(0, ios::end).tellg();
	if (size == -1)
		throw "File is empty";
	in.seekg(0);
	buf = new char[size + 1];
	in.read(buf, size);
	buf[size] = 0;
	bufIterator = buf;
	in.close();
	string tempString = "";

	while (true) {


		if (*bufIterator == ' ' || *bufIterator == 13) {


			if (NKM[0] == 0) {
				NKM[0] = stoi(tempString);
				tempString = "";
			}
			else
				if (NKM[1] == 0) {
					NKM[1] = stoi(tempString);
					tempString = "";
				}
				else
					if (NKM[2] == 0) {
						NKM[2] = stoi(tempString);
						tempString = "";
					}

			if (*bufIterator == ' ') {
				bufIterator++;
			}
			else {
				bufIterator++;
				bufIterator++;
				break;
			}

		}
		else {
			tempString += *bufIterator;
			bufIterator++;
		}

	}

	auto matrix1Rows = NKM[2];
	auto matrix1Columns = NKM[1];
	auto matrix2Rows = NKM[1];
	auto matrix2Columns = NKM[0];

	auto matrix1ElementsCount = matrix1Rows * matrix1Columns;
	auto matrix2ElementsCount = matrix2Rows * matrix2Columns;

	matrix1 = (float*)calloc(matrix1ElementsCount, sizeof(float));
	matrix2 = (float*)calloc(matrix2ElementsCount, sizeof(float));

	int i = 0;
	while (i != matrix1ElementsCount) {

		if ((int)*bufIterator != 32 && (int)*bufIterator != 13 && (int)*bufIterator != 10 && *bufIterator != '\0')
		{
			tempString += *bufIterator;
			bufIterator++;

		}
		else
		{
			if (tempString == "")
			{
				throw "Wrong number exception";
			}
			matrix1[i] = stod(tempString);
			i++;
			bufIterator++;
			tempString = "";
		}
		if ((int)*bufIterator == 10)
		{
			bufIterator++;
		}

	}


	/*for (size_t i = 0; i < matrix1ElementsCount; i++)
	{
		cout << endl;
		printf("matrix1[%i] = %f", i, matrix1[i]);
	}

	cout << endl;*/

	i = 0;
	while (i != matrix2ElementsCount) {

		if ((int)*bufIterator != 32 && (int)*bufIterator != 13 && (int)*bufIterator != 10 && *bufIterator != '\0')
		{
			tempString += *bufIterator;
			bufIterator++;

		}
		else
		{
			if (tempString == "")
			{
				throw "Wrong number exception";
			}
			matrix2[i] = stod(tempString);
			i++;
			bufIterator++;
			tempString = "";
		}
		if ((int)*bufIterator == 10)
		{
			bufIterator++;
		}

	}


	/*for (size_t i = 0; i < matrix2ElementsCount; i++)
	{
		cout << endl;
		printf("matrix2[%i] = %f", i, matrix2[i]);
	}*/

	int resultMatrixCapacity = matrix1Rows * matrix2Columns;

	resultMatrix = (float*)calloc(resultMatrixCapacity, sizeof(float));

	delete[] buf;
}

cl_device_id InformationAboutDevice(cl_platform_id* platformID, int numberOfDevice)
{
	cl_uint platformCount;
	int err = clGetPlatformIDs(0, NULL, &platformCount);//gets number of available platforms
	//printf("\nNumber of platforms - %i\n", platformCount);
	cl_platform_id* platforms = (cl_platform_id*)malloc(platformCount);
	clGetPlatformIDs(platformCount, platforms, NULL);//gets platform ids


	cl_uint numberOfDevices;
	cl_device_id* devices = NULL;

	vector<cl_device_id> devicesDiscreteGPU;
	vector<cl_device_id> devicesIntegratedGPU;
	vector<cl_device_id> devicesCPU;
	vector<cl_device_id> allDevicesIDs;

	int er = CL_INVALID_PLATFORM;

	const char* attributeNames[5] = { "CPU", "GPU", "ACCELERATOR", "DEFAULT", "ALL" };
	const cl_platform_info attributeTypes[5] = {
												CL_DEVICE_TYPE_CPU,
												CL_DEVICE_TYPE_GPU,
												CL_DEVICE_TYPE_ACCELERATOR,
												CL_DEVICE_TYPE_DEFAULT,
												CL_DEVICE_TYPE_ALL };

	cl_bool res = false;
	cl_uint  numberOfUnits = 0;
	size_t paramValueRet = 0;

	for (int i = 0; i < platformCount; i++)
	{
		//����� � ���������� GPU-���������, �������������� OpenCL
		err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, NULL, &numberOfDevices);

		if (numberOfDevices != 0 || err == 0) {

			devices = (cl_device_id*)malloc(numberOfDevices);
			err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, numberOfDevices, devices, NULL);
			if (err != CL_SUCCESS)
			{
				throw "Error: Failed wile getting device Id!\n";
			}

			for (size_t j = 0; j < numberOfDevices; j++)//�������� ���������� ���������� ��� ��� ���������������
			{
				err = clGetDeviceInfo(devices[j], CL_DEVICE_HOST_UNIFIED_MEMORY, 0, NULL, &paramValueRet);
				err = clGetDeviceInfo(devices[j], CL_DEVICE_HOST_UNIFIED_MEMORY, paramValueRet, &res, NULL);
				if (err != CL_SUCCESS)
				{
					throw "Error: Failed wile getting device info!\n";
				}
				if (res == false)
				{
					devicesDiscreteGPU.push_back(devices[j]);
				}
				else {
					devicesIntegratedGPU.push_back(devices[j]);
				}
			}
		}

		numberOfDevices = 0;

		//�������� ������� ��������� OpenCL � CPU
		err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_CPU, 0, NULL, &numberOfDevices);

		if (numberOfDevices != 0 || err == 0) {

			devices = (cl_device_id*)malloc(numberOfDevices);
			err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_CPU, numberOfDevices, devices, NULL);

			for (size_t j = 0; j < numberOfDevices; j++)
			{
				devicesCPU.push_back(devices[j]);
			}
		}
	}

	allDevicesIDs.insert(allDevicesIDs.end(), devicesDiscreteGPU.begin(), devicesDiscreteGPU.end());
	allDevicesIDs.insert(allDevicesIDs.end(), devicesIntegratedGPU.begin(), devicesIntegratedGPU.end());
	allDevicesIDs.insert(allDevicesIDs.end(), devicesCPU.begin(), devicesCPU.end());


	if (numberOfDevice > allDevicesIDs.size()) {
		auto id = allDevicesIDs[0];

		clGetDeviceInfo(id, CL_DEVICE_PLATFORM, 0, NULL, &paramValueRet);
		clGetDeviceInfo(id, CL_DEVICE_PLATFORM, paramValueRet, platformID, NULL);

		return id;
	}
	else {

		auto id = allDevicesIDs[numberOfDevice];

		clGetDeviceInfo(id, CL_DEVICE_PLATFORM, 0, NULL, &paramValueRet);
		clGetDeviceInfo(id, CL_DEVICE_PLATFORM, paramValueRet, platformID, NULL);

		return id;
	}
}

void DeviceInfo(cl_device_id deviceID) {//����� ���� � ������� 

	/////////////////////////////////////DEVICE INFO/////////////////////////////////////

	//display chosen device name
	size_t deviceNameSize;
	cl_int status = 0;
	status = clGetDeviceInfo(deviceID, CL_DEVICE_NAME, 0, NULL, &deviceNameSize);
	char* deviceName = (char*)malloc(deviceNameSize);
	status = clGetDeviceInfo(deviceID, CL_DEVICE_NAME, deviceNameSize, deviceName, NULL);
	printf("\nDevice name - %s\n", deviceName);

	//kernel numbers
	size_t kernelsNumberSize;
	status = clGetDeviceInfo(deviceID, CL_DEVICE_BUILT_IN_KERNELS, 0, NULL, &kernelsNumberSize);
	char* kernels = (char*)malloc(kernelsNumberSize);
	status = clGetDeviceInfo(deviceID, CL_DEVICE_BUILT_IN_KERNELS, kernelsNumberSize, kernels, NULL);
	//printf("\nCL_DEVICE_BUILT_IN_KERNELS - %s\n", kernels);

	//global cache size 
	cl_ulong globalCacheSize;
	status = clGetDeviceInfo(deviceID, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(globalCacheSize), &globalCacheSize, NULL);
	//printf("\nCL_DEVICE_GLOBAL_MEM_CACHE_SIZE - %llu bytes\n", (unsigned long long)globalCacheSize);

	//global memory size 
	cl_ulong globalMemSize;
	status = clGetDeviceInfo(deviceID, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(globalMemSize), &globalMemSize, NULL);
	//printf("\nCL_DEVICE_GLOBAL_MEM_SIZE - %llu bytes\n", (unsigned long long)globalMemSize);

	//local memory size 
	cl_ulong localMemSize;
	status = clGetDeviceInfo(deviceID, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(localMemSize), &localMemSize, NULL);
	//printf("\nCL_DEVICE_LOCAL_MEM_SIZE - %llu bytes\n", (unsigned long long)localMemSize);

	//CL_DEVICE_MAX_COMPUTE_UNITS 
	cl_uint numberOfComputeUnits;//���������� ���������� ����� ����������
	status = clGetDeviceInfo(deviceID, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(numberOfComputeUnits), &numberOfComputeUnits, NULL);
	//printf("\nCL_DEVICE_MAX_COMPUTE_UNITS - %u units\n", (unsigned int)numberOfComputeUnits);

	//CL_DEVICE_MAX_WORK_GROUP_SIZE
	size_t maxWorkGroupElementsCount;
	status = clGetDeviceInfo(deviceID, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWorkGroupElementsCount), &maxWorkGroupElementsCount, NULL);
	//printf("\nCL_DEVICE_MAX_WORK_GROUP_SIZE - %u units\n", (unsigned int)maxWorkGroupElementsCount);

	//CL_DEVICE_MAX_WORK_ITEM_SIZES
	size_t maxWorkItemSize[3];
	status = clGetDeviceInfo(deviceID, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(maxWorkItemSize), maxWorkItemSize, NULL);
	//printf("\nCL_DEVICE_MAX_WORK_ITEM_SIZES - %d:%d:%d\n", maxWorkItemSize[0], maxWorkItemSize[1], maxWorkItemSize[2]);

	free(deviceName);
	free(kernels);

	/////////////////////////////////////END DEVICE INFO/////////////////////////////////////
}

void KernelInfo(cl_kernel kernel, cl_device_id deviceID) {

	/////////////////////////////////////KERNEL INFO/////////////////////////////////////

	//kernel numbers
	size_t kernelWorkGroupSize;
	cl_int status = 0;
	status = clGetKernelWorkGroupInfo(kernel, deviceID, CL_KERNEL_WORK_GROUP_SIZE, sizeof(kernelWorkGroupSize)
		, &kernelWorkGroupSize, NULL);
	//printf("\nCL_KERNEL_WORK_GROUP_SIZE - %u\n", (unsigned int)kernelWorkGroupSize);

	//kernel local memory size
	cl_ulong kernelLocalMemSize;
	status = clGetKernelWorkGroupInfo(kernel, deviceID, CL_KERNEL_LOCAL_MEM_SIZE, sizeof(kernelLocalMemSize)
		, &kernelLocalMemSize, NULL);
	//printf("\nCL_KERNEL_LOCAL_MEM_SIZE - %llu\n", (unsigned long long)kernelLocalMemSize);

	//kernel numbers
	size_t kernelPreferredWorkSize;
	status = clGetKernelWorkGroupInfo(kernel, deviceID, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
		sizeof(kernelPreferredWorkSize), &kernelPreferredWorkSize, NULL);
	//printf("\nCL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE - %u\n", (unsigned int)kernelPreferredWorkSize);

	//kernel private memory size
	cl_ulong kernelPrivateMemSize;
	status = clGetKernelWorkGroupInfo(kernel, deviceID, CL_KERNEL_PRIVATE_MEM_SIZE, sizeof(kernelPrivateMemSize)
		, &kernelPrivateMemSize, NULL);
	//printf("\nCL_KERNEL_PRIVATE_MEM_SIZE - %llu\n", (unsigned long long)kernelPrivateMemSize);

	/////////////////////////////////////END KERNEL INFO/////////////////////////////////////
}
