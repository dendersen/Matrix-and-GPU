#pragma once
struct Matrix_int {
	int width;
	int height;
	int* data;
};
struct Matrix_float {
	int width;
	int height;
	float* data;
};
uint8_t HIP_AVAILABLE;
struct Matrix_float; 
struct Matrix_int;
typedef struct Matrix_int Matrix_I_t;
typedef struct Matrix_float Matrix_F_t;
static Matrix_I_t* generateMatrix_I (int width, int height, bool zero);
static Matrix_F_t* generateMatrix_F (int width, int height, bool zero);
static void deleteMatrix_I (Matrix_I_t* in, bool zero);
static void deleteMatrix_F (Matrix_F_t* in, bool zero);
static float MatrixRead_F (Matrix_F_t in, int x, int y);
static int MatrixWrite_I (Matrix_I_t in, int x, int y, int data);
static int MatrixWrite_F (Matrix_F_t in, int x, int y, int data);
static int MatrixRead_I (Matrix_I_t in, int x, int y);
static Matrix_I_t* MatrixAdd_I (Matrix_I_t* a, Matrix_I_t* b);
static Matrix_F_t* MatrixAdd_F (Matrix_F_t* a, Matrix_F_t* b);
static void MatrixPrint_I (Matrix_I_t* in);