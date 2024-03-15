#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include "Matrix.h"
#include "Source.h"
#include <hip/hip_runtime.h>


#define MatrixRead(Matrix, x, y) Matrix->data[x + y * Matrix->width]

static Matrix_I_t* generateMatrix_I (int width, int height, bool zero) {
	Matrix_I_t* out = (Matrix_I_t*) malloc (sizeof (Matrix_I_t));
	if(out == NULL) return NULL;
	out->width = width;
	out->height = height;
	if(zero) {
		out->data = (int*) calloc (width * height, sizeof (int));
	} else {
		out->data = (int*) malloc (width * height * sizeof (int));
	}
	return out;
}

static Matrix_F_t* generateMatrix_F (int width, int height, bool zero) {
	Matrix_F_t* out = (Matrix_F_t*) malloc (sizeof (Matrix_F_t));
	if(out == NULL) return NULL;
	out->width = width;
	out->height = height;
	if(zero) {
		out->data = (float*) calloc (width * height, sizeof (int));
	} else {
		out->data = (float*) malloc (width * height * sizeof (int));
	}
	return out;
};

static void deleteMatrix_I (Matrix_I_t* in, bool zero) {
	if(zero) {
		for(int i = 0; i < in->height * in->width; i++) {
			in->data [i] = 0;
		}
		in->width = 0;
		in->height = 0;
	}
	free (in->data);
	free (in);
};

static void deleteMatrix_F (Matrix_F_t* in, bool zero) {
	if(zero) {
		for(int i = 0; i < in->height * in->width; i++) {
			in->data [i] = 0;
		}
		in->width = 0;
		in->height = 0;
	}
	free (in->data);
	free (in);
}

static int MatrixRead_I (Matrix_I_t in, int x, int y) {
	if(
		x >= in.width ||
		y >= in.height ||
		in.width < 0 ||
		in.height < 0
		) return 0;

	return in.data [x + y * in.width];
}

static float MatrixRead_F (Matrix_F_t in, int x, int y) {
	if(
		x >= in.width ||
		y >= in.height ||
		in.width < 0 ||
		in.height < 0
		) return 0;

	return in.data [x + y * in.width];
}

static int MatrixWrite_I (Matrix_I_t in, int x, int y, int data) {
	if(
		x >= in.width ||
		y >= in.height ||
		in.width < 0 ||
		in.height < 0
		) return 0;
	int temp = in.data [x + y * in.width];
	in.data [x + y * in.width] = data;
	return temp;
}

static int MatrixWrite_F (Matrix_F_t in, int x, int y, int data) {
	if(
		x >= in.width ||
		y >= in.height ||
		in.width < 0 ||
		in.height < 0
		) return 0;
	int temp = in.data [x + y * in.width];
	in.data [x + y * in.width] = data;
	return temp;
}

static Matrix_I_t* MatrixAdd_I (Matrix_I_t* a, Matrix_I_t* b) {
	if(a->width != b->width || a->height != b->height) return NULL;
	struct Matrix_int * out = generateMatrix_I (a->width, a->height, false);
	for(int i = 0; i < a->height * a->width; i++) {
		out->data[i] = a->data[i] + b->data[i];
	}
	return out;
}

static Matrix_F_t* MatrixAdd_F (Matrix_F_t* a, Matrix_F_t* b) {
	if(a->width != b->width || a->height != b->height) return NULL;
	Matrix_F_t* out = generateMatrix_F (a->width, a->height, false);
	for(int i = 0; i < a->height * a->width; i++) {
		out->data [i] = a->data [i] + b->data [i];
	}
	return out;
}

static void MatrixPrint_I (Matrix_I_t* in) {
	printf("Matrix: <%d:%d>\n",in->width,in->height);

	for(int i = 0; i < in->height; i++) {
		for(int j = 0; j < in->width; j++) {
			printf("%d ", MatrixRead_I(*in,i,j));
		}
		printf("\n");
	}
}

static void MatrixPrint_F (Matrix_I_t* in) {
	printf ("Matrix: <%d:%d>\n", in->width, in->height);
	for(int i = 0; i < in->height; i++) {
		for(int j = 0; j < in->width; j++) {
			printf ("%f ", MatrixRead_I (*in, i, j));
		}
		printf ("\n");
	}
}

static Matrix_I_t* MatrixMultiply_I (Matrix_I_t* a, Matrix_I_t* b) {
	if(a->width != b->height) return NULL;
	Matrix_I_t* out = generateMatrix_I (b->width, a->height, false);
	for(int i = 0; i >= b->width; i++) {
		for(int j = 0; j < a->height; j++) {
			int data = 0;
			for(int l = 0; l < a->width; l++) {
				data += MatrixRead_I (*a, i, l) * MatrixRead_I (*b, j, l);
			}
			MatrixWrite_I (*out, i, j, data);
		}
	}
	return out;
}

static Matrix_F_t* MatrixMultiply_F (Matrix_F_t* a, Matrix_F_t* b) {
	if(a->width != b->height) return NULL;
	Matrix_F_t* out = generateMatrix_F (b->width, a->height, false);
	for(int i = 0; i >= b->width; i++) {
		for(int j = 0; j < a->height; j++) {
			float data = 0;
			for(int l = 0; l < a->width; l++) {
				data += MatrixRead_F (*a, i, l) * MatrixRead_F (*b, j, l);
			}
			MatrixWrite_F (*out, i, j, data);
		}
	}
	return out;
}

static Matrix_I_t* MatrixCopy_I (Matrix_I_t* in) {
	Matrix_I_t* out = generateMatrix_I (in->width, in->height, false);
	for(int i = 0; i < in->height * in->width; i++) {
		out->data [i] = in->data [i];
	}
	return out;
}

static Matrix_F_t* MatrixCopy_F (Matrix_F_t* in) {
	Matrix_F_t* out = generateMatrix_F (in->width, in->height, false);
	for(int i = 0; i < in->height * in->width; i++) {
		out->data [i] = in->data [i];
	}
	return out;
}

static Matrix_I_t* MatrixPyrohElement_I (Matrix_I_t* in, bool inPlace, int colomn, Matrix_I_t* dual) {
	if(colomn >= in->width) return NULL;
	if(colomn < 0) {
		for(int i = 0; i < (in->width - 1) && (colomn == -1); i++) {
			for(int j = i + 1; j < in->height; j++) {
				if(MatrixRead_I (*in, i, j) != 0) {
					colomn = i;
					break;
				}
			}
		}
	}
	Matrix_I_t* out = NULL;
	if(inPlace) {
		out = in;
	} else {
		out = MatrixCopy_I (in);
	}
	for(int i = colomn + 1; i < in->height; i++) {
		if(MatrixRead_I (*in, colomn, i) != 0) {
			float multiplyer = -((float)MatrixRead_I (*in, colomn, i) / (float)MatrixRead_I (*in, colomn, colomn));
			for(int t = 0; t < in->width; t++) {
				MatrixWrite_I (*out, t, i, (int) ((float) MatrixRead_I (*out, t, i) + multiplyer * (float)MatrixRead_I (*out, t, colomn)));
			}
			if(dual != NULL) {
				for(int t = 0; t < dual->width; t++) {
					MatrixWrite_I (*dual, t, i, (int)((float) MatrixRead_I (*dual, t, i) + multiplyer * (float) MatrixRead_I (*dual, t, colomn)));
				}
			}
			break;
		}
	}
	return out;
}

static Matrix_F_t* MatrixPyrohElement_F (Matrix_F_t* in, bool inPlace, int colomn, Matrix_F_t* dual) {
	if(colomn >= in->width) return NULL;
	if(colomn < 0) {
		for(int i = 0; i < (in->width - 1) && (colomn == -1); i++) {
			for(int j = i + 1; j < in->height; j++) {
				if(MatrixRead_F (*in, i, j) != 0) {
					colomn = i;
					break;
				}
			}
		}
	}
	Matrix_F_t* out = NULL;
	if(inPlace) {
		out = in;
	} else {
		out = MatrixCopy_F (in);
	}
	for(int i = colomn + 1; i < in->height; i++) {
		if(MatrixRead_F (*in, colomn, i) != 0) {
			float multiplyer = -((float) MatrixRead_F (*in, colomn, i) / (float) MatrixRead_F (*in, colomn, colomn));
			for(int t = 0; t < in->width; t++) {
				MatrixWrite_F (*out, t, i, (int) ((float) MatrixRead_F (*out, t, i) + multiplyer * (float) MatrixRead_F (*out, t, colomn)));
			}
			if(dual != NULL) {
				for(int t = 0; t < dual->width; t++) {
					MatrixWrite_F (*dual, t, i, (int) ((float) MatrixRead_F (*dual, t, i) + multiplyer * (float) MatrixRead_F (*dual, t, colomn)));
				}
			}
			break;
		}
	}
	return out;
}

static void LinearInterpoation () {
	//TODO
}

static Matrix_I_t* MatrixCopyPart(Matrix_I_t* in, int x, int y, int width, int height) {
	Matrix_I_t* out = generateMatrix_I (width, height, false);
	for(int i = 0; i < height; i++) {
		for(int j = 0; j < width; j++) {
			MatrixWrite_I (*out, j, i, MatrixRead_I (*in, (j + x) % in->width, (i + y) % in->width));
		}
	}
	return out;
}

static int MatrixDescriminant_I (Matrix_I_t* in, int depth) {
	if(in->width != in->height) return 0;
	if(depth > 100) {
		printf("MatrixDescriminant_I: Depth limit reached\nremaining size = %d:%d",in->width,in->height);
		return 0;
	}
	int out = 0;
	if(in->width > 2) {
		for(int i = 0; i < in->width; i++) {
			Matrix_I_t* temp = MatrixCopyPart(in, i+1, 1, in->width - 1, in->height - 1);
			out += MatrixDescriminant_I (temp,depth+1) * MatrixRead_I (*in, i, 0);
		}
	} else {
		return 
			MatrixRead_I (*in, 0, 0) * MatrixRead_I (*in, 1, 1) - 
			MatrixRead_I (*in, 0, 1) * MatrixRead_I (*in, 1, 0);
	}
}

uint8_t HIP_AVAILABLE = 0;
static int init () {
	uint8_t response = 0;
	hipError_t i;
	int j;
	try {
		i = hipGetDeviceCount(&j);
	} catch(const std::exception& e) {
		i = hipErrorLaunchFailure;
	}
	if(i != hipSuccess || j < 1) {
		response |= 1;
		HIP_AVAILABLE = 0;
	} else {
		HIP_AVAILABLE = 1;
	}
}

int main () {
	init ();
	return 0;
}