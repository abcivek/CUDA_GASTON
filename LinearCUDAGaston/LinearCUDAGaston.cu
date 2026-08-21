// CUDA - Gaston Linear Distinguisher Verifier
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "rdrand.h"      
#include <Windows.h>
#include <math.h>
#include <ctime>
#include <stdlib.h>

#define bit64 unsigned __int64
#define BLOCKS 1024
#define THREADS 1024

#define CUDA_CHECK(x) \
do { \
    cudaError_t err = x; \
    if (err != cudaSuccess) { \
        printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

// ***********************  timing *********************** //
double PCFreq = 0.0;
__int64 CounterStart = 0;

void StartCounter() { LARGE_INTEGER li; if (!QueryPerformanceFrequency(&li)) printf("QueryPerformanceFrequency failed!\n"); PCFreq = double(li.QuadPart) / 1000.0; QueryPerformanceCounter(&li); CounterStart = li.QuadPart; }
double GetCounter() { LARGE_INTEGER li; QueryPerformanceCounter(&li); return double(li.QuadPart - CounterStart) / PCFreq; }

// *********************** helper functions *********************** //
void print_state(bit64 state[5]) { for (int i = 0; i < 5; i++) printf("%016I64x\n", state[i]); }

// *********************** GPU functions *********************** //
__device__ __forceinline__ bit64 rotl(bit64 x, int l) {
	l &= 63;
	return (x << l) | (x >> ((64 - l) & 63));
}

__global__ void gpu_gaston_linear_chi(bit64 IV[], bit64 key[], bit64 nonce[], __int64 counter[], int round, __int64 trial) {
	int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
	int tIdx2 = 2 * tIdx;
	int tIdx3 = tIdx2 + 1;

	uint64_t gaston_rc[12] = {
    0xF0ULL, 0xE1ULL, 0xD2ULL,
    0xC3ULL, 0xB4ULL, 0xA5ULL,
    0x96ULL, 0x87ULL, 0x78ULL,
    0x69ULL, 0x5AULL, 0x4BULL
	};

	bit64 xor1, xor2, temp;

	bit64 initial0, initial1, initial2, initial3, initial4;
	bit64 P, Q;

	initial0 = IV[tIdx];
	initial1 = key[tIdx2];
	initial2 = key[tIdx3];
	initial3 = nonce[tIdx2];
	initial4 = nonce[tIdx3];

	for (__int64 c = 0; c < trial; c++) {
		// masked plaintexts:
		xor1 = initial0 & 0x0000000800000008 ^
			   initial1 & 0x1000000400000000 ^
			   initial2 & 0x1000000400000000 ^
			   initial3 & 0x0000000000000008 ^
			   initial4 & 0x0010000080000008;

		/* ---- chi ---- */   // -> first round
		P = initial0;
		Q = initial1;
		initial0 ^= (initial2 & ~initial1);
		initial1 ^= (initial3 & ~initial2);
		initial2 ^= (initial4 & ~initial3);
		initial3 ^= (P & ~initial4);
		initial4 ^= (Q & ~P);
		///  ----------------------------------

		for (int i = 0; i < round-1; i++) { // remaining rouds
			/* ---- rho-east ---- */
			initial0 = rotl(initial0, 0); //e0 =0
			initial1 = rotl(initial1, 60); //e1 =60
			initial2 = rotl(initial2, 22); //e2 =22
			initial3 = rotl(initial3, 27); //e3 =27
			initial4 = rotl(initial4, 4); //e4 =4

			/* ---- theta ---- */
			P = initial0 ^ initial1 ^ initial2 ^ initial3 ^ initial4;
			P ^= rotl(P, 1); // GASTON_r = 1

			Q = rotl(initial0, 25)
				^ rotl(initial1, 32)
				^ rotl(initial2, 52)
				^ rotl(initial3, 60)
				^ rotl(initial4, 63);
			// t0=25, t1=32, t2=52, t3=60,t4=63

			Q ^= rotl(Q, 18); //GASTON_s=18

			P ^= Q;
			P = rotl(P, 23); //GASTON_u = 23

			initial0 ^= P;
			initial1 ^= P;
			initial2 ^= P;
			initial3 ^= P;
			initial4 ^= P;

			/* ---- rho-west ---- */
			initial0 = rotl(initial0, 0); //w0=0
			initial1 = rotl(initial1, 56); //w1=56
			initial2 = rotl(initial2, 31); //w2=31
			initial3 = rotl(initial3, 46); //w3=46
			initial4 = rotl(initial4, 43); //w4=43

			/* ---- iota ---- */
			initial0 ^= gaston_rc[i+1];

			/* ---- chi ---- */
			P = initial0;
			Q = initial1;
			initial0 ^= (initial2 & ~initial1);
			initial1 ^= (initial3 & ~initial2);
			initial2 ^= (initial4 & ~initial3);
			initial3 ^= (P & ~initial4);
			initial4 ^= (Q & ~P);
		}

		// masked output:
		xor2 = initial0 & 0x0000000000000000 ^
			   initial1 & 0x0001000000400000 ^
			   initial2 & 0x0000000000400000 ^
			   initial3 & 0x0000000000001000 ^
			   initial4 & 0x0000000800004000;
 

		temp = xor1 ^ xor2;
		// parity calc:
		temp ^= temp >> 1;
		temp ^= temp >> 2;
		temp = (temp & 0x1111111111111111UL) * 0x1111111111111111UL;
		temp = (temp >> 60) & 1;

		if (temp == 0) counter[tIdx]++;
	}
}

__global__ void gpu_gaston_linear_east(bit64 IV[], bit64 key[], bit64 nonce[], __int64 counter[], int round, __int64 trial) {
	int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
	int tIdx2 = 2 * tIdx;
	int tIdx3 = tIdx2 + 1;

	uint64_t gaston_rc[12] = {
	0xF0ULL, 0xE1ULL, 0xD2ULL,
	0xC3ULL, 0xB4ULL, 0xA5ULL,
	0x96ULL, 0x87ULL, 0x78ULL,
	0x69ULL, 0x5AULL, 0x4BULL
	};

	bit64 xor1, xor2, temp;

	bit64 initial0, initial1, initial2, initial3, initial4;
	bit64 P, Q;

	initial0 = IV[tIdx];
	initial1 = key[tIdx2];
	initial2 = key[tIdx3];
	initial3 = nonce[tIdx2];
	initial4 = nonce[tIdx3];

	for (__int64 c = 0; c < trial; c++) {
		// masked plaintexts:
		xor1 = initial0 & 0x0000000800000000 ^
			   initial1 & 0x1000000400000000 ^
			   initial2 & 0x0000000000000000 ^
			   initial3 & 0x0000000000000008 ^
			   initial4 & 0x0010000080000000;

		///  ----------------------------------

		for (int i = 0; i < round; i++) { 
			/* ---- rho-east ---- */
			initial0 = rotl(initial0, 0); //e0 =0
			initial1 = rotl(initial1, 60); //e1 =60
			initial2 = rotl(initial2, 22); //e2 =22
			initial3 = rotl(initial3, 27); //e3 =27
			initial4 = rotl(initial4, 4); //e4 =4

			/* ---- theta ---- */
			P = initial0 ^ initial1 ^ initial2 ^ initial3 ^ initial4;
			P ^= rotl(P, 1); // GASTON_r = 1

			Q = rotl(initial0, 25)
				^ rotl(initial1, 32)
				^ rotl(initial2, 52)
				^ rotl(initial3, 60)
				^ rotl(initial4, 63);
			// t0=25, t1=32, t2=52, t3=60,t4=63

			Q ^= rotl(Q, 18); //GASTON_s=18

			P ^= Q;
			P = rotl(P, 23); //GASTON_u = 23

			initial0 ^= P;
			initial1 ^= P;
			initial2 ^= P;
			initial3 ^= P;
			initial4 ^= P;

			/* ---- rho-west ---- */
			initial0 = rotl(initial0, 0); //w0=0
			initial1 = rotl(initial1, 56); //w1=56
			initial2 = rotl(initial2, 31); //w2=31
			initial3 = rotl(initial3, 46); //w3=46
			initial4 = rotl(initial4, 43); //w4=43

			/* ---- iota ---- */
			initial0 ^= gaston_rc[i + 1];

			/* ---- chi ---- */
			P = initial0;
			Q = initial1;
			initial0 ^= (initial2 & ~initial1);
			initial1 ^= (initial3 & ~initial2);
			initial2 ^= (initial4 & ~initial3);
			initial3 ^= (P & ~initial4);
			initial4 ^= (Q & ~P);
		}

		// masked output:
		xor2 = initial0 & 0x0000000000000000 ^
			   initial1 & 0x0001000000400000 ^
			   initial2 & 0x0000000000400000 ^
			   initial3 & 0x0000000000001000 ^
			   initial4 & 0x0000000800004000;


		temp = xor1 ^ xor2;
		// parity calc:
		temp ^= temp >> 1;
		temp ^= temp >> 2;
		temp = (temp & 0x1111111111111111UL) * 0x1111111111111111UL;
		temp = (temp >> 60) & 1;

		if (temp == 0) counter[tIdx]++;
	}
}

// #######################  GASTON LINEER EXPERIMENT - MAIN ######################

long double gaston_main_chi(__int64 experiment, int round,  __int64 trial) {

	bit64* key, * key_d, * nonce, * nonce_d, * IV, * IV_d;
	__int64* counter = 0, * counter_d, bias;
	unsigned long long total_counter = 0; // instead of int64
	float milliseconds = 0;

	nonce = (bit64*)calloc(BLOCKS * THREADS * 2, sizeof(bit64));
	key = (bit64*)calloc(BLOCKS * THREADS * 2, sizeof(bit64));
	IV = (bit64*)calloc(BLOCKS * THREADS, sizeof(bit64)); //
	counter = (__int64*)calloc(BLOCKS * THREADS, sizeof(__int64));

	cudaMalloc((void**)&nonce_d, BLOCKS * THREADS * 2 * sizeof(bit64));
	cudaMalloc((void**)&key_d, BLOCKS * THREADS * 2 * sizeof(bit64));
	cudaMalloc((void**)&IV_d, BLOCKS * THREADS * sizeof(bit64));
	cudaMalloc((void**)&counter_d, BLOCKS * THREADS * sizeof(bit64));

	cudaEvent_t start, stop;
	for (int j = 0; j < THREADS * BLOCKS * 2; j++) {
		rdrand_64(nonce + j, 0);
		rdrand_64(key + j, 0);
	}

	for (int j = 0; j < THREADS * BLOCKS ; j++) {
		rdrand_64(IV + j, 0);
	}
	cudaMemcpy(counter_d, counter, BLOCKS * THREADS * sizeof(__int64), cudaMemcpyHostToDevice);
	cudaMemcpy(nonce_d, nonce, BLOCKS * THREADS * 2 * sizeof(bit64), cudaMemcpyHostToDevice);
	cudaMemcpy(key_d, key, BLOCKS * THREADS * 2 * sizeof(bit64), cudaMemcpyHostToDevice);
	cudaMemcpy(IV_d, IV, BLOCKS * THREADS * sizeof(bit64), cudaMemcpyHostToDevice); //

	StartCounter();
	cudaDeviceSynchronize();
	cudaEventCreate(&start);	cudaEventCreate(&stop);	cudaEventRecord(start);
	gpu_gaston_linear_chi << <BLOCKS, THREADS >> > (IV_d, key_d, nonce_d, counter_d, round, trial);
	cudaEventRecord(stop);	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);	printf("Time elapsed: %f milliseconds ", milliseconds);	printf("Time of kernel: %lf\n", GetCounter());

	cudaMemcpy(counter, counter_d, BLOCKS * THREADS * sizeof(__int64), cudaMemcpyDeviceToHost);
	for (int i = 0; i < BLOCKS * THREADS; i++) total_counter += counter[i];
	bias = (experiment) / 2 - total_counter;
	printf("Total counter: %I64d, Difference: %I64d, Bias: 2^-%lf\n\n", total_counter, bias, ((log(BLOCKS) + log(THREADS) + log(trial)) / log(2)) - (log(abs(bias)) / log(2)));

	cudaFree(key_d); cudaFree(nonce_d); cudaFree(counter_d); cudaFree(IV_d);
	free(key); free(nonce); free(counter); free(IV);

	return (long double)bias;

}

long double gaston_main_east(__int64 experiment, int round, __int64 trial) {

	bit64* key, * key_d, * nonce, * nonce_d, * IV, * IV_d;
	__int64* counter = 0, * counter_d, bias;
	unsigned long long total_counter = 0; // instead of int64
	float milliseconds = 0;

	nonce = (bit64*)calloc(BLOCKS * THREADS * 2, sizeof(bit64));
	key = (bit64*)calloc(BLOCKS * THREADS * 2, sizeof(bit64));
	IV = (bit64*)calloc(BLOCKS * THREADS, sizeof(bit64)); //
	counter = (__int64*)calloc(BLOCKS * THREADS, sizeof(__int64));

	cudaMalloc((void**)&nonce_d, BLOCKS * THREADS * 2 * sizeof(bit64));
	cudaMalloc((void**)&key_d, BLOCKS * THREADS * 2 * sizeof(bit64));
	cudaMalloc((void**)&IV_d, BLOCKS * THREADS * sizeof(bit64));
	cudaMalloc((void**)&counter_d, BLOCKS * THREADS * sizeof(bit64));

	cudaEvent_t start, stop;
	for (int j = 0; j < THREADS * BLOCKS * 2; j++) {
		rdrand_64(nonce + j, 0);
		rdrand_64(key + j, 0);
	}

	for (int j = 0; j < THREADS * BLOCKS; j++) {
		rdrand_64(IV + j, 0);
	}
	cudaMemcpy(counter_d, counter, BLOCKS * THREADS * sizeof(__int64), cudaMemcpyHostToDevice);
	cudaMemcpy(nonce_d, nonce, BLOCKS * THREADS * 2 * sizeof(bit64), cudaMemcpyHostToDevice);
	cudaMemcpy(key_d, key, BLOCKS * THREADS * 2 * sizeof(bit64), cudaMemcpyHostToDevice);
	cudaMemcpy(IV_d, IV, BLOCKS * THREADS * sizeof(bit64), cudaMemcpyHostToDevice); //

	StartCounter();
	cudaDeviceSynchronize();
	cudaEventCreate(&start);	cudaEventCreate(&stop);	cudaEventRecord(start);
	gpu_gaston_linear_east << <BLOCKS, THREADS >> > (IV_d, key_d, nonce_d, counter_d, round, trial);
	cudaEventRecord(stop);	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);	printf("Time elapsed: %f milliseconds ", milliseconds);	printf("Time of kernel: %lf\n", GetCounter());

	cudaMemcpy(counter, counter_d, BLOCKS * THREADS * sizeof(__int64), cudaMemcpyDeviceToHost);
	for (int i = 0; i < BLOCKS * THREADS; i++) total_counter += counter[i];
	bias = (experiment) / 2 - total_counter;
	printf("Total counter: %I64d, Difference: %I64d, Bias: 2^-%lf\n\n", total_counter, bias, ((log(BLOCKS) + log(THREADS) + log(trial)) / log(2)) - (log(abs(bias)) / log(2)));

	cudaFree(key_d); cudaFree(nonce_d); cudaFree(counter_d); cudaFree(IV_d);
	free(key); free(nonce); free(counter); free(IV);

	return (long double)bias;

}

void show_menu() {
	printf(">>> GASTON Linear Distinguisher Experiment <<<\n\n"
		"(1) Starting from chi layer\n"
		"(2) Regular version\n"
		"(3) Clear screen\n"
		"(4) Exit\n\n"
		"Choice: ");
}

int main(void) {
	int choice = 0;
	while (1) {
		show_menu();
		scanf_s("%d", &choice);

		// cudaSetDevice(0);
		if (choice == 1) {
			int trial_i = 0, round;
			__int64 experiment, trial = 1;

			printf("Trial = 2^20 +  ");
			scanf_s("%d", &trial_i);

			if (trial_i > 42) {
				printf("Too large!\n");
				return 1;
			}
			printf("Rounds: ");
			scanf_s("%d", &round);

			trial = (__int64)1 << trial_i; // 2**trial_i
			experiment = (__int64)trial * THREADS * BLOCKS;

			printf("Running the experiment with %lld (2** %lld) data\n", experiment, trial_i + 20);
			gaston_main_chi(experiment, round, trial);
		}

		if (choice == 2) {
			int trial_i = 0, round;
			__int64 experiment, trial = 1;

			printf("Trial = 2^20 +  ");
			scanf_s("%d", &trial_i);

			if (trial_i > 42) {
				printf("Too large!\n");
				return 1;
			}
			printf("Rounds: ");
			scanf_s("%d", &round);

			trial = (__int64)1 << trial_i; // 2**trial_i
			experiment = (__int64)trial * THREADS * BLOCKS;

			printf("Running the experiment with %lld (2** %lld) data\n", experiment, trial_i + 20);
			gaston_main_east(experiment, round, trial);
		}

		if (choice == 3) {
#ifdef _WIN32
			system("cls");
#else
			system("clear");
#endif
		}
		if (choice == 4) {
			printf("Exiting the program...\n");
			exit(0);
		}
	}
	system("PAUSE");
}

/*
//3-round linear trail with[6, 5, 6] active S - boxes and weight 34 [12, 10, 12] -> bias: [2^7, 2^6, 2^7]
x0 = 0x0000000800000008
x1 = 0x1000000400000000
x2 = 0x1000000400000000
x3 = 0x0000000000000008
x4 = 0x0010000080000008
------------------------- chi-1
x0 = 0x0000000800000000
x1 = 0x1000000400000000
x2 = 0x0000000000000000
x3 = 0x0000000000000008
x4 = 0x0010000080000000
------------------------- rho-east-2
x0 = 0x0000000800000000
x1 = 0x0100000040000000
x2 = 0x0000000000000000
x3 = 0x0000000040000000
x4 = 0x0100000800000000
------------------------- theta-2
x0 = 0x0000000800000000
x1 = 0x0100000040000000
x2 = 0x0000000000000000
x3 = 0x0000000040000000
x4 = 0x0100000800000000
------------------------- rho-west-2
x0 = 0x0000000800000000
x1 = 0x0001000000400000
x2 = 0x0000000000000000
x3 = 0x0000000000001000
x4 = 0x0000000800004000
------------------------- chi-2
x0 = 0x0000000000000000
x1 = 0x0001000000400000
x2 = 0x0000000000400000
x3 = 0x0000000000001000
x4 = 0x0000000800004000
------------- ----------- rho-east-3
x0 = 0x0000000000000000
x1 = 0x0000100000040000
x2 = 0x0000100000000000
x3 = 0x0000008000000000
x4 = 0x0000008000040000
------------------------- theta-3
x0 = 0x0000000000000000
x1 = 0x0000100000040000
x2 = 0x0000100000000000
x3 = 0x0000008000000000
x4 = 0x0000008000040000
------------------------- rho-west-3
x0 = 0x0000000000000000
x1 = 0x0000001000000400
x2 = 0x0000000000000800
x3 = 0x0000000000200000
x4 = 0x2000000000040000
------------------------- chi-3
x0 = 0x0000000000040000
x1 = 0x0000001000000400
x2 = 0x0000000000000C00
x3 = 0x0000000000200800
x4 = 0x2000000000240000

*/