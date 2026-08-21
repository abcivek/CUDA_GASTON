#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "rdrand.h"      
#include <Windows.h>
#include <math.h>
#include <ctime>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <time.h>

#define BLOCKS	1024
#define THREADS	1024

#define bit64 unsigned __int64 

// ***********************  timing *********************** // 
double PCFreq = 0.0;
__int64 CounterStart = 0;

void StartCounter() { LARGE_INTEGER li; if (!QueryPerformanceFrequency(&li)) printf("QueryPerformanceFrequency failed!\n"); PCFreq = double(li.QuadPart) / 1000.0; QueryPerformanceCounter(&li); CounterStart = li.QuadPart; }
double GetCounter() { LARGE_INTEGER li; QueryPerformanceCounter(&li); return double(li.QuadPart - CounterStart) / PCFreq; }

// ****************** helper functions ******************* //
__device__ __forceinline__ bit64 rotl(bit64 x, int l) {
	l &= 63;
	return (x << l) | (x >> ((64 - l) & 63));
}

static inline uint64_t rotl64(uint64_t x, unsigned int r)
{
	r &= 63;
	if (r == 0) return x;
	return (x << r) | (x >> (64 - r));
}
void print_state(const uint64_t state[5])
{
	for (int i = 0; i < 5; i++)
		printf("%016" PRIx64 "\n", state[i]);
	printf("\n");
}
static const uint64_t gaston_rc_cpu[12] = {
	0xF0ULL, 0xE1ULL, 0xD2ULL,
	0xC3ULL, 0xB4ULL, 0xA5ULL,
	0x96ULL, 0x87ULL, 0x78ULL,
	0x69ULL, 0x5AULL, 0x4BULL
};

// Gaston Permutation //
void gaston(uint64_t state[5])
{
	for (int round = 0; round < 12; round++)
	{
		uint64_t P, Q;

		/* ---- rho-east ---- */
		state[0] = rotl64(state[0], 0);
		state[1] = rotl64(state[1], 60);
		state[2] = rotl64(state[2], 22);
		state[3] = rotl64(state[3], 27);
		state[4] = rotl64(state[4], 4);

		/* ---- theta ---- */
		P = state[0] ^ state[1] ^ state[2] ^ state[3] ^ state[4];
		P ^= rotl64(P, 1);

		Q = rotl64(state[0], 25)
			^ rotl64(state[1], 32)
			^ rotl64(state[2], 52)
			^ rotl64(state[3], 60)
			^ rotl64(state[4], 63);

		Q ^= rotl64(Q, 18);

		P ^= Q;
		P = rotl64(P, 23);

		state[0] ^= P;
		state[1] ^= P;
		state[2] ^= P;
		state[3] ^= P;
		state[4] ^= P;

		/* ---- rho-west ---- */
		state[0] = rotl64(state[0], 0);
		state[1] = rotl64(state[1], 56);
		state[2] = rotl64(state[2], 31);
		state[3] = rotl64(state[3], 46);
		state[4] = rotl64(state[4], 43);

		/* ---- iota ---- */
		state[0] ^= gaston_rc_cpu[round];

		/* ---- chi ---- */
		P = state[0];
		Q = state[1];
		state[0] ^= (state[2] & ~state[1]);
		state[1] ^= (state[3] & ~state[2]);
		state[2] ^= (state[4] & ~state[3]);
		state[3] ^= (P & ~state[4]);
		state[4] ^= (Q & ~P);
	}
}

// *********************** Test vectors  **************************** //
void test_vectors(void)
{	//uint64_t state[5] = { 0x1F4AD9906DA6A254, 0x4B84D7F83F2BDDFA, 0x468A0853578A00E3, 0x6C05A0506DF7F66E, 0x4EFB22112453C964 };
	//uint64_t state[5] = { 0xFFFFFFFFFFFFFFFF, 0x0123456789ABCDEF, 0xFEDCBA9876543210, 0xAAAAAAAAAAAAAAAA, 0x0101010101010101 };
	uint64_t state[5] = { 0 };
	/*
	Expected:
	88b326096bebc635
	6ca8fb64bc5ce6ca
	f1ce3840d8190713
	54d70067438689b5
	f17fe863f958f32b
	*/
	printf("Initial state:\n");
	print_state(state);

	gaston(state);

	printf("State after 12 rounds:\n");
	print_state(state);
}

// *********************** GPU BENCHMARK - GASTON *********************** //

__constant__ uint64_t gaston_rc_gpu[12] = {
	0xF0ULL, 0xE1ULL, 0xD2ULL,
	0xC3ULL, 0xB4ULL, 0xA5ULL,
	0x96ULL, 0x87ULL, 0x78ULL,
	0x69ULL, 0x5AULL, 0x4BULL
};

// CUDA GASTON 12-round Benchmark
__global__ void GASTON12_benchmark_gpu(bit64 IV[], bit64 key[], bit64 nonce[], bit64 keystream, __int64 trial) {
	//threadIndex = tidx
	int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
	int tIdx2 = 2 * tIdx; 
	int tIdx3 = tIdx2 + 1;

	bit64 initial0, initial1, initial2, initial3, initial4;
	bit64 P, Q;

	bit64 IV2 = IV[tIdx];
	bit64 key0 = key[tIdx2];
	bit64 key1 = key[tIdx3];
	bit64 nonce0 = nonce[tIdx2];
	bit64 nonce1 = nonce[tIdx3];

	for (__int64 c = 0; c < trial; c++) {
		initial0 = IV2;
		initial1 = key0;
		initial2 = key1;
		initial3 = nonce0;
		initial4 = nonce1;
#pragma unroll 
		for (int i = 0; i < 12; i++) {
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
			initial0 ^= gaston_rc_gpu[i];

			/* ---- chi ---- */
			P = initial0;
			Q = initial1;
			initial0 ^= (initial2 & ~initial1);
			initial1 ^= (initial3 & ~initial2);
			initial2 ^= (initial4 & ~initial3);
			initial3 ^= (P & ~initial4);
			initial4 ^= (Q & ~P);
		}
		nonce1++;
		if (initial0 == keystream) printf("Hello world\n");
	}
}

void GASTON_benchmark_gpu() {
	bit64* nonce, * nonce_d, * keyrows, * keyrows_d, * IV, * IV_d;
	__int64 trial = 1;

	printf("Trial = 2^20 +  ");
	scanf_s("%I64d", &trial);
	trial = (__int64)1 << trial;

	float milliseconds = 0;

	nonce = (bit64*)calloc(BLOCKS * THREADS * 2, sizeof(bit64));
	keyrows = (bit64*)calloc(BLOCKS * THREADS * 2, sizeof(bit64));
	IV = (bit64*)calloc(BLOCKS * THREADS, sizeof(bit64));

	cudaMalloc((void**)&nonce_d, BLOCKS * THREADS * 2 * sizeof(bit64));
	cudaMalloc((void**)&keyrows_d, BLOCKS * THREADS * 2 * sizeof(bit64));
	cudaMalloc((void**)&IV_d, BLOCKS * THREADS * sizeof(bit64));

	cudaEvent_t start, stop;

	for (int j = 0; j < THREADS * BLOCKS * 2; j++) { rdrand_64(nonce + j, 0); rdrand_64(keyrows + j, 0); }
	for (int j = 0; j < THREADS * BLOCKS; j++) { rdrand_64(IV + j, 0); }

	cudaMemcpy(nonce_d, nonce, BLOCKS * THREADS * 2 * sizeof(bit64), cudaMemcpyHostToDevice);
	cudaMemcpy(keyrows_d, keyrows, BLOCKS * THREADS * 2 * sizeof(bit64), cudaMemcpyHostToDevice);
	cudaMemcpy(IV_d, IV, BLOCKS * THREADS * sizeof(bit64), cudaMemcpyHostToDevice);

	StartCounter();
	cudaDeviceSynchronize();
	cudaEventCreate(&start);	cudaEventCreate(&stop);	cudaEventRecord(start);

	GASTON12_benchmark_gpu << <BLOCKS, THREADS >> > (IV_d, keyrows_d, nonce_d, 0x0123456789abcdef, trial); // keystream=0x0123456789abcdef

	cudaEventRecord(stop);	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("CUDA error: %s\n", cudaGetErrorString(err));

	printf("Time elapsed: %f milliseconds ", milliseconds);
	printf("Time of kernel: %lf\n", GetCounter());

	cudaFree(keyrows_d); cudaFree(nonce_d); cudaFree(IV_d);
	free(nonce); free(keyrows); free(IV);

}

void show_menu() {
	printf(
		"(0) Test vectors\n"
		"(1) Gaston 12-round CPU Benchmark\n"
		"(2) CUDA GASTON 12-round Benchmark\n"
		"(3) Clear screen\n"
		"(4) Exit\n\n"
		"Choice: ");
}

int main(void) {
	//int deviceCount;
	//cudaGetDeviceCount(&deviceCount);
	//printf("GPUs available: %d\n", deviceCount);
	//cudaSetDevice(0); 
	int choice = 0;

	while (1) {
		show_menu();
		scanf_s("%d", &choice);
		// ***************************************
		if (choice == 0) test_vectors();
		// ***************************************
		else if (choice == 1) { // Gaston 12-round CPU Benchmark
			uint64_t s;
			uint64_t trial_i;
			uint64_t experiment_size;

			printf("Data size (2^(20 + ? )): ");
			if (scanf_s("%llu", &trial_i) != 1)
				return 0;

			if (trial_i > 43)   // because exp + 20 must be < 64 
			{
				printf("Exponent too large (would overflow 64-bit).\n");
				continue;
			}

			experiment_size = 1ULL << (trial_i + 20);
			printf("Running 2^(%llu) iterations...\n", trial_i + 20);
			uint64_t state[5] = { 0, 0, 0, 0, 0 };
			clock_t start = clock();
			for (s = 0; s < experiment_size; s++)
				gaston(state);
			clock_t end = clock();

			// Prevent compiler from optimizing loop away 
			volatile uint64_t sink = state[0];
			(void)sink; // suppress unused warning

			printf("Time: %.3f seconds\n",
				(double)(end - start) / CLOCKS_PER_SEC);
		}
		// ***************************************
		else if (choice == 2) GASTON_benchmark_gpu(); // CUDA GASTON 12-round Benchmark
		// ***************************************
		else if (choice == 3) {
#ifdef _WIN32
			system("cls");
#else
			system("clear");
#endif
		}
		// ***************************************
		else if (choice == 4) {
			printf("Exiting the program...\n");
			exit(0);
		}
		// ***************************************
		else { printf("Invalid choice.\n"); }
	}

	return 0;
}