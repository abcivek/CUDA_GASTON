// GASTON differential-linear cryptanalysis
// starts to check from the non-linear layer
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "rdrand.h"      
#include <Windows.h>
#include <math.h>
#include <ctime>
#include <stdlib.h>
#include <stdint.h>

#define BLOCKS 1024
#define THREADS 1024

// ***********************  timing *********************** //
double PCFreq = 0.0;
int64_t CounterStart = 0;

void StartCounter() { LARGE_INTEGER li; if (!QueryPerformanceFrequency(&li)) printf("QueryPerformanceFrequency failed!\n"); PCFreq = double(li.QuadPart) / 1000.0; QueryPerformanceCounter(&li); CounterStart = li.QuadPart; }
double GetCounter() { LARGE_INTEGER li; QueryPerformanceCounter(&li); return double(li.QuadPart - CounterStart) / PCFreq; }

// *********************** helper functions *********************** //
void print_state(uint64_t state[5]) { for (int i = 0; i < 5; i++) printf("%016I64x\n", state[i]); }

// *********************** GPU functions *********************** //
__device__ __forceinline__ uint64_t rotl(uint64_t x, int l) {
	l &= 63;
	return (x << l) | (x >> ((64 - l) & 63));
}

__global__ void gpuGastonS(uint64_t IV[], uint64_t key[], uint64_t nonce[], int64_t counter[], int round, int64_t trial) {

	int tIdx = blockIdx.x * blockDim.x + threadIdx.x; if (tIdx >= BLOCKS * THREADS) return; // kernel bound check
	int tIdx2 = 2 * tIdx;
	int tIdx3 = tIdx2 + 1;

	uint64_t initial0 = IV[tIdx]; 
	uint64_t initial1 = key[tIdx2];
	uint64_t initial2 = key[tIdx3];
	uint64_t initial3 = nonce[tIdx2];
	uint64_t initial4 = nonce[tIdx3];
	uint64_t pair0, pair1, pair2, pair3, pair4;
	uint64_t P, Q;

	uint64_t mask0, mask1, mask2, mask3, mask4;
	uint64_t t0;

	// 3r DL initial difference: x3 -> 0X8000000000000000
	// 4r DL initial difference: x0 -> 0x2000000000000000

	for (int c = 0; c < trial; c++) {
		pair0 = initial0; 
		pair1 = initial1; 
		pair2 = initial2; 
		pair3 = initial3 ^ 0X8000000000000000;
		pair4 = initial4; 

		///////////////  INITIAL /////////////// 
		// ---- chi ---- //  --> for the first round
		P = initial0;
		Q = initial1;
		initial0 ^= (initial2 & ~initial1);
		initial1 ^= (initial3 & ~initial2);
		initial2 ^= (initial4 & ~initial3);
		initial3 ^= (P & ~initial4);
		initial4 ^= (Q & ~P);

		for (int i = 0; i < round-1; i++) { // --> remaining rounds
			// ---- rho-east ---- //
			initial0 = rotl(initial0, 0); //e0 =0
			initial1 = rotl(initial1, 60); //e1 =60
			initial2 = rotl(initial2, 22); //e2 =22
			initial3 = rotl(initial3, 27); //e3 =27
			initial4 = rotl(initial4, 4); //e4 =4

			// ---- theta ---- //
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

			// ---- rho-west ---- //
			initial0 = rotl(initial0, 0); //w0=0
			initial1 = rotl(initial1, 56); //w1=56
			initial2 = rotl(initial2, 31); //w2=31
			initial3 = rotl(initial3, 46); //w3=46
			initial4 = rotl(initial4, 43); //w4=43

			// ---- chi ---- //
			P = initial0;
			Q = initial1;
			initial0 ^= (initial2 & ~initial1);
			initial1 ^= (initial3 & ~initial2);
			initial2 ^= (initial4 & ~initial3);
			initial3 ^= (P & ~initial4);
			initial4 ^= (Q & ~P);
		}
		///////////////  PAIR /////////////// 
		// ---- chi ---- // --> for the first round
		P = pair0;
		Q = pair1;
		pair0 ^= (pair2 & ~pair1);
		pair1 ^= (pair3 & ~pair2);
		pair2 ^= (pair4 & ~pair3);
		pair3 ^= (P & ~pair4);
		pair4 ^= (Q & ~P);

		for (int i = 0; i < round-1 ; i++) {// --> remaining rounds
			// ---- rho-east ---- */
			pair0 = rotl(pair0, 0); //e0 =0
			pair1 = rotl(pair1, 60); //e1 =60
			pair2 = rotl(pair2, 22); //e2 =22
			pair3 = rotl(pair3, 27); //e3 =27
			pair4 = rotl(pair4, 4); //e4 =4

			// ---- theta ---- //
			P = pair0 ^ pair1 ^ pair2 ^ pair3 ^ pair4;
			P ^= rotl(P, 1); // GASTON_r = 1

			Q = rotl(pair0, 25)
				^ rotl(pair1, 32)
				^ rotl(pair2, 52)
				^ rotl(pair3, 60)
				^ rotl(pair4, 63);
			// t0=25, t1=32, t2=52, t3=60,t4=63

			Q ^= rotl(Q, 18); //GASTON_s=18

			P ^= Q;
			P = rotl(P, 23); //GASTON_u = 23

			pair0 ^= P;
			pair1 ^= P;
			pair2 ^= P;
			pair3 ^= P;
			pair4 ^= P;

			// ---- rho-west ---- //
			pair0 = rotl(pair0, 0); //w0=0
			pair1 = rotl(pair1, 56); //w1=56
			pair2 = rotl(pair2, 31); //w2=31
			pair3 = rotl(pair3, 46); //w3=46
			pair4 = rotl(pair4, 43); //w4=43

			// ---- chi ---- //
			P = pair0;
			Q = pair1;
			pair0 ^= (pair2 & ~pair1);
			pair1 ^= (pair3 & ~pair2);
			pair2 ^= (pair4 & ~pair3);
			pair3 ^= (P & ~pair4);
			pair4 ^= (Q & ~P);

		}

		// linear approx (2nd)
		mask0 = (initial0 ^ pair0) & 0x0000000000000000ULL;
		mask1 = (initial1 ^ pair1) & 0x0001000000400000ULL;
		mask2 = (initial2 ^ pair2) & 0x0000000000400000ULL;
		mask3 = (initial3 ^ pair3) & 0x0000000000001000ULL;
		mask4 = (initial4 ^ pair4) & 0x0000000800004000ULL;

		t0 = mask0 ^ mask1 ^ mask2 ^ mask3 ^ mask4;

		t0 ^= t0 >> 1;
		t0 ^= t0 >> 2;
		t0 = (t0 & 0x1111111111111111UL) * 0x1111111111111111UL;
		t0 = (t0 >> 60) & 1;
		counter[tIdx] += (t0 == 0);

	}
}

// #######################  GASTON ROTATE ######################
__global__ void gaston_gpu_rotate(uint64_t IV[], uint64_t key[], uint64_t nonce[], int64_t counter[], int round, int64_t trial, int rotation)
{
	int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (tIdx >= BLOCKS * THREADS) return;
	int tIdx2 = 2 * tIdx;
	int tIdx3 = tIdx2 + 1;

	uint64_t initial0 = IV[tIdx]; 
	uint64_t initial1 = key[tIdx2];
	uint64_t initial2 = key[tIdx3];
	uint64_t initial3 = nonce[tIdx2];
	uint64_t initial4 = nonce[tIdx3];

	uint64_t pair0, pair1, pair2, pair3, pair4;
	uint64_t P, Q;

	uint64_t mask0, mask1, mask2, mask3, mask4;
	uint64_t t0;

	// safe rotation masks
	int r = rotation & 63;
	uint64_t temp0 = (r == 0) ? 0x7FFFFFFFFFFFFFFFULL : ((0x7FFFFFFFFFFFFFFFULL >> r) ^ (0x7FFFFFFFFFFFFFFFULL << (64 - r)));
	uint64_t temp1 = (r == 0) ? 0x8000000000000000ULL : ((0x8000000000000000ULL >> r) ^ (0x8000000000000000ULL << (64 - r)));

	for (int c = 0; c < trial; c++) {

		pair0 = initial0; //^ temp1;
		pair1 = initial1 ^ temp1;
		pair2 = initial2; //^ temp1;
		pair3 = initial3;//^ temp1;
		pair4 = initial4; //^ temp1;

		///////////////  INITIAL /////////////// 
		// ---- chi ---- // --> for the first round
		P = initial0;
		Q = initial1;
		initial0 ^= (initial2 & ~initial1);
		initial1 ^= (initial3 & ~initial2);
		initial2 ^= (initial4 & ~initial3);
		initial3 ^= (P & ~initial4);
		initial4 ^= (Q & ~P);

		for (int i = 0; i < round-1; i++) { // --> remaining rounds
			// ---- rho-east ---- //
			initial0 = rotl(initial0, 0); //e0 =0
			initial1 = rotl(initial1, 60); //e1 =60
			initial2 = rotl(initial2, 22); //e2 =22
			initial3 = rotl(initial3, 27); //e3 =27
			initial4 = rotl(initial4, 4); //e4 =4

			// ---- theta ---- //
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

			// ---- rho-west ---- //
			initial0 = rotl(initial0, 0); //w0=0
			initial1 = rotl(initial1, 56); //w1=56
			initial2 = rotl(initial2, 31); //w2=31
			initial3 = rotl(initial3, 46); //w3=46
			initial4 = rotl(initial4, 43); //w4=43

			// ---- chi ---- //
			P = initial0;
			Q = initial1;
			initial0 ^= (initial2 & ~initial1);
			initial1 ^= (initial3 & ~initial2);
			initial2 ^= (initial4 & ~initial3);
			initial3 ^= (P & ~initial4);
			initial4 ^= (Q & ~P);


		}

		///////////////  PAIR /////////////// 
		
		// ---- chi ---- // --> for the first round
		P = pair0;
		Q = pair1;
		pair0 ^= (pair2 & ~pair1);
		pair1 ^= (pair3 & ~pair2);
		pair2 ^= (pair4 & ~pair3);
		pair3 ^= (P & ~pair4);
		pair4 ^= (Q & ~P);

		for (int i = 0; i < round-1; i++) { // --> remaining rounds
			// ---- rho-east ---- //
			pair0 = rotl(pair0, 0); //e0 =0
			pair1 = rotl(pair1, 60); //e1 =60
			pair2 = rotl(pair2, 22); //e2 =22
			pair3 = rotl(pair3, 27); //e3 =27
			pair4 = rotl(pair4, 4); //e4 =4

			// ---- theta ---- //
			P = pair0 ^ pair1 ^ pair2 ^ pair3 ^ pair4;
			P ^= rotl(P, 1); // GASTON_r = 1

			Q = rotl(pair0, 25)
				^ rotl(pair1, 32)
				^ rotl(pair2, 52)
				^ rotl(pair3, 60)
				^ rotl(pair4, 63);
			// t0=25, t1=32, t2=52, t3=60,t4=63

			Q ^= rotl(Q, 18); //GASTON_s=18

			P ^= Q;
			P = rotl(P, 23); //GASTON_u = 23

			pair0 ^= P;
			pair1 ^= P;
			pair2 ^= P;
			pair3 ^= P;
			pair4 ^= P;

			// ---- rho-west ---- //
			pair0 = rotl(pair0, 0); //w0=0
			pair1 = rotl(pair1, 56); //w1=56
			pair2 = rotl(pair2, 31); //w2=31
			pair3 = rotl(pair3, 46); //w3=46
			pair4 = rotl(pair4, 43); //w4=43

			// ---- chi ---- //
			P = pair0;
			Q = pair1;
			pair0 ^= (pair2 & ~pair1);
			pair1 ^= (pair3 & ~pair2);
			pair2 ^= (pair4 & ~pair3);
			pair3 ^= (P & ~pair4);
			pair4 ^= (Q & ~P);

		}

		// linear approx (3rd)
		mask0 = (initial0 ^ pair0) & 0x0000000000040000ULL;
		mask1 = (initial1 ^ pair1) & 0x0000001000000400ULL;
		mask2 = (initial2 ^ pair2) & 0x0000000000000c00ULL;
		mask3 = (initial3 ^ pair3) & 0x0000000000200800ULL;
		mask4 = (initial4 ^ pair4) & 0x2000000000240000ULL;
		;

		t0 = mask0 ^ mask1 ^ mask2 ^ mask3 ^ mask4;

		t0 ^= t0 >> 1;
		t0 ^= t0 >> 2;
		t0 = (t0 & 0x1111111111111111UL) * 0x1111111111111111UL;
		t0 = (t0 >> 60) & 1;
		counter[tIdx] += (t0 == 0);

	}
}

/*  LINEAR MASKS:
round1:               ----- bias (one round): 2^-7
mask0 = (initial0 ^ pair0) & 0x0000000800000000ULL;
mask1 = (initial1 ^ pair1) & 0x1000000400000000ULL;
mask2 = (initial2 ^ pair2) & 0x0000000000000000ULL;
mask3 = (initial3 ^ pair3) & 0x0000000000000008ULL;
mask4 = (initial4 ^ pair4) & 0x0010000080000000ULL;

round2:               ----- bias (one round): 2^-6  -> for the 3r DL
mask0 = (initial0 ^ pair0) & 0x0000000000000000ULL;
mask1 = (initial1 ^ pair1) & 0x0001000000400000ULL;
mask2 = (initial2 ^ pair2) & 0x0000000000400000ULL;
mask3 = (initial3 ^ pair3) & 0x0000000000001000ULL;
mask4 = (initial4 ^ pair4) & 0x0000000800004000ULL;

round3:               ----- bias (one round):  2^-7 -> for the 4r DL
mask0 = (initial0 ^ pair0) & 0x0000000000040000ULL;
mask1 = (initial1 ^ pair1) & 0x0000001000000400ULL;
mask2 = (initial2 ^ pair2) & 0x0000000000000c00ULL;
mask3 = (initial3 ^ pair3) & 0x0000000000200800ULL;
mask4 = (initial4 ^ pair4) & 0x2000000000240000ULL;


total bias (2 rounds combined): 2^-12
total bias (3 rounds combined): 2^-18
*/

// #######################  GASTON SINGLE EXPERIMENT - MAIN ######################

long double gaston_single(int64_t experiment, int round, int64_t trial) {

	uint64_t* key, * key_d, * nonce, * nonce_d, * IV, * IV_d;
	int64_t* counter = 0, * counter_d, total_counter = 0, bias;
	float milliseconds = 0;

	FILE* fp;
	fopen_s(&fp, "DLResult.txt", "w");

	IV = (uint64_t*)calloc(BLOCKS * THREADS, sizeof(uint64_t));
	nonce = (uint64_t*)calloc(BLOCKS * THREADS * 2, sizeof(uint64_t));
	key = (uint64_t*)calloc(BLOCKS * THREADS * 2, sizeof(uint64_t));
	counter = (int64_t*)calloc(BLOCKS * THREADS, sizeof(int64_t));

	cudaMalloc((void**)&IV_d, BLOCKS * THREADS * sizeof(uint64_t));
	cudaMalloc((void**)&nonce_d, BLOCKS * THREADS * 2 * sizeof(uint64_t));
	cudaMalloc((void**)&key_d, BLOCKS * THREADS * 2 * sizeof(uint64_t));
	cudaMalloc((void**)&counter_d, BLOCKS * THREADS * sizeof(int64_t));

	cudaEvent_t start, stop;
	for (int j = 0; j < THREADS * BLOCKS * 2; j++) { rdrand_64(nonce + j, 0); rdrand_64(key + j, 0); }
	for (int j = 0; j < THREADS * BLOCKS; j++) { rdrand_64(IV + j, 0); }

	cudaMemcpy(IV_d, IV, BLOCKS * THREADS * sizeof(uint64_t), cudaMemcpyHostToDevice);
	cudaMemcpy(counter_d, counter, BLOCKS * THREADS * sizeof(int64_t), cudaMemcpyHostToDevice);
	cudaMemcpy(nonce_d, nonce, BLOCKS * THREADS * 2 * sizeof(uint64_t), cudaMemcpyHostToDevice);
	cudaMemcpy(key_d, key, BLOCKS * THREADS * 2 * sizeof(uint64_t), cudaMemcpyHostToDevice);

	StartCounter();
	cudaDeviceSynchronize();

	cudaEventCreate(&start);	cudaEventCreate(&stop);	cudaEventRecord(start);

	gpuGastonS << <BLOCKS, THREADS >> > (IV_d, key_d, nonce_d, counter_d, round, trial);

	cudaEventRecord(stop);	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);	printf("Time elapsed: %f min ", milliseconds / 60000.0);	printf("Time of kernel: %lf\n", GetCounter());

	cudaMemcpy(counter, counter_d, BLOCKS * THREADS * sizeof(int64_t), cudaMemcpyDeviceToHost);
	for (int i = 0; i < BLOCKS * THREADS; i++) total_counter += counter[i];
	bias = (experiment) / 2 - total_counter;
	if (bias == experiment/2) {
		printf("Total counter: %lld, Difference: %lld, Bias: ~0 (no detectable bias)\n\n",
			total_counter, bias);
		fprintf(fp, "Round: %d, Experiment: %lld data, Total counter: %lld, Difference: %lld, Bias: ~0 (no detectable bias)\n\n",
			round, experiment, total_counter, bias);
	}
	else {
		double log_bias = log2((double)llabs(bias));
		double logN = log2((double)experiment);
		printf("Total counter: %lld, Difference: %lld, Bias: 2^-%lf\n\n",
			total_counter, bias, logN - log_bias);
		fprintf(fp, "Round: %d, Experiment: %lld data, Total counter: %lld, Difference: %lld, Bias: 2^-%lf\n\n",
			round, experiment, total_counter, bias, logN - log_bias);
	}
	cudaFree(key_d); cudaFree(nonce_d); cudaFree(counter_d); cudaFree(IV_d);
	free(key); free(nonce); free(counter); free(IV);
	fclose(fp);

	return (long double)bias;

}

void show_menu() {
	printf(">>> Gaston Differential-Linear Distinguisher Finder/Verifier <<<\n\n"
		"(1) DL path verify\n"
		"(2) GPU rotation\n"
		"(3) Clear screen\n"
		"(4) Exit\n\n"
		"Choice: ");
}

int main(void) {
	int choice = 0;
	while (1) {
		show_menu();
		scanf_s("%d", &choice);

		if (choice == 3) {
#ifdef _WIN32
			system("cls");
#else
			system("clear");
#endif
			continue;
		}

		if (choice == 4) {
			printf("Exiting the program...\n");
			exit(0);
		}

		int trial_i = 0, round;
		int64_t experiment, trial = 1;

		printf("Trial = 2^20 +  ");
		scanf_s("%d", &trial_i);

		if (trial_i > 42) {
			printf("Too large!\n");
			return 1;
		}

		printf("Rounds: ");
		scanf_s("%d", &round);

		trial = (int64_t)1 << trial_i; // 2**trial_i
		experiment = (int64_t)trial * THREADS * BLOCKS;

		printf("Running the experiment with %lld (2** %lld) data\n", experiment, trial_i + 20);

		// cudaSetDevice(0);
		if (choice == 1) {
			gaston_single(experiment, round, trial);
		}

		else if (choice == 2) { // (3) GPU rotation version
			printf(">>> GASTON DL Distinguisher Finder <<<\n\n");

			FILE* fp;

			uint64_t* key, * key_d, * nonce, * nonce_d, * IV, * IV_d;
			int64_t* counter, * counter_d;
			int64_t total_counter = 0, bias;
			float milliseconds = 0;

			fopen_s(&fp, "Rotation.txt", "w");


			IV = (uint64_t*)calloc(BLOCKS * THREADS, sizeof(uint64_t));
			key = (uint64_t*)calloc(BLOCKS * THREADS * 2, sizeof(uint64_t));
			nonce = (uint64_t*)calloc(BLOCKS * THREADS * 2, sizeof(uint64_t));
			counter = (int64_t*)calloc(BLOCKS * THREADS, sizeof(int64_t));

			cudaMalloc(&IV_d, BLOCKS * THREADS * sizeof(uint64_t));
			cudaMalloc(&key_d, BLOCKS * THREADS * 2 * sizeof(uint64_t));
			cudaMalloc(&nonce_d, BLOCKS * THREADS * 2 * sizeof(uint64_t));
			cudaMalloc(&counter_d, BLOCKS * THREADS * sizeof(int64_t));

			for (int rotation = 0; rotation < 64; rotation++) {
				printf("Rotation: %d\n", rotation);
				fprintf(fp, "Rotation: %d\n", rotation);

				//reset counter
				memset(counter, 0, BLOCKS * THREADS * sizeof(int64_t));
				total_counter = 0;

				cudaMemcpy(counter_d, counter, BLOCKS * THREADS * sizeof(int64_t), cudaMemcpyHostToDevice);

				// random input
				for (int k = 0; k < BLOCKS * THREADS; k++) {
					rdrand_64(IV + k, 0);
				}

				for (int j = 0; j < BLOCKS * THREADS * 2; j++) {
					rdrand_64(nonce + j, 0);
					rdrand_64(key + j, 0);
				}

				cudaMemcpy(IV_d, IV, BLOCKS * THREADS * sizeof(uint64_t), cudaMemcpyHostToDevice);
				cudaMemcpy(key_d, key, BLOCKS * THREADS * 2 * sizeof(uint64_t), cudaMemcpyHostToDevice);
				cudaMemcpy(nonce_d, nonce, BLOCKS * THREADS * 2 * sizeof(uint64_t), cudaMemcpyHostToDevice);

				cudaDeviceSynchronize();
				StartCounter();

				gaston_gpu_rotate << <BLOCKS, THREADS >> > (IV_d, key_d, nonce_d, counter_d, round, trial, rotation);

				cudaDeviceSynchronize();
				printf("Kernel time: %lf min, %lf ms\n", GetCounter() / 60000.0, GetCounter());

				cudaMemcpy(counter, counter_d, BLOCKS * THREADS * sizeof(int64_t), cudaMemcpyDeviceToHost);

				for (int i = 0; i < BLOCKS * THREADS; i++)
					total_counter += counter[i];

				int64_t diff = (experiment / 2) - total_counter;

				double log_bias;
				if (diff == 0) log_bias = 0;
				else log_bias = log2((double)llabs(diff));

				double logN = log2((double)experiment);
				double result = logN - log_bias;

				if (llabs(total_counter)/2 == llabs(diff)) {
					printf("Total counter: %lld, Difference: %lld, Bias: ~0 (no detectable bias)\n\n",
						total_counter, bias);
				}
				else {
					printf("Total: %lld, Diff: %lld, Bias: 2^-%lf\n\n",
						total_counter, diff, result);

					fprintf(fp, "Total: %lld, Diff: %lld, Bias: 2^-%lf\n\n",
						total_counter, diff, result);
				}



				fflush(fp);
			}
			cudaFree(key_d); cudaFree(nonce_d); cudaFree(counter_d); cudaFree(IV_d);

			free(key); free(nonce); free(counter); free(IV);

			fclose(fp);

		}

	}
	system("PAUSE");
}

/*
* With NVIDIA 4060 GPU, 4r experiment:
 # 2^38: 0.53 min (64 rotation: 34 min)
 # 2^40: 2.15 min (64 rotation: 2 hour 20 min)
 # 2^42: 8.5 min  (64 rotation: 9 hour)
/*
Rotation:0  0X8000000000000000  1000000000000000000000000000000000000000000000000000000000000000
Rotation:1  0x4000000000000000  0100000000000000000000000000000000000000000000000000000000000000
Rotation:2  0x2000000000000000  0010000000000000000000000000000000000000000000000000000000000000
Rotation:3  0x1000000000000000  0001000000000000000000000000000000000000000000000000000000000000
Rotation:4  0x0800000000000000  0000100000000000000000000000000000000000000000000000000000000000
Rotation:5  0x0400000000000000  0000010000000000000000000000000000000000000000000000000000000000
Rotation:6  0x0200000000000000  0000001000000000000000000000000000000000000000000000000000000000
Rotation:7  0x0100000000000000  0000000100000000000000000000000000000000000000000000000000000000
Rotation:8  0x0080000000000000  0000000010000000000000000000000000000000000000000000000000000000
Rotation:9  0x0040000000000000  0000000001000000000000000000000000000000000000000000000000000000
Rotation:10 0x0020000000000000  0000000000100000000000000000000000000000000000000000000000000000
Rotation:11 0x0010000000000000  0000000000010000000000000000000000000000000000000000000000000000
Rotation:12 0x0008000000000000  0000000000001000000000000000000000000000000000000000000000000000
Rotation:13 0x0004000000000000  0000000000000100000000000000000000000000000000000000000000000000
Rotation:14 0x0002000000000000  0000000000000010000000000000000000000000000000000000000000000000
Rotation:15 0x0001000000000000  0000000000000001000000000000000000000000000000000000000000000000
Rotation:16 0x0000800000000000  0000000000000000100000000000000000000000000000000000000000000000
Rotation:17 0x0000400000000000  0000000000000000010000000000000000000000000000000000000000000000
Rotation:18 0x0000200000000000  0000000000000000001000000000000000000000000000000000000000000000
Rotation:19 0x0000100000000000  0000000000000000000100000000000000000000000000000000000000000000
Rotation:20 0x0000080000000000  0000000000000000000010000000000000000000000000000000000000000000
Rotation:21 0x0000040000000000  0000000000000000000001000000000000000000000000000000000000000000
Rotation:22 0x0000020000000000  0000000000000000000000100000000000000000000000000000000000000000
Rotation:23 0x0000010000000000  0000000000000000000000010000000000000000000000000000000000000000
Rotation:24 0x0000008000000000  0000000000000000000000001000000000000000000000000000000000000000
Rotation:25 0x0000004000000000  0000000000000000000000000100000000000000000000000000000000000000
Rotation:26 0x0000002000000000  0000000000000000000000000010000000000000000000000000000000000000
Rotation:27 0x0000001000000000  0000000000000000000000000001000000000000000000000000000000000000
Rotation:28 0x0000000800000000  0000000000000000000000000000100000000000000000000000000000000000
Rotation:29 0x0000000400000000  0000000000000000000000000000010000000000000000000000000000000000
Rotation:30 0x0000000200000000  0000000000000000000000000000001000000000000000000000000000000000
Rotation:31 0x0000000100000000  0000000000000000000000000000000100000000000000000000000000000000
Rotation:32 0x0000000080000000  0000000000000000000000000000000010000000000000000000000000000000
Rotation:33 0x0000000040000000  0000000000000000000000000000000001000000000000000000000000000000
Rotation:34 0x0000000020000000  0000000000000000000000000000000000100000000000000000000000000000
Rotation:35 0x0000000010000000  0000000000000000000000000000000000010000000000000000000000000000
Rotation:36 0x0000000008000000  0000000000000000000000000000000000001000000000000000000000000000
Rotation:37 0x0000000004000000  0000000000000000000000000000000000000100000000000000000000000000
Rotation:38 0x0000000002000000  0000000000000000000000000000000000000010000000000000000000000000
Rotation:39 0x0000000001000000  0000000000000000000000000000000000000001000000000000000000000000
Rotation:40 0x0000000000800000  0000000000000000000000000000000000000000100000000000000000000000
Rotation:41 0x0000000000400000  0000000000000000000000000000000000000000010000000000000000000000
Rotation:42 0x0000000000200000  0000000000000000000000000000000000000000001000000000000000000000
Rotation:43 0x0000000000100000  0000000000000000000000000000000000000000000100000000000000000000
Rotation:44 0x0000000000080000  0000000000000000000000000000000000000000000010000000000000000000
Rotation:45 0x0000000000040000  0000000000000000000000000000000000000000000001000000000000000000
Rotation:46 0x0000000000020000  0000000000000000000000000000000000000000000000100000000000000000
Rotation:47 0x0000000000010000  0000000000000000000000000000000000000000000000010000000000000000
Rotation:48 0x0000000000008000  0000000000000000000000000000000000000000000000001000000000000000
Rotation:49 0x0000000000004000  0000000000000000000000000000000000000000000000000100000000000000
Rotation:50 0x0000000000002000  0000000000000000000000000000000000000000000000000010000000000000
Rotation:51 0x0000000000001000  0000000000000000000000000000000000000000000000000001000000000000
Rotation:52 0x0000000000000800  0000000000000000000000000000000000000000000000000000100000000000
Rotation:53 0x0000000000000400  0000000000000000000000000000000000000000000000000000010000000000
Rotation:54 0x0000000000000200  0000000000000000000000000000000000000000000000000000001000000000
Rotation:55 0x0000000000000100  0000000000000000000000000000000000000000000000000000000100000000
Rotation:56 0x0000000000000080  0000000000000000000000000000000000000000000000000000000010000000
Rotation:57 0x0000000000000040  0000000000000000000000000000000000000000000000000000000001000000
Rotation:58 0x0000000000000020  0000000000000000000000000000000000000000000000000000000000100000
Rotation:59 0x0000000000000010  0000000000000000000000000000000000000000000000000000000000010000
Rotation:60 0x0000000000000008  0000000000000000000000000000000000000000000000000000000000001000
Rotation:61 0x0000000000000004  0000000000000000000000000000000000000000000000000000000000000100
Rotation:62 0x0000000000000002  0000000000000000000000000000000000000000000000000000000000000010
Rotation:63 0x0000000000000001  0000000000000000000000000000000000000000000000000000000000000001
*/