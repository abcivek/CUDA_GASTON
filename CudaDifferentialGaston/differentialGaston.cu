// differential distinguisher check for gaston
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

__device__ __forceinline__ uint64_t xorshift64star(uint64_t& state)
{
	state ^= state >> 12;
	state ^= state << 25;
	state ^= state >> 27;
	return state * 0x2545F4914F6CDD1DULL;
}

__global__ void gpuGastonDiff(uint64_t IV[], uint64_t key[], uint64_t nonce[], int64_t counter[], int round, int64_t trial) {

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

	for (int64_t c = 0; c < trial; c++) {
		/*
		// 3r path - Initial Difference
		pair0 = initial0 ^ 0x0000000000040001ULL;
		pair1 = initial1 ^ 0x0000000000000010ULL;
		pair2 = initial2 ^ 0x0000000002000080ULL;
		pair3 = initial3 ^ 0x0000000000100204ULL;
		pair4 = initial4 ^ 0x0000000100004000ULL;
		*/
		// 2r path - Initial Difference
		pair0 = initial0 ^ 0x0000000000000000ULL;
		pair1 = initial1 ^ 0x0020000000020000ULL;
		pair2 = initial2 ^ 0x8000000000000000ULL;
		pair3 = initial3 ^ 0x0404000000400000ULL;
		pair4 = initial4 ^ 0x0000000000000000ULL;

		// ---- chi initial ---- //  --> for the first round
		P = initial0;
		Q = initial1;
		initial0 ^= (initial2 & ~initial1);
		initial1 ^= (initial3 & ~initial2);
		initial2 ^= (initial4 & ~initial3);
		initial3 ^= (P & ~initial4);
		initial4 ^= (Q & ~P);

		for (int i = 0; i < round-1; i++) {  // Remaining full rounds; initial chi is counted as round 1
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

		// ---- chi pair ---- //  --> for the first round
		P = pair0;
		Q = pair1;
		pair0 ^= (pair2 & ~pair1);
		pair1 ^= (pair3 & ~pair2);
		pair2 ^= (pair4 & ~pair3);
		pair3 ^= (P & ~pair4);
		pair4 ^= (Q & ~P);

		for (int i = 0; i < round-1; i++) { // Remaining full rounds; initial chi is counted as round 1
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

        //output difference:
		if ((pair0 ^ initial0) == 0x0010000000000000ULL &&
			(pair1 ^ initial1) == 0x0010020000000028ULL &&
			(pair2 ^ initial2) == 0x0010000000000008ULL &&
			(pair3 ^ initial3) == 0x0800000080000008ULL &&
			(pair4 ^ initial4) == 0x0000020000000000ULL)
		{
			counter[tIdx]++;
		}

		// Randomize the next input from the current output
		uint64_t randomizer = initial0 ^ initial1 ^ initial2 ^ initial3 ^ initial4;
		randomizer = xorshift64star(randomizer);

		initial0 ^= randomizer;
		initial1 ^= randomizer;
		initial2 ^= randomizer;
		initial3 ^= randomizer;
		initial4 ^= randomizer;
	}
}
/*
//2r path 1r output difference:
if ((pair0 ^ initial0) == 0x0000000000000000ULL &&
    (pair1 ^ initial1) == 0x0020000000020000ULL &&
	(pair2 ^ initial2) == 0x8000000000000000ULL &&
	(pair3 ^ initial3) == 0x0404000000400000ULL &&
	(pair4 ^ initial4) == 0x0000000000000000ULL)

//2r path 2r output difference:
if ((pair0 ^ initial0) == 0x0010000000000000ULL &&
    (pair1 ^ initial1) == 0x0010020000000028ULL &&
    (pair2 ^ initial2) == 0x0010000000000008ULL &&
    (pair3 ^ initial3) == 0x0800000080000008ULL &&
    (pair4 ^ initial4) == 0x0000020000000000ULL)

///////////////////////////////////////////////////////////

// 3r path - first round output difference: 
if ((pair0 ^ initial0) == 0x0000000000040001ULL &&
	(pair1 ^ initial1) == 0x0000000000000010ULL &&
	(pair2 ^ initial2) == 0x0000000002000080ULL &&
	(pair3 ^ initial3) == 0x0000000000100204ULL &&
	(pair4 ^ initial4) == 0x0000000100004000ULL)

// 3r path - 2nd round output difference:
if ((pair0 ^ initial0) == 0x1000000000040001ULL &&
	(pair1 ^ initial1) == 0x0100000000000000ULL &&
	(pair2 ^ initial2) == 0x100000002000c800ULL &&
	(pair3 ^ initial3) == 0x0000000020008800ULL &&
	(pair4 ^ initial4) == 0x2000000000008000ULL)
*/
// #######################  GASTON SINGLE EXPERIMENT - MAIN ######################

long double gaston_diffD(int64_t experiment, int round, int64_t trial) {

	uint64_t* key, * key_d, * nonce, * nonce_d, * IV, * IV_d;
	int64_t* counter = 0, * counter_d, total_counter = 0;
	float milliseconds = 0;

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

	gpuGastonDiff <<<BLOCKS, THREADS >>> (IV_d, key_d, nonce_d, counter_d, round, trial);

	cudaEventRecord(stop);	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);	printf("Time elapsed: %f min ", milliseconds / 60000.0);	printf("Time of kernel: %lf\n", GetCounter());

	cudaMemcpy(counter, counter_d, BLOCKS * THREADS * sizeof(int64_t), cudaMemcpyDeviceToHost);
	for (int i = 0; i < BLOCKS * THREADS; i++) total_counter += counter[i];
	
	double probability = (double)total_counter / (double)experiment;

	if (total_counter == 0)
		printf("Total counter: 0, Probability: < 2^-%lld (no observable probability)\n\n",
			(long long)log2((double)experiment));
	else
		printf("Total counter: %lld, Probability: 2^-%lf\n\n",
			total_counter, -log2(probability));

	cudaFree(key_d); cudaFree(nonce_d); cudaFree(counter_d); cudaFree(IV_d);
	free(key); free(nonce); free(counter); free(IV);

	return (long double)probability;

}

void show_menu() {
	printf(">>> Gaston GPU-Accelerated Differential Distinguisher Verifier <<<\n\n"
		"(1) Differential Distinguisher\n"
		"(2) Clear screen\n"
		"(3) Exit\n\n"
		"Choice: ");
}

int main(void) {
	int choice = 0;
	while (1) {
		show_menu();
		scanf_s("%d", &choice);

		int trial_i = 0, round;
		int64_t experiment, trial = 1;

		// cudaSetDevice(0);
		if (choice == 1) {
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

			gaston_diffD(experiment, round, trial);
		}

		if (choice == 2) {
#ifdef _WIN32
			system("cls");
#else
			system("clear");
#endif
		}
		if (choice == 3) {
			printf("Exiting the program...\n");
			exit(0);
		}
	}
	system("PAUSE");
}


