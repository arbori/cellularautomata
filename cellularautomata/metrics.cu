#define __CUDA_INTERNAL_COMPILATION__
#include "math_functions.h"
#undef __CUDA_INTERNAL_COMPILATION__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cellularautomata.h"

#include "Memory.h"

#include <cmath>
#include <iostream>

#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <ctime>       /* clock_t, clock, CLOCKS_PER_SEC */

using namespace std;

#define MAX_NUMBER_OF_STATES 8

//#define MAX_R 12
//#define MAX_NEIGHBORHOOD_SIZE 25
//#define MAX_MI_NUMBER (MAX_R*MAX_R)

cell_t TransitionFunction(Tree& tree, cell_t* env, float r, size_t length,
	size_t i);

/*!
*
*/
__host__ __device__ int FromDigits(cell_t* line, int ini, int fim, cell_t k) {
	int value = 0;
	int bits = (fim - ini);

	for (int n = bits; n > 0; n--) {
		value += static_cast<int>(powf(k, static_cast<float>(bits - n))) * line[fim - (bits - n) - 1];
	}

	return value;
}

/*!
*

__host__ __device__ int WinNumber(cell_t* M, int X, int Y, int x, int y, int width, int height, cell_t k) {
	// Encara a janela como uma sequencia de bits.
	int bit_size = width * height;

	if (bit_size <= 1) {
		return static_cast<size_t>(M[x + y*X]);
	}

	int result = 0;

	int istart = y - (height / 2);
	int iend   = y + (height / 2);
	int jstart = x - (width / 2);
	int jend   = x + (width / 2);

	int bit = 0;
	int l, c;

	for (int i = istart; i < iend; i++) {
		for (int j = jstart; j < jend; j++) {
			if (i < 0) l = i + Y;
			else if (i >= Y) l = i - Y;
			else l = i;

			if (j < 0) c = j + X;
			else if (j >= X) c = j - X;
			else c = j;

			result += M[c + l*X] * static_cast<int>(pow(2, bit++));
		}
	}

	return result;
}
*/

/*!
*
*/
__host__ __device__ int NeighborhoodNumber(cell_t* line, int width, int i, float r, cell_t k) {
	// Toda vizinhança é o dobro do raio mais a célula central
	int neighborhoodSize = static_cast<int>(2 * r + 1);

	if (neighborhoodSize <= 1) {
		return static_cast<int>(line[i]);
	}

	// A quantidade de células à esquerda. Se o tamanho for impar a quantidade
	// de células em torno da célula central são iguais, caso contrário a 
	// quantidada à esquarda é uma menor.
	int loffset = (neighborhoodSize % 2 != 0) ? neighborhoodSize / 2 : neighborhoodSize / 2 - 1;
	int roffset = neighborhoodSize / 2;
	int ini = i - loffset;
	int fim = i + roffset + 1;

	if (ini < 0) {
		ini = width + ini;
	}
	else if (ini >= width) {
		ini = ini - width;
	}
	if (fim >= width) {
		fim = fim - width;
	}
	else if (fim < 0) {
		fim = width - fim;
	}

	int result = 0;
	int potencia = neighborhoodSize - 1;

	while (ini != fim) {
		result += line[ini++] * static_cast<int>(powf(k, static_cast<float>(potencia--)));

		if (ini < 0) {
			ini = width + ini;
		}
		else if (ini >= width) {
			ini = ini - width;
		}
	}

	return result;
}

/*!
*
*/
__global__ void SiteEntropyKernel(cell_t* mat, float* entropy, int X, int Y, int Z, float r, cell_t k, float* states) {
	int x = threadIdx.x;
	int z = threadIdx.z + blockIdx.z * blockDim.z;

	if (x >= X || z >= Z) {
		return;
	}

	int s = 0;

	for (int y = 0; y < Y; y++) {
		s = mat[x + y*X + z*X*Y];

		states[s + x*k + z*k*X] += 1.0;
	}

	entropy[x + z*X] = 0.0f;

	for (s = 0; s < k; s++) {
		if (states[s + x*k + z*k*X] != 0.0f) {
			states[s + x*k + z*k*X] /= Y;

			entropy[x + z*X] += -static_cast<float>(
				states[s + x*k + z*k*X] *
				log2f(states[s + x*k + z*k*X])
			);
		}
	}
}

/*!
*
*/
void SiteEntropyDevice(Memory<cell_t>& mat, Memory<float>& entropy, int X, int Y, int Z, float r, cell_t k)
{
	int size = 0;

	for (int z = 0; z < Z; z++) {
		size += X * Z;
	}

	size *= k;

	Memory<float> states(size);

	states.assignAll(0);
	states.hostToDevice();

	mat.hostToDevice();

	dim3 thread(X, 1, 1024 / X);
	dim3 block(1, 1, Z / thread.z);

	SiteEntropyKernel <<<block, thread>>>(mat.device(), entropy.device(), X, Y, Z,
		r, k, states.device());

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	entropy.deviceToHost();
}

/*!
*
*/
void SiteEntropy(Memory<cell_t>& mat, Memory<float>& entropy, int X, int Y, int Z, float r, cell_t k) {
	Memory<float> states(k);

	int s = 0;

	for (int z = 0; z < Z; z++) {
		for (int x = 0; x < X; x++) {
			for (s = 0; s < k; s++) {
				states[s] = 0.0f;
			}

			for (int y = 0; y < Y; y++) {
				s = mat[x + y*X + z*X * Y];

				states[s] += 1.0;
			}

			entropy[x + z*X] = 0.0f;

			for (s = 0; s < k; s++) {
				if (states[s] != 0.0f) {
					states[s] /= Y;

					entropy[x + z*X] += -static_cast<float>(
						states[s] * log2f(states[s]));
				}
			}
		}
	}
}

/*!
 *
 */
__host__ __device__ void computeNE(cell_t* mat, float* entropy, float* states,
		size_t X, size_t M, size_t w, float r, cell_t k, int y) {
	size_t m = 0;

	for (m = 0; m < M; m++) {
		states[m] = 0.0f;
	}

	for (size_t x = 0; x < X; x++) {
		for (size_t l = y; l < y + w; l++) {
			m = NeighborhoodNumber(&mat[0 + l*X], X, x, r, k);

			states[m] += 1.0;
		}
	}

	entropy[y] = 0.0f;

	for (m = 0; m < M; m++) {
		if (states[m] != 0.0f) {
			entropy[y] += -static_cast<float>(
				(states[m] / (X*w)) * log2f(states[m] / (X*w)));
		}
	}
}
/*!
 *
 */
__global__ void ComputeNeighborhoodEntropyDevice(cell_t* mat, float* entropy, float* states,
		size_t X, size_t Y, size_t Z, size_t M, size_t w, float r, cell_t k) {
	int y = threadIdx.y;
	int z = blockIdx.z; // +threadIdx.z*blockDim.x;

	if (y >= Y || z >= Z) {
		return;
	}

	computeNE(mat + (z*X*Y), entropy + (z*Y), states + (y*M + z*M*(Y - w + 1)), X, M, w, r, k, y);
}

void NeighborhoodEntropyDevice(Memory<cell_t>& mat, Memory<float>& entropy, size_t X, size_t Y, size_t Z, size_t w, float r, cell_t k) {
	size_t M = static_cast<int>(powf(k, 2 * r + 1));
	Memory<float> states(M * Y * Z);

	mat.hostToDevice();

	// Ajusta o valor de w para o valor mínimo.
	if (w < 1 || w > Y) {
		w = Y;
	}

	dim3 bloco(1, 1, Z);
	dim3 threads(1, Y - w + 1, 1);

	ComputeNeighborhoodEntropyDevice << <bloco, threads >> > (mat.device(), entropy.device(), states.device(), X, Y, Z, M, w, r, k);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	entropy.deviceToHost();

	states.offset(0);
	states.deviceToHost();

	/*
	cout << "States from device\n\n";
	for (int z = 0; z < Z; z++) {
		for (int y = 0; y < (Y - w + 1); y++) {
			for (int m = 0; m < M; m++) {
				cout << states[m + y*M + z*M*(Y - w + 1)] << " ";
			}

			cout << endl;
		}

		cout << endl;
		cout << endl;
	}
	*/
}

/*!
 *
 */
void NeighborhoodEntropy(Memory<cell_t>& mat, Memory<float>& entropy, size_t X, size_t Y, size_t Z, size_t w, float r, cell_t k) {
	// Tamanho da vizinhança
	size_t M = static_cast<int>(powf(k, 2 * r + 1));
	// Cubo que guarda a quantidade de cada estado
	Memory<float> states(M * Y * Z);

	// Ajusta o valor de w para o valor mínimo.
	if (w < 1 || w > Y) {
		w = Y;
	}

	states.assignAll(0);

	for (size_t z = 0; z < Z; z++) {
		mat.offset(z*X*Y);
		entropy.offset(z*Y);

		// Percorre cada célula do reticulado
		for (size_t y = 0; y < Y - w + 1; y++) {
			states.offset(y*M + z*M*(Y - w + 1));

			computeNE(mat.host(), entropy.host(), states.host(), X, M, w, r, k, y);
		}
	}

	mat.offset(0);
	entropy.offset(0);

	/*
	states.offset(0);

	cout << "States from host\n\n";
	for (int z = 0; z < Z; z++) {
		for (int y = 0; y < (Y - w + 1); y++) {
			for (int m = 0; m < M; m++) {
				cout << states[m + y*M + z*M*(Y - w + 1)] << " ";
			}

			cout << endl;
		}

		cout << endl;
		cout << endl;
	}
	*/
}

/*!
 *
 */
__global__ void MutualInformationDevice(cell_t* mat, float* mi, cell_t k, int Y) {
//	int Z = gridDim.x;
	int X = blockDim.x;

	int x0 = threadIdx.x;
	int z = blockIdx.x;

	// Define a matriz de probabilidade dos valores das colunas de mat.
	// São 2 colunas por k linhas para calcular a probabiliade 
	// de cada coluna x0 e x1.
	float sitep[2 * MAX_NUMBER_OF_STATES];

	// A probabiliade conjunta das colunas X0 e x1 precisará de 
	// k*k linhas e uma coluna para cada par de culunas.
	float colsp[MAX_NUMBER_OF_STATES * MAX_NUMBER_OF_STATES];

	size_t n;
	int idx;

	// Varia a segunda coluna.
	for (int x1 = 0; x1 < X; x1++) {
		// Zera os acumuladores.
		for (n = 0; n < 2 * k; sitep[n++] = 0.0f);
		for (n = 0; n < k * k; colsp[n++] = 0.0f);

		// Quantifica os estados isolados de cada coluna em sitep 
		// e os estados conjuntos das colunas x0 e x1 em colsp
		for (int y = 0; y < Y; y++) {
			// Calcula o indice da primeira coluna para a linha 
			// do estado correspondente, o valor de mat na posição
			idx = 0 + mat[x0 + y*X + z*X*Y] * 2;
			// Incrementa o estado correspondente para a coluna x0
			sitep[idx] += (1.0f / Y);

			// Calcula o indice da segunda coluna para a linha 
			// do estado correspondente, o valor de mat na posição
			idx = 1 + mat[x1 + y*X + z*X*Y] * 2;
			// Incrementa o estado correspondente para a coluna x1
			sitep[idx] += (1.0f / Y);

			// Calcula o indice do estado conjunto representado pelo 
			// valor numérico de base k. A coluna x0 é a do dígito mais
			// significativo e a x1 o menos significativo.
			idx = mat[x0 + y*X + z*X*Y] * k
				+ mat[x1 + y*X + z*X*Y];
			// Incrementa o estado conjunto na linha correspondente
			colsp[idx] += (1.0f / Y);
		}

		// Zera a posição da IM para fazer a soma.
		mi[x0 + x1*X + z*X*X] = 0;

		// Varia entre os estados possíveis.
		for (n = 0; n < k*k; n++) {
			// Não se calcula log negativo, então ...
			if (colsp[n] != 0.0f && sitep[0 + (n / k) * 2] != 0.0f && sitep[1 + (n % k) * 2] != 0.0f) {
				mi[x0 + x1*X + z*X*X] += colsp[n] *
					log2f(colsp[n] / (sitep[0 + (n / k) * 2] * sitep[1 + (n % k) * 2]));
			}
		}
	}
}

/*!
*
* Informação Mútua mede a quantidade de informação que pode ser obtida sobre
* uma variável aleatória, observando outra. É importante em comunicação, onde
* ele pode ser utilizado para maximizar a quantidade de informação
* compartilhada entre os sinais enviados e recebidos.
*
* No contexto dos Automatos Celulares (AC) a Informação Mútua entre duas
* colunas da evolução temporal resultante da execução do AC oferece informação
* importante sobre o comportamento dinâmico da regra.
*
* A informação mútua da coluna X' da evolução temporal do AC em relação a outra
* coluna X" é dada por:
*
* I(X',X") = &Sigma;<SUB>x' &isin; X'</SUB>&Sigma;<SUB>x" &isin; X"</SUB> p(x',x")&sdot;log(p(x',x")/(p(x')p(x")))
*/
void MutualInformation(Memory<cell_t>& mat, Memory<float>& mi, cell_t k, int X, int Y, int Z) {
	mat.hostToDevice();
	mi.hostToDevice();

	MutualInformationDevice <<<Z, X>>> (mat.device(), mi.device(), k, Y);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	mi.deviceToHost();
}

/*!
*
* Informação Mútua mede a quantidade de informação que pode ser obtida sobre
* uma variável aleatória, observando outra. É importante em comunicação, onde
* ele pode ser utilizado para maximizar a quantidade de informação
* compartilhada entre os sinais enviados e recebidos.
*
* No contexto dos Automatos Celulares (AC) a Informação Mútua entre duas
* colunas da evolução temporal resultante da execução do AC oferece informação
* importante sobre o comportamento dinâmico da regra.
*
* A informação mútua da coluna X' da evolução temporal do AC em relação a outra
* coluna X" é dada por:
*
* I(X',X") = &Sigma;<SUB>x' &isin; X'</SUB>&Sigma;<SUB>x" &isin; X"</SUB> p(x',x")&sdot;log(p(x',x")/(p(x')p(x")))
*/
void MutualInformationHost(Memory<cell_t>& mat, Memory<float>& mi, cell_t k, int X, int Y, int Z) {
	// Define a matriz de probabilidade dos valores das colunas de mat.
	// São 2 colunas por k linhas para calcular a probabiliade 
	// de cada coluna x0 e x1.
	Memory<float> sitep(2 * MAX_NUMBER_OF_STATES);

	// A probabiliade conjunta das colunas X0 e x1 precisará de 
	// k*k linhas e uma coluna para cada par de culunas.
	Memory<float> colsp(MAX_NUMBER_OF_STATES * MAX_NUMBER_OF_STATES);

	size_t n;
	int idx;

	// Varia as matrizes.
	for (int z = 0; z < Z; z++) {
		// Varia a primeira coluna.
		for (int x0 = 0; x0 < X; x0++) {
			// Varia a segunda coluna.
			for (int x1 = 0; x1 < X; x1++) {
				// Zera os acumuladores.
				for (n = 0; n < static_cast<size_t>(2 * k); n++) sitep[n] = 0.0f;
				for (n = 0; n < static_cast<size_t>(k * k); n++) colsp[n] = 0.0f;

				// Quantifica os estados isolados de cada coluna em sitep 
				// e os estados conjuntos das colunas x0 e x1 em colsp
				for (int y = 0; y < Y; y++) {
					// Calcula o indice da primeira coluna para a linha 
					// do estado correspondente, o valor de mat na posição
					idx = 0 + mat[x0 + y*X + z*X*Y] * 2;
					// Incrementa o estado correspondente para a coluna x0
					sitep[idx] += (1.0f / Y);

					// Calcula o indice da segunda coluna para a linha 
					// do estado correspondente, o valor de mat na posição
					idx = 1 + mat[x1 + y*X + z*X*Y] * 2;
					// Incrementa o estado correspondente para a coluna x1
					sitep[idx] += (1.0f / Y);

					// Calcula o indice do estado conjunto representado pelo 
					// valor numérico de base k. A coluna x0 é a do dígito mais
					// significativo e a x1 o menos significativo.
					idx = mat[x0 + y*X + z*X*Y] * k
						+ mat[x1 + y*X + z*X*Y];
					// Incrementa o estado conjunto na linha correspondente
					colsp[idx] += (1.0f / Y);
				}

				// Zera a posição da IM para fazer a soma.
				mi[x0 + x1*X + z*X*X] = 0;

				// Varia entre os estados possíveis.
				for (n = 0; n < static_cast<size_t>(k*k); n++) {
					// Não se calcula log negativo, então ...
					if (colsp[n] != 0.0f && sitep[0 + (n / k) * 2] != 0.0f && sitep[1 + (n % k) * 2] != 0.0f) {
						mi[x0 + x1*X + z*X*X] += colsp[n] *
							log2f(colsp[n] / (sitep[0 + (n / k) * 2] * sitep[1 + (n % k) * 2]));
					}
				}
			}
		}
	}
}

/*!
* Calcula o grau de diferença entre as matrizes A e B. Retorna a taxa de diferença, valor entre 0 e 1, entre as matrizes.
*/
__global__ void SpreadingDevice(cell_t* mat, float* spreading, int X, int Y, int Z, int desZ0, int desZ1) {
	int z0 = threadIdx.x + desZ0 * blockDim.x;
	int z1 = threadIdx.y + desZ1 * blockDim.y;

	if (z0 >= Z || z1 >= Z) {
		return;
	}

	spreading[z0 + z1*Z] = 0.0f;

	for (int y = 0; y < Y; y++) {
		for (int x = 0; x < X; x++) {
			if (mat[x + y*X + z0*X*Y] != mat[x + y*X + z1*X*Y]) {
				spreading[z0 + z1*Z] += 1.0f;
			}
		}
	}

	spreading[z0 + z1*Z] /= static_cast<float>(X*Y);
}

void Spreading(Memory<cell_t>& mat, Memory<float>& spreading, int X, int Y, int Z) {
	Memory<float> colsum(X * Z * Z);

	mat.hostToDevice();
	spreading.hostToDevice();

	dim3 threds(30, 30);
	dim3 blocks(Z / 30, Z / 30);

	for (unsigned int desz0 = 0; desz0 < blocks.x; desz0++) {
		for (unsigned int desz1 = 0; desz1 < blocks.y; desz1++) {
			SpreadingDevice << <1, threds >> >(mat.device(), spreading.device(), X, Y, Z, desz0, desz1);

			checkCudaErrors(cudaGetLastError());
			checkCudaErrors(cudaDeviceSynchronize());
		}
	}

	spreading.deviceToHost();
}

void SpreadingHost(Memory<cell_t>& mat, Memory<float>& spreading, int X, int Y, int Z) {
	for (int z0 = 0; z0 < Z; z0++) {
		for (int z1 = 0; z1 < Z; z1++) {
			spreading[z0 + z1*Z] = 0.0f;

			for (int y = 0; y < Y; y++) {
				for (int x = 0; x < X; x++) {
					if (mat[x + y*X + z0*X*Y] != mat[x + y*X + z1*X*Y]) {
						spreading[z0 + z1*Z] += 1.0f;
					}
				}
			}

			spreading[z0 + z1*Z] /= static_cast<float>(X*Y);
		}
	}
}

float BinarySensibilityHost(rule_t n, float r) {
	cell_t k = 2;

	float result = 0.0f;

	// Quantidade de células na vizinhança
	size_t neighborhood_size = static_cast<size_t>(2 * r + 1);
	// Quantidade de vizinhanças possíveis
	size_t number_of_neighborhoods =
		static_cast<size_t>(pow(k, neighborhood_size));

	Memory<cell_t> neighborhood(2 * neighborhood_size);

	// Constroi a arvore de decisão para a regra.
	Tree ruleTree(r, n);

	// Preenche o reticulado com os valores das vizinhanças possíveis
	for (size_t non = 0; non < number_of_neighborhoods; non++) {
		size_t numero_b10 = non;

		for (size_t ns = neighborhood_size - 1; ns >= 0 && ns < neighborhood_size; ns--) {
			if (numero_b10 > 0) {
				neighborhood[ns] = numero_b10 % k;
			}
			else {
				neighborhood[ns] = 0;
			}

			numero_b10 /= k;
		}

		cell_t a = TransitionFunction(
			ruleTree, neighborhood.host(), r, neighborhood_size, r + 1);
		cell_t b;

		//
		for (size_t v = 0; v < neighborhood_size; v++) {
			neighborhood[v] = (1 - neighborhood[v])*(1 - neighborhood[v]);

			b = TransitionFunction(
				ruleTree, neighborhood.host(), r, neighborhood_size, r);

			if (a != b) {
				result += 1.0f;
			}

			neighborhood[v] = (1 - neighborhood[v])*(1 - neighborhood[v]);
		}
	}

	result /= (number_of_neighborhoods * neighborhood_size);

	return result;
}
