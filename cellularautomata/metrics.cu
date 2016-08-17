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

using namespace std;

#define MAX_NUMBER_OF_STATES 8

#define MAX_R 12
#define MAX_NEIGHBORHOOD_SIZE 25
#define MAX_MI_NUMBER (MAX_R*MAX_R)

/*!
*
*/
__host__ __device__ int FromDigits(cell_t* line, int ini, int fim, cell_t k) {
	int value = 0;
	int bits = (fim - ini);

	for (int n = bits; n > 0; n--) {
		value += static_cast<int>(powf(k, bits - n)) * line[fim - (bits - n) - 1];
	}

	return value;
}

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
		result += line[ini++] * powf(k, potencia--);

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
__global__ void EntropyDevice(cell_t* mat, float* entropy, int X, int Y, int Z, float r, cell_t k, float* states, int size) {
	int x = threadIdx.x;
	int z = blockIdx.x;

	if (x >= X || z >= Z) {
		return;
	}

	int s = 0;

	for (int y = 0; y < Y; y++) {
		s = NeighborhoodNumber(&mat[0 + y*X + z*X*Y], X, x, r, k);

		states[x + s*X + z*X*size] += 1.0;
	}

	entropy[x + z*X] = 0.0f;

	for (s = 0; s < size; s++) {
		if (states[x + s*X + z*X*size] != 0.0f) {
			states[x + s*X + z*X*size] /= Y;

			entropy[x + z*X] += -static_cast<float>(
				states[x + s*X + z*X*size] * 
				log2f(states[x + s*X + z*X*size])
			);
		}
	}
}

/*!
*
*/
void Entropy(Memory<cell_t>& mat, Memory<float>& entropy, int X, int Y, int Z, float r, cell_t k)
{
	int size = static_cast<int>(X * powf(k, 2 * r + 1) * Z);
	Memory<float> states(size);

	states.assignAll(0);
	states.hostToDevice();

	mat.hostToDevice();

	EntropyDevice <<<Z, X>>>(mat.device(), entropy.device(), X, Y, Z,
		r, k, states.device(), static_cast<int>(powf(k, 2 * r + 1)));

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	entropy.deviceToHost();
}

/*!
*
*/
void EntropyHost(Memory<cell_t>& mat, Memory<float>& entropy, int X, int Y, int Z, float r, cell_t k) {
	int size = static_cast<int>(powf(k, 2 * r + 1));
	Memory<float> states(size);

	int s = 0;

	for (int z = 0; z < Z; z++) {
		for (int x = 0; x < X; x++) {
			for (int n = 0; n < size; n++) {
				states[n] = 0.0f;
			}

			for (int y = 0; y < Y; y++) {
				s = NeighborhoodNumber(&mat[0 + y*X + z*X*Y], X, x, r, k);

				states[s] += 1.0;
			}

			entropy[x + z*X] = 0.0f;

			for (int i = 0; i < size; i++) {
				if (states[i] != 0.0f) {
					states[i] /= Y;

					entropy[x + z*X] += -static_cast<float>(
						states[i] * log2f(states[i]) );
				}
			}
		}

		for (int x = 0; x < X; x++) {
			cout << entropy[x] << ";";
		}

		cout << endl;

	}
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

	for (int desz0 = 0; desz0 < blocks.x; desz0++) {
		for (int desz1 = 0; desz1 < blocks.y; desz1++) {
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

void SensibilityHost(Memory<cell_t>& mat, Memory<float>& sensibility, int X, int Y, int Z) {
	srand(time(NULL));

	int x;
	int estado;
	Memory<cell_t> variacoes(mat.size());

	for (int i = 0; i < mat.size(); i++) {
		variacoes[i] = mat[i];
	}

	for (int z = 0; z < Z; z++) {
		x = rand() % Y;

		estado = mat[x + 0 * X + z*X*Y];
		mat[x + 0 * X + z*X*Y] = (estado - 1)*(estado - 1);


	}
}
