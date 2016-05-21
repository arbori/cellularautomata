#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cellularautomata.h"
#include "entropy.h"

#include <stdio.h>

int main() {
	rule_t n = 30;
	size_t r = 1;
	cell_t k = 2;
	size_t t = 300;

	env_t init(101);
	size_t pos = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 1;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;
	init.cells[pos++] = 0;

	env_t& ca = CellularAutomata(n, k, r, init, t);

	size_t size = ca.X * ca.Y;
	int* env = new int[size];
	float *entropy = new float[ca.X];

	for (size_t i = 0; i < size; i++) {
		env[i] = ca.cells[i];
	}

	// Add vectors in parallel.
	cudaError_t cudaStatus = BinarySiteEntropy(env, ca.X, ca.Y, entropy);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "BinarySiteEntropy failed!");
		return 1;
	}

	for (size_t i = 0; i < ca.Y; i++) {
		for (size_t j = 0; j < ca.X; j++) {
			printf("%f\t", entropy[i]);
		}
		printf("\n");
	}

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}

/*!
* A fun��o de transi��o tem como finalidade calcular o estado da
* c�lula i em t+1.
*
* \param tree Arvore de decis�o bin�ria, configurada com a regra do AC.
* \param env O ambiente do AC que deve ser levado para a nova configura��o de
* estados.
* \param r O raio utilizado para se definir a vizinhan�a.
* \param i C�lula central da vizinhan�a.
*/
cell_t TransitionFunction(tree_t& tree, cell_t* env, size_t r, size_t length,
	size_t i) {
	// Calcular o indice da primeira c�lula da vizinhan�a como o eslocamento �
	// esquerda a partir da c�lula central. Caso o deslocamento seja negativo,
	// soma-se o tamanho do reticulado para se obter o �ndice mais � direita do
	// reticulado. Assume-se aqui uma condi��o de contorno peri�dica.
	size_t pointer = i >= r ? (i - r) : length - r + i;

	// Guarda a posi��o na arvore de decis�o a partir do qual ser� feito o
	// deslocamento da arvore.
	int node = -1; //env[pointer];

	// Calcula quantas c�lulas h� no reticulado e decrementa o total a cada
	// deslocamento na arvore.
	for (size_t m = 2 * r + 1; m > 0; m--) {
		node = 2 * node + 2 + env[pointer++];

		if (pointer >= length) {
			pointer -= length;
		}
	}

	return tree.nodes[node];
}

/*!
* Cria uma matriz como resultado da evolu��o temporal do Automato Celular.
*
* \param n N�mero da regra.
* \param k N�mero de estados poss�veis.
* \param r Raio da vizinhan�a.
* \param init Condi��o inicial. Configura��o inicial do reticulado.
* \param width Tamanho do reticulado.
* \param t Quantidade de itera��es.
* \param transient Tamanho do per�odo transiente. Valor padr�o � 0.
*/
env_t& CellularAutomata(rule_t n, cell_t k, size_t r, env_t& init,
	size_t t, size_t transient) {
	//
	size_t height = t - transient + 1;
	env_t& result = *(new env_t(init.X, height));
	size_t y = 0;

	tree_t ruleTree(r, n);

	// Copia a condi��o inicial para a primeira linha.
	for (size_t x = 0; x < result.X; x++) {
		result.cells[x + y*result.X] = init.cells[x + y*init.X];
	}

	// Aplica a regra a cada linha do resultado do AC.
	for (y = 1; y < result.Y; y++) {
		for (size_t x = 0; x < result.X; x++) {
			result.cells[x + y*result.X] =
				TransitionFunction(
				ruleTree,
				&result.cells[0 + (y - 1)*result.X],
				r,
				result.X,
				x);
		}
	}

	return result;
}

/*!
* Cria um reticulado unidimencional.
*/
env_t::env_t(size_t x) {
	X = x;
	Y = 1;
	cells = new cell_t[X];
}

/*!
* Cria um reticulado bidimencional.
*/
env_t::env_t(size_t x, size_t y) {
	X = x;
	Y = y;
	cells = new cell_t[X*Y];
}

env_t::~env_t() {
	delete[] cells;
	cells = (cell_t*)0;
}

/*!
* Cria a estrutura da arvore bin�ria.
*/
tree_t::tree_t(size_t r) {
	size = treeSize(2 * r + 1);
	nodes = new cell_t[size];
}

/*!
* Cria a estrutura da arvore e preenche as folhas com o padr�o
* bin�rio da regra.
*/
tree_t::tree_t(size_t r, rule_t rule) {
	size = treeSize(r);
	nodes = new cell_t[size];

	// Decomp�e a regra em digitos bin�rios, guardando-os nas folhas da arvore.
	for (size_t i = size - pow(2, 2 * r + 1); i < size; i++) {
		nodes[i] = rule % 2;
		rule /= 2;
	}
}

/*!
* Libera recursos.
*/
tree_t::~tree_t() {
	delete[] nodes;
	nodes = (cell_t*) 0;
}

/*!
* Calcula o tamanho do array necess�rio para guardar a arvore de decis�o.
* Por ser bin�ria, cada n� da �rvore possui dois n�s filhos, o que permite
* calcular quantos n�s a arvore possui usando a soma Soma(2^i, {i, 1, m}).
* Onde m � o tamanho da vizinhan�a. Caso se queira representar o n� raiz,
* i deve come�ar em 0.
*/
size_t tree_t::treeSize(size_t r) {
	size_t size = 0;
	size_t sum = 2;
	size_t m = 2 * r + 1;

	size = 0;
	for (size_t x = 1; x <= m; x++) {
		size += sum;

		sum *= 2;
	}

	return size;
}

