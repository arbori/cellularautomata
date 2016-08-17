#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cellularautomata.h"

#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

using namespace std;

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
}

///////////////////////////////////////////////////////////////////////////////
/*!
*/
void StartRandom() {
	srand(time(NULL));
}

/*!
*/
void RandomInteger(Memory<cell_t>& line, int length) {
	for (int x = 0; x < length; x++) {
		line[x] = rand() % 2;
	}
}

/*!
 */
float Mean(float* values, int X, int Y) {
	float result = 0.0f;

	for (int y = 0; y < Y; y++) {
		for (int x = 0; x < X; x++) {
			result += values[x + y*X];
		}
	}

	result /= (X * Y);

	return result;
}


/*!
* A função de transição tem como finalidade calcular o estado da
* célula i em t+1.
*
* \param tree Arvore de decisão binária, configurada com a regra do AC.
* \param env O ambiente do AC que deve ser levado para a nova configuração de
* estados.
* \param r O raio utilizado para se definir a vizinhança.
* \param i Célula central da vizinhança.
*/
cell_t TransitionFunction(Tree& tree, cell_t* env, float r, size_t length,
	size_t i) {
	// Calcular o indice da primeira célula da vizinhança como o eslocamento à
	// esquerda a partir da célula central. Caso o deslocamento seja negativo,
	// soma-se o tamanho do reticulado para se obter o índice mais à direita do
	// reticulado. Assume-se aqui uma condição de contorno periódica.
	size_t pointer = i >= r ? static_cast<size_t>(i - r) : static_cast<size_t>(length - r + i);

	// Guarda a posição na arvore de decisão a partir do qual será feito o
	// deslocamento da arvore.
	int node = -1;

	// Calcula quantas células há no reticulado e decrementa o total a cada
	// deslocamento na arvore.
	for (size_t m = static_cast<size_t>(2 * r + 1); m > 0; m--) {
		node = 2 * node + 2 + env[pointer++];

		if (pointer >= length) {
			pointer -= length;
		}
	}

	return tree[node];
}

/*!
*
*/
inline void ApplyRule(cell_t* tar, cell_t* src, Tree& ruleTree, size_t X, float r) {
	for (size_t x = 0; x < X; x++) {
		tar[x] = TransitionFunction(ruleTree, src, r, X, x);
	}
}

/*!
* Cria uma matriz como resultado da evolução temporal do Automato Celular.
*
* \param n Número da regra.
* \param k Número de estados possíveis.
* \param r Raio da vizinhança.
* \param init Condição inicial. Configuração inicial do reticulado.
* \param width Tamanho do reticulado.
* \param t Quantidade de iterações.
* \param transient Tamanho do período transiente. Valor padrão é 0.
*/
env_t* CellularAutomata(rule_t n, cell_t k, float r, env_t& init,
	size_t t, size_t transient) {
	// O tamanho do resultado leva em conta a evolução temporal mais uma linha
	// para a condição inicial.
	size_t height = t - transient + 1;
	env_t& result = *(new env_t(init.X, height));

	// Constroi a arvore de decisão para a regra.
	Tree ruleTree(r, n);

	// Preenche a condição inicial
	for (size_t x = 0; x < init.X; x++) {
		result.cells[x] = init.cells[x];
	}

	// Variáveis auxiliares para a transição de estados
	cell_t* tar;
	cell_t* src;

	// Para cada momento da do período transiente, ...
	for (size_t y = 0; y < transient; y++) {
		// ... se a momento for par, guarda o resultado na segunda linha.
		if (y % 2 == 0) {
			src = &result.cells[0];
			tar = &result.cells[result.X];
		}
		// Se a momento for impar, guarda o resultado na primeira linha.
		else {
			src = &result.cells[result.X];
			tar = &result.cells[0];
		}

		// Calcula a transição.
		ApplyRule(tar, src, ruleTree, result.X, r);
	}

	// Troca os valores da primeira linha pelos das segunda para que
	// a última evolução temporal esteja na primeira linha.
	if (transient % 2 != 0) {
		src = &result.cells[result.X];
		tar = &result.cells[0];

		ApplyRule(tar, src, ruleTree, result.X, r);
	}

	// Após o período transiente, processa a evolução temporal.
	for (size_t y = 0; y < t - transient; y++) {
		src = &result.cells[0 + y*result.Y];
		tar = &result.cells[0 + (y + 1)*result.Y];

		ApplyRule(tar, src, ruleTree, result.X, r);
	}

	return &result;
}


/*!
*
*/
void CellularAutomata(rule_t n, cell_t k, float r, Memory<cell_t>& ca, size_t width, size_t t, size_t transient) {
	// O tamanho do resultado leva em conta a evolução temporal mais uma linha
	// para a condição inicial.
	size_t height = t - transient + 1;

	// Constroi a arvore de decisão para a regra.
	Tree ruleTree(r, n);

	// Variáveis auxiliares para a transição de estados
	cell_t* tar;
	cell_t* src;

	// Para cada momento da do período transiente, ...
	for (size_t y = 0; y < transient; y++) {
		// ... se a momento for par, guarda o resultado na segunda linha.
		if (y % 2 == 0) {
			src = &ca[0];
			tar = &ca[width];
		}
		// Se a momento for impar, guarda o resultado na primeira linha.
		else {
			src = &ca[width];
			tar = &ca[0];
		}

		// Calcula a transição.
		ApplyRule(tar, src, ruleTree, width, r);
	}

	// Troca os valores da primeira linha pelos das segunda para que
	// a última evolução temporal esteja na primeira linha.
	if (transient % 2 != 0) {
		src = &ca[width];
		tar = &ca[0];

		ApplyRule(tar, src, ruleTree, width, r);
	}

	// Após o período transiente, processa a evolução temporal.
	for (size_t y = 1; y < height - 1; y++) {
		src = &ca[0 + (y - 1)*width];
		tar = &ca[0 + y*width];

		ApplyRule(tar, src, ruleTree, width, r);
	}
}
