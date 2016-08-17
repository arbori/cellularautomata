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
* A fun��o de transi��o tem como finalidade calcular o estado da
* c�lula i em t+1.
*
* \param tree Arvore de decis�o bin�ria, configurada com a regra do AC.
* \param env O ambiente do AC que deve ser levado para a nova configura��o de
* estados.
* \param r O raio utilizado para se definir a vizinhan�a.
* \param i C�lula central da vizinhan�a.
*/
cell_t TransitionFunction(Tree& tree, cell_t* env, float r, size_t length,
	size_t i) {
	// Calcular o indice da primeira c�lula da vizinhan�a como o eslocamento �
	// esquerda a partir da c�lula central. Caso o deslocamento seja negativo,
	// soma-se o tamanho do reticulado para se obter o �ndice mais � direita do
	// reticulado. Assume-se aqui uma condi��o de contorno peri�dica.
	size_t pointer = i >= r ? static_cast<size_t>(i - r) : static_cast<size_t>(length - r + i);

	// Guarda a posi��o na arvore de decis�o a partir do qual ser� feito o
	// deslocamento da arvore.
	int node = -1;

	// Calcula quantas c�lulas h� no reticulado e decrementa o total a cada
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
env_t* CellularAutomata(rule_t n, cell_t k, float r, env_t& init,
	size_t t, size_t transient) {
	// O tamanho do resultado leva em conta a evolu��o temporal mais uma linha
	// para a condi��o inicial.
	size_t height = t - transient + 1;
	env_t& result = *(new env_t(init.X, height));

	// Constroi a arvore de decis�o para a regra.
	Tree ruleTree(r, n);

	// Preenche a condi��o inicial
	for (size_t x = 0; x < init.X; x++) {
		result.cells[x] = init.cells[x];
	}

	// Vari�veis auxiliares para a transi��o de estados
	cell_t* tar;
	cell_t* src;

	// Para cada momento da do per�odo transiente, ...
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

		// Calcula a transi��o.
		ApplyRule(tar, src, ruleTree, result.X, r);
	}

	// Troca os valores da primeira linha pelos das segunda para que
	// a �ltima evolu��o temporal esteja na primeira linha.
	if (transient % 2 != 0) {
		src = &result.cells[result.X];
		tar = &result.cells[0];

		ApplyRule(tar, src, ruleTree, result.X, r);
	}

	// Ap�s o per�odo transiente, processa a evolu��o temporal.
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
	// O tamanho do resultado leva em conta a evolu��o temporal mais uma linha
	// para a condi��o inicial.
	size_t height = t - transient + 1;

	// Constroi a arvore de decis�o para a regra.
	Tree ruleTree(r, n);

	// Vari�veis auxiliares para a transi��o de estados
	cell_t* tar;
	cell_t* src;

	// Para cada momento da do per�odo transiente, ...
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

		// Calcula a transi��o.
		ApplyRule(tar, src, ruleTree, width, r);
	}

	// Troca os valores da primeira linha pelos das segunda para que
	// a �ltima evolu��o temporal esteja na primeira linha.
	if (transient % 2 != 0) {
		src = &ca[width];
		tar = &ca[0];

		ApplyRule(tar, src, ruleTree, width, r);
	}

	// Ap�s o per�odo transiente, processa a evolu��o temporal.
	for (size_t y = 1; y < height - 1; y++) {
		src = &ca[0 + (y - 1)*width];
		tar = &ca[0 + y*width];

		ApplyRule(tar, src, ruleTree, width, r);
	}
}
