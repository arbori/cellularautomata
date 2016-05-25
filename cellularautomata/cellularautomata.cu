#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cellularautomata.h"
#include "entropy.h"

#include <iostream>
#include <cmath>

using namespace std;

int main() {
	rule_t n = 30;
	size_t r = 1;
	cell_t k = 2;
	size_t t = 8;

	env_t init(7);
	size_t pos = 0;
	init.cells[pos++] = 1;
	init.cells[pos++] = 0;
	init.cells[pos++] = 1;
	init.cells[pos++] = 0;
	init.cells[pos++] = 1;
	init.cells[pos++] = 1;
	init.cells[pos++] = 0;

	env_t& ca = CellularAutomata(n, k, r, init, t);

	float* mi = MutualInformation(ca, k);

	for (size_t x0 = 0; x0 < ca.X; x0++) {
		for (size_t x1 = 0; x1 < ca.X; x1++) {
			cout << mi[x0 + x1*ca.X] << "	";
		}

		cout << endl;
	}

	cout << endl;

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
	for (size_t i = size - static_cast<size_t>(pow(2, 2 * r + 1)); i < size; i++) {
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


/*!
*
* Informa��o M�tua mede a quantidade de informa��o que pode ser obtida sobre
* uma vari�vel aleat�ria, observando outra. � importante em comunica��o, onde
* ele pode ser utilizado para maximizar a quantidade de informa��o
* compartilhada entre os sinais enviados e recebidos.
*
* No contexto dos Automatos Celulares (AC) a Informa��o M�tua entre duas
* colunas da evolu��o temporal resultante da execu��o do AC oferece informa��o
* importante sobre o comportamento din�mico da regra.
*
* A informa��o m�tua da coluna X' da evolu��o temporal do AC em rela��o a outra
* coluna X" � dada por:
*
* I(X',X") = &Sigma;<SUB>x' &isin; X'</SUB>&Sigma;<SUB>x" &isin; X"</SUB> p(x',x")&sdot;log(p(x',x")/(p(x')p(x")))
*/
float* MutualInformation(env_t& m, cell_t k) {
	size_t x, y;

	// Define a matriz de probabilidade local da matriz m.
	float *sitep = new float[m.X * k];

	// Calcula a probabilidade local de cada estado de cada coluna.
	for (x = 0; x < m.X; x++) {
		// Zera a coluna da matriz de probabilidade local
		for (y = 0; y < k; y++) {
			sitep[x + y*m.X] = 0.0f;
		}
		// Acumula a quantidade de cada estado na coluna x.
		for (y = 0; y < m.Y; y++) {
			sitep[x + m.cells[x + y*m.X] * m.X] += (1.0f / m.Y);
		}
	}

	// O c�lculo da informa��o m�tua � feito sobre todos os pares de colunas
	// poss�veis. Ent�o ser� calculado a informa��o m�tual das colunas (0,0),
	// (1,0), (2,0), ... (X,0). Para cada par de coluna � calculado um valor de
	// informa��o m�tua, que ser�o guardados na primeira coluna da matriz
	// resultante. Na sequ�ncia ser�o calculadas as informa��es m�tual para os
	// pares (0,1), (1,1), ..., (X,1) e ser�o guardados na segunda colna da
	// matriz resultante.
	//
	// Isso mostra que a matriz resultante do c�lculo da Informa��o M�tua � uma
	// matriz quadrada X por X.
	float *result = new float[m.X * m.X];
	for (size_t n = 0; n < m.X*m.X; result[n++] = 0.0f);

	// Define o vetor de probabilidade dos valores do par de colunas.
	// O n�mero de elementos do vetor � k^ncolunas. No caso, com s�o duas
	// colunas, k^2.
	float *colsp = new float[(k*k)];
	// Indice do vetor colsp calculada pelo valor das colunas
	size_t i;

	float divisor;
	cell_t cellv0, cellv1;

	// Varia a primeira coluna da informa��o m�tua.
	for (size_t x0 = 0; x0 < m.X; x0++) {
		// Varia a segunda coluna da informa��o m�tua.
		for (size_t x1 = 0; x1 < m.X; x1++) {
			// Zera o vetor para fazer a contagem.
			for (i = 0; i < (size_t)k*k; i++) {
				colsp[i] = 0.0f;
			}

			// Calcula a probabilidade dos valores conjuntos das colunas
			// (x0, x1)
			for (y = 0; y < m.Y; y++) {
				// Convers�o de base para decimal.
				i = m.cells[x0 + y*m.X] * k + m.cells[x0 + y*m.X];

				colsp[i] += (1.0f / m.Y);
			}

			for (y = 0; y < m.Y; y++) {
				i = m.cells[x0 + y*m.X] * k + m.cells[x0 + y*m.X];
				cellv0 = m.cells[x0 + y*m.X];
				cellv1 = m.cells[x1 + y*m.X];

				divisor = sitep[x0 + cellv0*m.X] * sitep[x1 + cellv1*m.X];

				result[x0 + x1*m.X] += colsp[i] * log(colsp[i] / divisor);
			}
		}
	}

	delete[] sitep;
	delete[] colsp;

	return result;
}
