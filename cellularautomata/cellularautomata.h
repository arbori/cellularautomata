#pragma once

typedef long long rule_t;

typedef unsigned short cell_t;

class env_t {
public:
	size_t X;
	size_t Y;
	cell_t* cells;

	/*!
	* Cria um reticulado unidimencional.
	*/
	env_t(size_t x);

	/*!
	* Cria um reticulado bidimencional.
	*/
	env_t(size_t x, size_t y);

	~env_t();
};

class tree_t {
public:
	size_t size;
	cell_t* nodes;

	/*!
	* Cria a estrutura da arvore bin�ria.
	*/
	tree_t(size_t r);

	/*!
	* Cria a estrutura da arvore e preenche as folhas com o padr�o
	* bin�rio da regra.
	*/
	tree_t(size_t r, rule_t rule);

	/*!
	* Libera recursos.
	*/
	~tree_t();

private:
	/*!
	* Calcula o tamanho do array necess�rio para guardar a arvore de decis�o.
	* Por ser bin�ria, cada n� da �rvore possui dois n�s filhos, o que permite
	* calcular quantos n�s a arvore possui usando a soma Soma(2^i, {i, 1, m}).
	* Onde m � o tamanho da vizinhan�a. Caso se queira representar o n� raiz,
	* i deve come�ar em 0.
	*/
	size_t treeSize(size_t r);
};

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
	size_t i);

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
	size_t t, size_t transient = 0);