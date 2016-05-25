#pragma once

typedef long long rule_t;

typedef unsigned short cell_t;

/*!
 *
 */
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

/*!
 *
 */
class tree_t {
public:
	size_t size;
	cell_t* nodes;

	/*!
	* Cria a estrutura da arvore binária.
	*/
	tree_t(size_t r);

	/*!
	* Cria a estrutura da arvore e preenche as folhas com o padrão
	* binário da regra.
	*/
	tree_t(size_t r, rule_t rule);

	/*!
	* Libera recursos.
	*/
	~tree_t();

private:
	/*!
	* Calcula o tamanho do array necessário para guardar a arvore de decisão.
	* Por ser binária, cada nó da árvore possui dois nós filhos, o que permite
	* calcular quantos nós a arvore possui usando a soma Soma(2^i, {i, 1, m}).
	* Onde m é o tamanho da vizinhança. Caso se queira representar o nó raiz,
	* i deve começar em 0.
	*/
	size_t treeSize(size_t r);
};

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
//cell_t TransitionFunction(tree_t& tree, cell_t* env, size_t r, size_t length,
//	size_t i);

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
env_t& CellularAutomata(rule_t n, cell_t k, size_t r, env_t& init,
	size_t t, size_t transient = 0);

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
float* MutualInformation(env_t& m, cell_t k);
