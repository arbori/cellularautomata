#pragma once

#include "Types.h"
#include "Memory.h"
#include "Tree.h"

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
*/
void StartRandom();

/*!
*/
void RandomInteger(Memory<cell_t>& line, int length);

/*!
 */
float Mean(float* values, int X, int Y);

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
	size_t t, size_t transient = 0);

/*!
*
*/
void CellularAutomata(rule_t n, cell_t k, float r, Memory<cell_t>& ca, size_t width,
	size_t t, size_t transient = 0);
