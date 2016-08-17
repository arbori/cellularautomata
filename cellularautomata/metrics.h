#pragma once

#include "cellularautomata.h"
#include "Memory.h"

/*!
*
*/
void Entropy(Memory<cell_t>& mat, Memory<float>& entropy, int X, int Y, int Z, float r, cell_t k);
void EntropyHost(Memory<cell_t>& mat, Memory<float>& entropy, int X, int Y, int Z, float r, cell_t k);

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
void MutualInformation(Memory<cell_t>& mat, Memory<float>& mi, cell_t k, int X, int Y, int Z);
void MutualInformationHost(Memory<cell_t>& mat, Memory<float>& mi, cell_t k, int X, int Y, int Z);

/*!
* Calcula o grau de diferença entre as matrizes A e B. Retorna a taxa de diferença, valor entre 0 e 1, entre as matrizes.
*/
void Spreading(Memory<cell_t>& mat, Memory<float>& spreading, int X, int Y, int Z);
void SpreadingHost(Memory<cell_t>& mat, Memory<float>& spreading, int X, int Y, int Z);
