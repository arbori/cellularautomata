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
void MutualInformation(Memory<cell_t>& mat, Memory<float>& mi, cell_t k, int X, int Y, int Z);
void MutualInformationHost(Memory<cell_t>& mat, Memory<float>& mi, cell_t k, int X, int Y, int Z);

/*!
* Calcula o grau de diferen�a entre as matrizes A e B. Retorna a taxa de diferen�a, valor entre 0 e 1, entre as matrizes.
*/
void Spreading(Memory<cell_t>& mat, Memory<float>& spreading, int X, int Y, int Z);
void SpreadingHost(Memory<cell_t>& mat, Memory<float>& spreading, int X, int Y, int Z);
