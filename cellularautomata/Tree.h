#pragma once

#include "Types.h"
#include "Memory.h"

#include <cmath>

/*!
*
*/
class Tree : public Memory<cell_t> {
public:
	/*!
	* Cria a estrutura da arvore e preenche as folhas com o padrão
	* binário da regra.
	*/
	Tree(float r, rule_t rule) : Memory<cell_t>(treeSize(r)) {
		// Decompõe a regra em digitos binários, guardando-os nas folhas da arvore.
		for (size_t i = size() - static_cast<size_t>(std::powl(2, 2 * r + 1)); i < size(); i++) {
			operator[](i) = rule % 2;
			rule /= 2;
		}
	}

private:
	/*!
	* Calcula o tamanho do array necessário para guardar a arvore de decisão.
	* Por ser binária, cada nó da árvore possui dois nós filhos, o que permite
	* calcular quantos nós a arvore possui usando a soma Soma(2^i, {i, 1, m}).
	* Onde m é o tamanho da vizinhança. Caso se queira representar o nó raiz,
	* i deve começar em 0.
	*/
	size_t treeSize(float r) {
		size_t size = 0;
		size_t sum = 2;
		size_t m = static_cast<size_t>(2 * r + 1);

		size = 0;
		for (size_t x = 1; x <= m; x++) {
			size += sum;

			sum *= 2;
		}

		return size;
	}
};

