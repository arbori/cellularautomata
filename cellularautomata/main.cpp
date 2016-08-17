#include "cellularautomata.h"
#include "metrics.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <ctime>       /* clock_t, clock, CLOCKS_PER_SEC */

using namespace std;

void readRules(string& filename, Memory<rule_t>& rules) {
	ifstream file(filename.c_str());
	char value[8];

	if (file.is_open())
	{
		for (size_t i = 0; i < rules.size(); ++i)
		{
			file >> value;

			rules[i] = atol(value);
		}
	}

	file.close();
}

void readMatrix(ifstream& in, Memory<cell_t>& ca, int X, int Y) {
	string state;

	if (in.is_open())
	{
		for (int i = 0; i < Y; ++i)
		{
			for (int j = 0; j < X; ++j)
			{
				in >> state;

				ca[j + i*X] = (state.c_str()[0] - '0');
			}
		}
	}
}

void testeEntropy(Memory<cell_t>& ca, int X, int Y, int Z = 1) {
	Memory<float> entropy(X * Z);

	Entropy(ca, entropy, X, Y, Z, 0, 2);
	for (int z = 0; z < Z; z++) {
		for (int x = 0; x < X; x++) {
			cout << entropy[x + z*X] << ";";
		}

		cout << endl;
	}

	cout << endl;

	Entropy(ca, entropy, X, Y, Z, 1, 2);
	for (int z = 0; z < Z; z++) {
		for (int x = 0; x < X; x++) {
			cout << entropy[x + z*X] << ";";
		}

		cout << endl;
	}

	cout << endl;
}

void testPerformace() {
	float dtdev, dthost;
	clock_t ini;
	int EXPERIENCIAS = 1000;

	int X = 16;
	int Y = 32;
	int Z = 3;
	int length = X * Y * Z;
	Memory<cell_t> ca(length);
	Memory<float> siteEntropy(X * Z);

	RandomInteger(ca, length);

	cout << "Inicio da avaliacao da performace..." << endl;

	ini = clock();
	for (int i = 0; i < EXPERIENCIAS; i++) {
		Entropy(ca, siteEntropy, X, Y, Z, 1, 2);
	}
	dtdev = static_cast<float>(clock() - ini) / CLOCKS_PER_SEC;
	cout << "Entropy: " << dtdev << "s" << endl;

	ini = clock();
	for (int i = 0; i < EXPERIENCIAS; i++) {
		EntropyHost(ca, siteEntropy, X, Y, Z, 1, 2);
	}
	dthost = static_cast<float>(clock() - ini) / CLOCKS_PER_SEC;
	cout << "EntropyHost: " << dthost << "s" << endl;

	cout << "Entropy / EntropyHost: " << dtdev / dthost << endl;
}

void testeMutualInformation() {
	int X = 8;
	int Y = 8;
	int Z = 3;

	int k = 2;

	Memory<cell_t> mat(X * Y * Z);
	Memory<float> mi(X * X * Z);

	int i = 0;
	mat[i++] = 1; mat[i++] = 1; mat[i++] = 0; mat[i++] = 0; mat[i++] = 1; mat[i++] = 0; mat[i++] = 0; mat[i++] = 0;
	mat[i++] = 1; mat[i++] = 0; mat[i++] = 1; mat[i++] = 0; mat[i++] = 1; mat[i++] = 0; mat[i++] = 0; mat[i++] = 1;
	mat[i++] = 1; mat[i++] = 0; mat[i++] = 1; mat[i++] = 0; mat[i++] = 1; mat[i++] = 1; mat[i++] = 1; mat[i++] = 0;
	mat[i++] = 1; mat[i++] = 0; mat[i++] = 1; mat[i++] = 0; mat[i++] = 0; mat[i++] = 1; mat[i++] = 1; mat[i++] = 1;
	mat[i++] = 0; mat[i++] = 0; mat[i++] = 0; mat[i++] = 0; mat[i++] = 0; mat[i++] = 0; mat[i++] = 0; mat[i++] = 1;
	mat[i++] = 1; mat[i++] = 1; mat[i++] = 0; mat[i++] = 0; mat[i++] = 0; mat[i++] = 0; mat[i++] = 0; mat[i++] = 0;
	mat[i++] = 0; mat[i++] = 0; mat[i++] = 1; mat[i++] = 1; mat[i++] = 1; mat[i++] = 1; mat[i++] = 1; mat[i++] = 0;
	mat[i++] = 0; mat[i++] = 0; mat[i++] = 0; mat[i++] = 1; mat[i++] = 1; mat[i++] = 1; mat[i++] = 1; mat[i++] = 0;
	mat[i++] = 1; mat[i++] = 0; mat[i++] = 0; mat[i++] = 1; mat[i++] = 0; mat[i++] = 1; mat[i++] = 0; mat[i++] = 1;
	mat[i++] = 0; mat[i++] = 1; mat[i++] = 1; mat[i++] = 0; mat[i++] = 0; mat[i++] = 1; mat[i++] = 0; mat[i++] = 1;
	mat[i++] = 1; mat[i++] = 1; mat[i++] = 0; mat[i++] = 0; mat[i++] = 1; mat[i++] = 1; mat[i++] = 0; mat[i++] = 0;
	mat[i++] = 1; mat[i++] = 1; mat[i++] = 1; mat[i++] = 0; mat[i++] = 0; mat[i++] = 1; mat[i++] = 0; mat[i++] = 0;
	mat[i++] = 1; mat[i++] = 0; mat[i++] = 0; mat[i++] = 1; mat[i++] = 0; mat[i++] = 0; mat[i++] = 1; mat[i++] = 0;
	mat[i++] = 0; mat[i++] = 0; mat[i++] = 0; mat[i++] = 1; mat[i++] = 0; mat[i++] = 0; mat[i++] = 1; mat[i++] = 0;
	mat[i++] = 1; mat[i++] = 0; mat[i++] = 1; mat[i++] = 0; mat[i++] = 0; mat[i++] = 1; mat[i++] = 0; mat[i++] = 0;
	mat[i++] = 0; mat[i++] = 0; mat[i++] = 0; mat[i++] = 1; mat[i++] = 0; mat[i++] = 1; mat[i++] = 1; mat[i++] = 1;
	mat[i++] = 0; mat[i++] = 1; mat[i++] = 1; mat[i++] = 0; mat[i++] = 0; mat[i++] = 0; mat[i++] = 0; mat[i++] = 1;
	mat[i++] = 0; mat[i++] = 0; mat[i++] = 1; mat[i++] = 0; mat[i++] = 0; mat[i++] = 1; mat[i++] = 0; mat[i++] = 0;
	mat[i++] = 1; mat[i++] = 0; mat[i++] = 0; mat[i++] = 1; mat[i++] = 1; mat[i++] = 1; mat[i++] = 0; mat[i++] = 0;
	mat[i++] = 1; mat[i++] = 1; mat[i++] = 1; mat[i++] = 0; mat[i++] = 0; mat[i++] = 1; mat[i++] = 0; mat[i++] = 0;
	mat[i++] = 1; mat[i++] = 0; mat[i++] = 0; mat[i++] = 1; mat[i++] = 0; mat[i++] = 0; mat[i++] = 1; mat[i++] = 0;
	mat[i++] = 1; mat[i++] = 0; mat[i++] = 0; mat[i++] = 1; mat[i++] = 0; mat[i++] = 0; mat[i++] = 1; mat[i++] = 0;
	mat[i++] = 0; mat[i++] = 1; mat[i++] = 0; mat[i++] = 0; mat[i++] = 0; mat[i++] = 1; mat[i++] = 1; mat[i++] = 0;
	mat[i++] = 0; mat[i++] = 1; mat[i++] = 0; mat[i++] = 1; mat[i++] = 0; mat[i++] = 0; mat[i++] = 0; mat[i++] = 1;

	MutualInformationHost(mat, mi, k, X, Y, Z);

	for (int z = 0; z < Z; z++) {
		for (int x0 = 0; x0 < X; x0++) {
			for (int x1 = 0; x1 < X; x1++) {
				cout << mi.host()[x0 + x1*X + z*X*Y] << ";";
			}

			cout << endl;
		}

		cout << "\n\n";
	}

	cout << endl;
}

void testeSpreading(string& filedata, int X, int Y, int Z) {
	Memory<cell_t> ca(X * Y * Z);
	Memory<float> spreadingHost(Z * Z);
	Memory<float> spreading(Z * Z);

	ifstream in(filedata.c_str());

	if (in.is_open()) {
		readMatrix(in, ca, X, Y*Z);

		SpreadingHost(ca, spreadingHost, X, Y, Z);

		for (int z0 = 0; z0 < Z; z0++) {
			for (int z1 = 0; z1 < Z; z1++) {
				cout << spreadingHost[z0 + z1*Z] << ";";
			}

			cout << "\n";
		}

		cout << "\n\n";

		Spreading(ca, spreading, X, Y, Z);

		for (int z0 = 0; z0 < Z; z0++) {
			for (int z1 = 0; z1 < Z; z1++) {
				cout << spreading[z0 + z1*Z] << ";";
			}

			cout << "\n";
		}

		cout << endl;
	}
	else {
		cout << "Nao foi possivel abrir o arquivo " << filedata << endl;
	}
}

bool salvarRegra(string& filename, rule_t n, cell_t k, float r, size_t t, size_t transient, size_t width, size_t amostras,
		Memory<cell_t>& init, Memory<float>& siteEntropy, Memory<float>& entropy, Memory<float>& mi, Memory<float>& spreading) {
	ofstream fout;
	
	int X = width;
	int Y = t - transient;
	int Z = amostras;

	fout.open(filename.c_str());

	if (!fout.is_open() || fout.bad() || fout.fail()) {
		return false;
	}

	fout.precision(12);

	fout << "{\n\t{" 
			<< n << "," << k << "," << r << "," << t << "," 
			<< transient << "," << width << "," << amostras 
		<< "},\n\t{";

	for (int z = 0; z < Z; z++) {
		if (z == 0) {
			fout << "\n";
		}

		fout << "\t\t{";
		for (int x = 0; x < X; x++) {
			fout << init[x + z*X];

			if (x < X - 1) {
				fout << ",";
			}
		}

		fout << "}";

		if (z < Z - 1) {
			fout << ",\n";
		}
		else {
			fout << "\n";
		}
	}

	fout.setf(ios_base::fixed, ios_base::floatfield);

	fout << "\t},\n\t{";
	for (int z = 0; z < Z; z++) {
		if (z == 0) {
			fout << "\n";
		}

		fout << "\t\t{";
		for (int x = 0; x < X; x++) {
			fout << siteEntropy[x + z*X];

			if (x < X - 1) {
				fout << ",";
			}
		}

		fout << "}";

		if (z < Z - 1) {
			fout << ",\n";
		}
		else {
			fout << "\n";
		}
	}

	fout << "\t},\n\t{";

	for (int z = 0; z < Z; z++) {
		if (z == 0) {
			fout << "\n";
		}

		fout << "\t\t{";
		for (int x = 0; x < X; x++) {
			fout << entropy[x + z*X];

			if (x < X - 1) {
				fout << ",";
			}
		}

		fout << "}";

		if (z < Z - 1) {
			fout << ",\n";
		}
		else {
			fout << "\n";
		}
	}

	fout << "\t},\n\t{";

	for (int x1 = 0; x1 < X; x1++) {
		if (x1 == 0) {
			fout << "\n";
		}

		fout << "\t\t{";
		for (int x0 = 0; x0 < X; x0++) {
			fout << mi[x0 + x1*X];

			if (x0 < X - 1) {
				fout << ",";
			}
		}

		fout << "}";

		if (x1 < X - 1) {
			fout << ",\n";
		}
		else {
			fout << "\n";
		}
	}

	fout << "\t},\n\t{";

	for (int z1 = 0; z1 < Z; z1++) {
		if (z1 == 0) {
			fout << "\n";
		}

		fout << "\t\t{";
		for (int z0 = 0; z0 < Z; z0++) {
			fout << spreading[z0 + z1*Z];

			if (z0 < Z - 1) {
				fout << ",";
			}
		}

		fout << "}";

		if (z1 < Z - 1) {
			fout << ",\n";
		}
		else {
			fout << "\n";
		}

		fout.setf(0, ios_base::floatfield);
	}

	fout << "\t}\n}";

	fout.flush();
	fout.close();

	return true;
}

void calculoExperimento(string& dir, Memory<rule_t>& rules, cell_t k, float r, size_t t, size_t transient, size_t width, size_t amostras) {
	StartRandom();

	size_t height = t - transient;

	Memory<cell_t> mat(width * height * amostras);
	Memory<cell_t> init(width * amostras);
	Memory<float> siteEntropy(width * amostras);
	Memory<float> entropy(width * amostras);
	Memory<float> mi(width * width * amostras);
	Memory<float> spreading(amostras * amostras);

	std::setprecision(4);
	char buffer[8];

	for (size_t n = 0; n < rules.size(); n++) {
		itoa(rules[n], buffer, 10);

		string filename = dir;
		filename += "\\regra-";
		filename += buffer;
		filename += ".txt";

		cout << "Regra " << rules[n] << endl;

		for (size_t amostra = 0; amostra < amostras; amostra++) {
			mat.offset(width * height * amostra);

			RandomInteger(mat, width);
			for (size_t w = 0; w < width; w++) {
				init[w + amostra * width] = mat[w];
			}

			CellularAutomata(rules[n], k, r, mat, width, t, transient);

			cout << "\b\b\b\b\b\b\b\b\b\b" << (static_cast<float>(100 * amostra) / static_cast<float>(amostras)) << "%";
		}

		mat.offset(0);

		cout << "\b\b\b\b\b\b\b\b\b\b          \b\b\b\b\b\b\b\b\b\b";

		// Calcula a entropia local
		cout << "Calculando a Entropia Local...";
		Entropy(mat, siteEntropy, width, height, amostras, 0, k);
		cout << "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b"
			<< "                              "
			<< "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b";

		// Entropia de vizinhança
		cout << "Calculando a Entropia de viainhanca...";
		Entropy(mat, entropy, width, height, amostras, r, k);
		cout << "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b"
			<< "                                      "
			<< "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b";

		// Informação Mútua
		cout << "Calculando a Informacao Mutua...";
		MutualInformation(mat, mi, k, width, height, amostras);
		cout << "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b" 
			<< "                                "
			<< "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b";

		// Espalhamento
		cout << "Calculando a Espalhamento...";
		Spreading(mat, spreading, width, height, amostras);
		cout << "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b"
			<< "                            "
			<< "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b";

		cout << "Salvando experimento para a regra...";
		salvarRegra(filename, rules[n], k, r, t, transient, width, amostras, init, siteEntropy, entropy, mi, spreading);
		cout << "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b"
			<< "                                   "
			<< "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b";
	}
}

void main() {
	/*
	string dir = "C:\\Users\\arbor\\Source\\Repos\\cellularautomata\\output\\R1.0";
	string filenameRules = "C:\\Users\\arbor\\Source\\Repos\\cellularautomata\\equivalente-rules-r1.0.data";

	Memory<rule_t> rules(88);

	readRules(filenameRules, rules);

	cell_t k = 2;
	float r = 1.0;
	size_t t = 300;
	size_t transient = 200;
	size_t width = 100;
	size_t amostras = 150;

	calculoExperimento(dir, rules, k, r, t, transient, width, amostras);
	*/

	string dir = "F:\\Documents\\Doutorado\\CAClassification\\output\\R1.5";
	string filenameRules = "C:\\Users\\arbor\\Source\\Repos\\cellularautomata\\equivalente-rules-r1.5.data";

	Memory<rule_t> rules(16704);

	readRules(filenameRules, rules);
		
	cell_t k = 2;
	float r = 1.5;
	size_t t = 300;
	size_t transient = 200;
	size_t width = 100;
	size_t amostras = 150;

	calculoExperimento(dir, rules, k, r, t, transient, width, amostras);
}
