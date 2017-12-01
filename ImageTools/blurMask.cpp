/*
 * 
 */

#include <iostream>
#include "filtering.h"

using namespace std;

int main(int argc, char** argv) {
	int min_args = 1+3;
	if(argc < min_args) {
		cout << "ERROR - Usage: \n" << "<mask png filepath> <sigma blur(float)> <output filename>" << endl;
	}

	Image mask(argv[1]);
	float sigma = atof(argv[2]);

	Image output = gaussianBlur_separable(mask, sigma);
	string outfilepath = "./../images/mask-images/";
	outfilepath.append(argv[3]);
	output.write(outfilepath);
}