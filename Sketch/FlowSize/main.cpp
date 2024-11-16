#include "CMSketch.h"
#include "CMSACSketch.h"
#include "CUSketch.h"
#include "CUSACSketch.h"
#include "PSketch.h"
#include "PSACSketch.h"
#include <fstream>

CMSketch *Sketch = new CMSketch(3, 11092);
std::unordered_map<std::string, uint> actual_size;

int main(){
	/*Insert your code here using the flowsize interface*/

	std::ifstream file("binary.dat",std::ios::binary);
	if(!file){
		std::cout<<"error, file not found!\n";
		return -1;
	}
	unsigned char* buffer = new unsigned char [13];
	memset(buffer,0,13);
	while(file.read(reinterpret_cast<char*>(buffer),13)|| file.gcount() > 0)
	{
		std::string data(reinterpret_cast<char*>(buffer), file.gcount());
		cuc* constData = buffer;
		Sketch->Insert(constData);
		actual_size[data]++;
	}
	file.close();
	file.open("binary.dat",std::ios::binary);
	memset(buffer,0,13);
	FILE* out = fopen("result.txt", "w");
 	// for (auto it = actual_size.begin(); it != actual_size.end(); ++it) 
	// {
    //     Sketch->PrintCounterFile(reinterpret_cast<cuc*>(const_cast<char*>(it->first.c_str())), it->second, out);
    // }
	cuc* path = (unsigned char*)"parameter.txt";
	Sketch->LoadPara(path);
	float AAE = 0;
	while(file.read(reinterpret_cast<char*>(buffer),13)|| file.gcount() > 0)
	{
		std::string data(reinterpret_cast<char*>(buffer), 13);
		cuc* constData = buffer;
		// AAE+=Sketch->CalculateAAE(constData,actual_size[data]);
		uint a = actual_size[data];
		uint query_val = Sketch->Query(constData,true);
		// printf("Actual Size: %u Query Value: %u\n", a, query_val);
		// fprintf_s(out,"Actual Size: %u Query Value: %u\n", a, query_val);
		// AAE+=Sketch->CalculateAAE_ML(constData,a,query_val);
		Sketch->PrintCounterFile(constData,a,out);
	}
	AAE /= actual_size.size();
	std::cout<<"AAE: "<<AAE;
	fclose(out);
	return 0;
}